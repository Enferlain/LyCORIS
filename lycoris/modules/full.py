from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LycorisBaseModule
from ..logging import logger
from typing import Optional

# Import ramtorch primitives for CPUBouncingLinear streaming
try:
    from ramtorch.modules.linear import (
        CPUBouncingLinear,
        _get_device_state,
    )
except ImportError:
    CPUBouncingLinear = None
    _get_device_state = None


@cache
def log_cpu_diff_creation():
    logger.info(
        "FullModule is wrapping a RamTorch layer. "
        "Both 'org' and 'diff' weights will reside on CPU and be streamed via RamTorch."
    )


class _CPUStreamedDiffBouncingFn(torch.autograd.Function):
    """
    Autograd function for Full on CPUBouncingLinear with both org and diff on CPU.

    - org and diff weights/bias live on CPU (pinned).
    - Copies both org and diff to GPU on the current stream.
    - Merges org + scale * diff on GPU and does a SINGLE GEMM.
    - Backward computes grad_input and grad_w_diff only; org is frozen.
    """

    @staticmethod
    def forward(ctx, x, w_diff_cpu, b_diff_cpu, w_org_cpu, b_org_cpu, scale):
        # Decide device and compute dtype
        device = x.device if x.is_cuda else torch.device("cuda", torch.cuda.current_device())
        autocast_enabled = torch.is_autocast_enabled()
        compute_dtype = torch.get_autocast_gpu_dtype() if autocast_enabled else x.dtype

        # Allocate GPU buffers and copy from CPU synchronously on current stream
        w_org_gpu = torch.empty_like(w_org_cpu, device=device)
        w_org_gpu.copy_(w_org_cpu, non_blocking=False)

        if w_diff_cpu is not None:
            w_diff_gpu = torch.empty_like(w_diff_cpu, device=device)
            w_diff_gpu.copy_(w_diff_cpu, non_blocking=False)
        else:
            w_diff_gpu = None

        if b_org_cpu is not None:
            b_org_gpu = torch.empty_like(b_org_cpu, device=device)
            b_org_gpu.copy_(b_org_cpu, non_blocking=False)
        else:
            b_org_gpu = None

        if b_diff_cpu is not None:
            b_diff_gpu = torch.empty_like(b_diff_cpu, device=device)
            b_diff_gpu.copy_(b_diff_cpu, non_blocking=False)
        else:
            b_diff_gpu = None

        # Merge on GPU in compute dtype
        s = float(scale)
        x_c = x if x.dtype == compute_dtype else x.to(compute_dtype)
        w_org_c = w_org_gpu.to(compute_dtype)
        w_diff_c = w_diff_gpu.to(compute_dtype) if w_diff_gpu is not None else None

        if w_diff_c is not None:
            w_merged = w_org_c.add(w_diff_c, alpha=s)
        else:
            w_merged = w_org_c

        b_merged = None
        if b_org_gpu is not None:
            b_org_c = b_org_gpu.to(compute_dtype)
            b_merged = b_org_c
            if b_diff_gpu is not None:
                b_merged = b_merged.add(b_diff_gpu.to(compute_dtype), alpha=s)
        elif b_diff_gpu is not None:
            b_merged = b_diff_gpu.to(compute_dtype) * s

        out = F.linear(x_c, w_merged, b_merged)

        # Optional NaN guard (remove once stable)
        # if not torch.isfinite(out).all():
        #     print("NaN/Inf in Full CPU-streamed forward",
        #           "x", x_c.min().item(), x_c.max().item(),
        #           "w_org", w_org_c.min().item(), w_org_c.max().item())
        #     raise RuntimeError("Non-finite in _CPUStreamedDiffBouncingFn.forward")

        # Save minimal state for backward
        ctx.save_for_backward(x, w_diff_cpu)
        ctx.w_org_cpu = w_org_cpu
        ctx.b_diff_present = b_diff_cpu is not None
        ctx.scale = s
        ctx.device = device
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, w_diff_cpu = ctx.saved_tensors
        w_org_cpu = ctx.w_org_cpu
        b_diff_present = ctx.b_diff_present
        scale = ctx.scale
        device = ctx.device

        # Rebuild merged weight in FP32 on GPU
        w_merged = torch.empty_like(w_org_cpu, device=device, dtype=torch.float32)
        w_merged.copy_(w_org_cpu.to(device=device, dtype=torch.float32), non_blocking=False)
        if w_diff_cpu is not None:
            w_merged.add_(w_diff_cpu.to(device=device, dtype=torch.float32), alpha=scale)

        # FP32 math for stability
        grad_out_fp32 = grad_out.to(torch.float32)
        x_fp32 = x.to(torch.float32)

        # grad_input = grad_out @ W_merged  (F.linear uses y = x @ W^T)
        grad_input = grad_out_fp32.matmul(w_merged)

        # grad_W_merged = grad_out^T @ x
        go_2d = grad_out_fp32.flatten(0, -2)
        x_2d = x_fp32.flatten(0, -2)
        grad_w_merged = go_2d.t().matmul(x_2d)

        # grad_w_diff = scale * grad_w_merged  (org is frozen)
        grad_w_diff_gpu = grad_w_merged * scale

        grad_b_diff_gpu = None
        if b_diff_present:
            reduce_dims = tuple(range(grad_out.ndim - 1))
            grad_b_diff_gpu = grad_out_fp32.sum(dim=reduce_dims) * scale

        # Move grads back to CPU to match diff param device/dtype
        grad_w_diff = grad_w_diff_gpu.to(w_diff_cpu.device, dtype=w_diff_cpu.dtype, non_blocking=False)
        if grad_b_diff_gpu is not None:
            grad_b_diff = grad_b_diff_gpu.to(w_diff_cpu.device, dtype=w_diff_cpu.dtype, non_blocking=False)
        else:
            grad_b_diff = None

        # Gradients for (x, w_diff_cpu, b_diff_cpu, w_org_cpu, b_org_cpu, scale)
        return grad_input, grad_w_diff, grad_b_diff, None, None, None


class FullModule(LycorisBaseModule):
    name = "full"
    support_module = {"linear", "conv1d", "conv2d", "conv3d"}
    weight_list = ["diff", "diff_b"]
    weight_list_det = ["diff"]

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=0.0,
        rank_dropout=0.0,
        module_dropout=0.0,
        use_tucker=False,
        use_scalar=False,
        rank_dropout_scale=False,
        bypass_mode=None,
        **kwargs,
    ):
        """
        Full-rank LyCORIS module.

        For CPUBouncingLinear:
        - original weights/bias are snapshotted on CPU (frozen org).
        - diff weights/bias live on pinned CPU and are streamed via ramtorch.
        For other modules, behaves like standard GPU full finetune.
        """
        super().__init__(
            lora_name,
            org_module,
            multiplier,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            rank_dropout_scale=rank_dropout_scale,
            bypass_mode=bypass_mode,
            **kwargs,
        )

        if self.module_type not in self.support_module:
            raise ValueError(f"{self.module_type} is not supported in Full algo.")

        self._use_ramtorch_linear = (
            CPUBouncingLinear is not None
            and _get_device_state is not None
            and isinstance(org_module, CPUBouncingLinear)
            and self.module_type == "linear"
        )

        # Snapshot original weights/bias on CPU (used as 'org' in Full)
        self._org_weight = [self.org_module[0].weight.data.cpu().clone()]
        if self.org_module[0].bias is not None:
            self.org_bias = [self.org_module[0].bias.data.cpu().clone()]
        else:
            self.org_bias = None

        if self._use_ramtorch_linear:
            # Trainable diff params on CPU, pinned for fast H2D
            self.diff = nn.Parameter(
                torch.zeros_like(self._org_weight[0], device="cpu").pin_memory()
            )
            if self.org_bias is not None:
                self.diff_b = nn.Parameter(
                    torch.zeros_like(self.org_bias[0], device="cpu").pin_memory()
                )
            else:
                self.register_parameter("diff_b", None)

            log_cpu_diff_creation()
        else:
            # Non-ramtorch path: standard GPU full finetune
            self.diff = nn.Parameter(torch.zeros_like(org_module.weight))
            if org_module.bias is not None:
                self.diff_b = nn.Parameter(torch.zeros_like(org_module.bias))
            else:
                self.register_parameter("diff_b", None)

    @classmethod
    def make_module_from_state_dict(cls, lora_name, orig_module, diff, diff_b):
        module = cls(lora_name, orig_module, 1)
        module.diff.data.copy_(diff)
        if diff_b is not None and getattr(module, "diff_b", None) is not None:
            module.diff_b.data.copy_(diff_b)
        return module

    def apply_to(self, **kwargs):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    def restore(self):
        self.org_module[0].forward = self.org_forward

    def custom_state_dict(self):
        return {
            "diff": self.diff.data.cpu(),
            "diff_b": self.diff_b.data.cpu() if self.diff_b is not None else None,
        }

    def load_weight_prehook(self, state_dict, prefix, *args, **kwargs):
        for key in ["diff", "diff_b"]:
            full_key = f"{prefix}{key}"
            if full_key in state_dict:
                param = getattr(self, key)
                if param is not None:
                    param.data.copy_(state_dict.pop(full_key))

    def _apply(self, fn):
        # Prevent diff params from moving off CPU when using ramtorch, but allow dtype changes
        if getattr(self, "_use_ramtorch_linear", False):
            probe = torch.empty(0, device="cpu", dtype=self.diff.dtype)
            target_dtype = fn(probe).dtype

            if self.diff is not None and self.diff.dtype != target_dtype:
                self.diff.data = self.diff.data.to(dtype=target_dtype)
            if self.diff_b is not None and self.diff_b.dtype != target_dtype:
                self.diff_b.data = self.diff_b.data.to(dtype=target_dtype)

            for m in self.children():
                m._apply(fn)
            for name, buf in self._buffers.items():
                if buf is not None:
                    self._buffers[name] = fn(buf)
            return self

        return super()._apply(fn)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if self.module_dropout and self.training and torch.rand(1) < self.module_dropout:
            return self.org_forward(x, *args, **kwargs)

        if self._use_ramtorch_linear:
            # org and diff both on CPU; stream via ramtorch and do single GEMM
            return _CPUStreamedDiffBouncingFn.apply(
                x,
                self.diff,
                self.diff_b,
                self._org_weight[0],
                self.org_bias[0] if self.org_bias is not None else None,
                self.multiplier,
            )
        else:
            # Non-ramtorch path: merge on GPU and use standard op
            merged_weight = self.org_module[0].weight + self.multiplier * self.diff
            merged_bias = self.org_module[0].bias
            if merged_bias is not None and self.diff_b is not None:
                merged_bias = merged_bias + self.multiplier * self.diff_b
            return self.op(x, weight=merged_weight, bias=merged_bias, **self.kw_dict)
