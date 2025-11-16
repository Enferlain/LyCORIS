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
            CPUBouncingLinear is not None and _get_device_state is not None
            and isinstance(org_module, CPUBouncingLinear)
            and self.module_type == "linear"
        )

        # NEW: optional direct mode for ramtorch (default True here)
        self.direct_ramtorch = self._use_ramtorch_linear and kwargs.get("direct_ramtorch", True)

        if self.is_quant and not self._use_ramtorch_linear:
            raise ValueError("Quantized Linear is not supported in Full algo.")
        if self.bypass_mode:
            raise ValueError("Bypass mode is not supported in Full algo.")

        # Snapshot base weights/bias on CPU for diff computation
        self._org_weight = [self.org_module[0].weight.data.cpu().clone()]
        if self.org_module[0].bias is not None:
            self.org_bias = [self.org_module[0].bias.data.cpu().clone()]
        else:
            self.org_bias = None

        if self._use_ramtorch_linear and self.direct_ramtorch:
            # Direct mode: train CPUBouncingLinear weights in-place.
            # Register references so optimizer sees them as this module's params.
            self.full_weight = self.org_module[0].weight   # shared Parameter
            if self.org_module[0].bias is not None:
                self.full_bias = self.org_module[0].bias
            else:
                self.register_parameter("full_bias", None)
        elif self._use_ramtorch_linear:
            # Existing diff-based ramtorch path (unchanged)
            self.diff_weight = nn.Parameter(torch.zeros_like(org_module.weight, device="cpu").pin_memory())
            self.diff_weight.is_ramtorch = True
            if org_module.bias is not None:
                self.diff_bias = nn.Parameter(torch.zeros_like(org_module.bias, device="cpu").pin_memory())
                self.diff_bias.is_ramtorch = True
            else:
                self.register_parameter("diff_bias", None)
        else:
            # Non-ramtorch path: standard Full behavior
            self.weight = nn.Parameter(torch.zeros_like(org_module.weight))
            if org_module.bias is not None:
                self.bias = nn.Parameter(torch.zeros_like(org_module.bias))
            else:
                self.register_parameter("bias", None)

        self.is_diff = not (self._use_ramtorch_linear and self.direct_ramtorch)

        @classmethod
        def make_module_from_state_dict(cls, lora_name, orig_module, diff, diff_b):
            module = cls(lora_name, orig_module, 1)

            if module._use_ramtorch_linear and getattr(module, "direct_ramtorch", False):
                # Direct mode: apply diff into CPUBouncingLinear weights
                if diff is not None:
                    new_w = orig_module.weight.data + diff.to(orig_module.weight.device)
                    orig_module.weight.data.copy_(new_w)
                if diff_b is not None and orig_module.bias is not None:
                    new_b = orig_module.bias.data + diff_b.to(orig_module.bias.device)
                    orig_module.bias.data.copy_(new_b)
                module._org_weight[0] = orig_module.weight.data.cpu().clone()
                if orig_module.bias is not None:
                    module.org_bias = [orig_module.bias.data.cpu().clone()]
                return module

            if module._use_ramtorch_linear:
                # RamTorch diff-weight path
                module.diff_weight.data.copy_(diff)
                if diff_b is not None and getattr(module, "diff_bias", None) is not None:
                    module.diff_bias.data.copy_(diff_b)
                return module

            # Non-ramtorch path: original Full behavior
            module.weight.copy_(diff)
            if diff_b is not None:
                if orig_module.bias is not None:
                    module.bias.copy_(diff_b)
                else:
                    module.bias = nn.Parameter(diff_b)
            module.is_diff = True
            return module

    def apply_to(self, **kwargs):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    def restore(self):
        self.org_module[0].forward = self.org_forward

    def custom_state_dict(self):
        # Direct ramtorch mode: compute diff from in-place trained CPUBouncingLinear weights
        if self._use_ramtorch_linear and self.direct_ramtorch:
            base_w = self._org_weight[0]              # CPU snapshot
            cur_w = self.org_module[0].weight.data.cpu()
            diff = cur_w - base_w

            diff_b = None
            if self.org_bias is not None and self.org_module[0].bias is not None:
                base_b = self.org_bias[0]
                cur_b = self.org_module[0].bias.data.cpu()
                diff_b = cur_b - base_b

            sd = {"diff": diff}
            if diff_b is not None:
                sd["diff_b"] = diff_b
            return sd

        # RamTorch diff-weight path
        if self._use_ramtorch_linear:
            sd = {"diff": self.diff_weight.data.cpu()}
            if self.diff_bias is not None:
                sd["diff_b"] = self.diff_bias.data.cpu()
            return sd

        # Non-ramtorch path (original behavior)
        diff = self.weight.data.cpu() - self._org_weight[0].cpu()
        sd = {"diff": diff}
        if self.bias is not None and self.org_bias is not None:
            sd["diff_b"] = self.bias.data.cpu() - self.org_bias[0]
        return sd

    def load_weight_prehook(self, state_dict, prefix, *args, **kwargs):
        if self._use_ramtorch_linear and getattr(self, "direct_ramtorch", False):
            diff_key = f"{prefix}diff"
            diff_b_key = f"{prefix}diff_b"

            if diff_key in state_dict:
                diff = state_dict.pop(diff_key)
                new_w = self._org_weight[0].to(diff.device) + diff
                self.org_module[0].weight.data.copy_(new_w)

            if diff_b_key in state_dict and self.org_bias is not None and self.org_module[0].bias is not None:
                diff_b = state_dict.pop(diff_b_key)
                new_b = self.org_bias[0].to(diff_b.device) + diff_b
                self.org_module[0].bias.data.copy_(new_b)
            return

        # Existing behavior for non-direct ramtorch and non-ramtorch
        if self._use_ramtorch_linear:
            diff = state_dict.pop(f"{prefix}diff")
            self.diff_weight.data.copy_(diff)
            diff_b_key = f"{prefix}diff_b"
            if diff_b_key in state_dict:
                diff_b = state_dict.pop(diff_b_key)
                if self.diff_bias is not None:
                    self.diff_bias.data.copy_(diff_b)
        else:
            diff = state_dict.pop(f"{prefix}diff")
            self.weight.data.copy_(self._org_weight[0].to(diff.device) + diff)
            diff_b_key = f"{prefix}diff_b"
            if diff_b_key in state_dict:
                diff_b = state_dict.pop(diff_b_key)
                if self.bias is not None:
                    self.bias.data.copy_(self.org_bias[0].to(diff_b.device) + diff_b)

    def _apply(self, fn):
        # 1) Ramtorch direct mode: protect CPUBouncingLinear params from device moves
        if getattr(self, "_use_ramtorch_linear", False) and getattr(self, "direct_ramtorch", False):
            # Underlying CPUBouncingLinear
            linear = self.org_module[0]
            w = linear.weight
            b = linear.bias

            # Mimic CPUBouncingLinear._apply: allow dtype change, keep device=cpu
            dummy = torch.tensor(0.0, device="cpu", dtype=w.dtype)
            result = fn(dummy)
            target_dtype = result.dtype

            if w.dtype != target_dtype:
                w.data = w.data.to(dtype=target_dtype)
            if b is not None and b.dtype != target_dtype:
                b.data = b.data.to(dtype=target_dtype)

            # Apply fn only to children/buffers, not to CPUBouncingLinear params
            for m in self.children():
                # org_module[0] will handle its own _apply if needed
                if m is not linear:
                    m._apply(fn)
            for name, buf in self._buffers.items():
                if buf is not None:
                    self._buffers[name] = fn(buf)
            return self

        # 2) Ramtorch diff-based mode: keep existing CPU-pinned diff behavior
        if getattr(self, "_use_ramtorch_linear", False) and not getattr(self, "direct_ramtorch", False):
            probe = torch.empty(0, device="cpu", dtype=self.diff_weight.dtype)
            target_dtype = fn(probe).dtype

            if self.diff_weight is not None and self.diff_weight.dtype != target_dtype:
                self.diff_weight.data = self.diff_weight.data.to(dtype=target_dtype)
            if getattr(self, "diff_bias", None) is not None and self.diff_bias.dtype != target_dtype:
                self.diff_bias.data = self.diff_bias.data.to(dtype=target_dtype)

            for m in self.children():
                m._apply(fn)
            for name, buf in self._buffers.items():
                if buf is not None:
                    self._buffers[name] = fn(buf)
            return self

        # 3) Non-ramtorch path: default behavior
        return super()._apply(fn)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if self.module_dropout and self.training and torch.rand(1) < self.module_dropout:
            return self.org_forward(x, *args, **kwargs)

        if self._use_ramtorch_linear and self.direct_ramtorch:
            # Direct mode: pure CPUBouncingLinear forward, no custom autograd.
            return self.org_forward(x, *args, **kwargs)

        if self._use_ramtorch_linear:
            # Existing diff-based ramtorch path
            return _BouncingLinearDiffMixFn.apply(
                x,
                self._org_weight[0],
                self.org_bias[0] if self.org_bias is not None else None,
                self.diff_weight,
                self.diff_bias,
                self.multiplier,
                torch.cuda.current_device(),
            )
        else:
            weight, bias = self.get_merged_weight(self.multiplier, device=x.device)
            kw_dict = self.kw_dict | {"weight": weight, "bias": bias}
            return self.op(x, **kw_dict)

    def get_merged_weight(self, multiplier=1.0, shape=None, device=None):
        # This is the original Full implementation from upstream LyCORIS.
        # Used only in the non-ramtorch path (conv/linear on GPU).
        if self.is_diff:
            # weight/bias store only the diff; org is in _org_weight/org_bias.
            diff_w = self.weight * multiplier
            diff_b = self.bias * multiplier if self.bias is not None else None

            org_w = self._org_weight[0].to(device, dtype=diff_w.dtype)
            weight = org_w + diff_w

            bias = None
            if self.org_bias is not None:
                org_b = self.org_bias[0].to(device)
                if diff_b is not None:
                    bias = org_b + diff_b
                else:
                    bias = org_b
            return weight, bias
        else:
            # weight/bias already contain merged weights; scale only the diff part.
            org_w = self._org_weight[0].to(self.weight.device, dtype=self.weight.dtype)
            diff = self.weight - org_w
            weight = org_w + diff * multiplier

            bias = None
            if self.bias is not None and self.org_bias is not None:
                org_b = self.org_bias[0].to(self.bias.device, dtype=self.bias.dtype)
                diff_b = self.bias - org_b
                bias = org_b + diff_b * multiplier

            return weight, bias
