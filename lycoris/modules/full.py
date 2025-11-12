from functools import cache

import torch
import torch.nn as nn

from .base import LycorisBaseModule
from ..logging import logger

from typing import Optional

# Import ramtorch linear layer and its internal helpers for compatibility
try:
    from ramtorch.modules.linear import (
        CPUBouncingLinear,
        _get_device_state,
        _invoke_tensor_hooks,
        _invoke_post_accum_tensor_hooks,
        _invoke_zero_2_tensor_hooks,
    )
except ImportError:
    CPUBouncingLinear = None
    _get_device_state = None
    _invoke_tensor_hooks = None
    _invoke_post_accum_tensor_hooks = None
    _invoke_zero_2_tensor_hooks = None


@cache
def log_bypass_override():
    return logger.warning(
        "Automatic Bypass-Mode detected in algo=full, "
        "override with bypass_mode=False since algo=full not support bypass mode. "
        "If you are using quantized model which require bypass mode, please don't use algo=full. "
    )


class _BouncingLinearDiffMixFn(torch.autograd.Function):
    """
    Custom autograd function for memory-efficient mixed streaming.
    It sequentially streams a base weight and a diff weight into a single
    GPU buffer slot to compute a blended output, respecting autocast dtypes.
    """
    @staticmethod
    def forward(ctx, x, w_org_cpu, b_org_cpu, w_diff_cpu, b_diff_cpu, scale, device="cuda"):
        autocast_enabled = torch.is_autocast_enabled()
        autocast_dtype = torch.get_autocast_gpu_dtype() if autocast_enabled else None

        state = _get_device_state(device)
        ts = state["transfer_stream"]
        tf_event = state["transfer_forward_finished_event"]
        cs_start = state["compute_forward_start_event"]
        w_buf = state["w_buffers"]
        b_buf = state["b_buffers"]

        idx = state["forward_clk"]
        state["forward_clk"] ^= 1
        
        compute_dtype = autocast_dtype if autocast_enabled else x.dtype
        x_c = x.to(compute_dtype)

        # 1. Stream W_org and compute base output
        with torch.cuda.stream(ts):
            ts.wait_event(cs_start)
            w_buf[idx] = w_org_cpu.to(device, non_blocking=True)
            b_buf[idx] = b_org_cpu.to(device, non_blocking=True) if b_org_cpu is not None else None
            tf_event.record()
        torch.cuda.current_stream().wait_event(tf_event)
        cs_start.record()
        
        w_c = w_buf[idx].to(compute_dtype)
        b_c = b_buf[idx].to(compute_dtype) if b_buf[idx] is not None else None
        y = torch.nn.functional.linear(x_c, w_c, b_c)

        # 2. Stream W_diff into the *same* buffer and compute delta
        with torch.cuda.stream(ts):
            ts.wait_event(cs_start)
            w_buf[idx] = w_diff_cpu.to(device, non_blocking=True)
            b_buf[idx] = b_diff_cpu.to(device, non_blocking=True) if b_diff_cpu is not None else None
            tf_event.record()
        torch.cuda.current_stream().wait_event(tf_event)
        cs_start.record()

        w_c = w_buf[idx].to(compute_dtype)
        b_c = b_buf[idx].to(compute_dtype) if b_buf[idx] is not None else None
        y_delta = torch.nn.functional.linear(x_c, w_c, b_c)

        # 3) Blend and cast back appropriately
        y.add_(y_delta, alpha=float(scale))
        if torch.is_autocast_enabled():
            y = y.to(torch.get_autocast_gpu_dtype())
        else:
            y = y.to(x.dtype)

        # Save tensors for backward
        b_org_sentinel = x.new_zeros(0) if b_org_cpu is None else b_org_cpu
        b_diff_sentinel = x.new_zeros(0) if b_diff_cpu is None else b_diff_cpu
        ctx.save_for_backward(x, w_org_cpu, b_org_sentinel, w_diff_cpu, b_diff_sentinel)
        ctx.device = device
        ctx.scale = float(scale)
        # Optional: if you want backward to know whether autocast was on
        ctx.autocast_enabled = torch.is_autocast_enabled()
        ctx.autocast_dtype = torch.get_autocast_gpu_dtype() if ctx.autocast_enabled else None
        return y

    @staticmethod
    def backward(ctx, grad_out):
        # Unpack all saved tensors (bias entries may be empty sentinels)
        x, w_org_cpu, b_org_s, w_diff_cpu, b_diff_s = ctx.saved_tensors
        b_org_cpu  = None if b_org_s.numel()  == 0 else b_org_s
        b_diff_cpu = None if b_diff_s.numel() == 0 else b_diff_s

        device, s = ctx.device, ctx.scale

        state = _get_device_state(device)
        ts, tgs = state["transfer_stream"], state["transfer_grad_stream"]
        wbw, wg, bg = state["w_bwd_buffers"], state["w_grad_buffers"], state["b_grad_buffers"]
        wga, bga = state["w_grad_accum_buffers"], state["b_grad_accum_buffers"]
        tb_event = state["transfer_backward_finished_event"]
        cs_start, cs_done = state["compute_backward_start_event"], state["compute_backward_finished_event"]
        twb_event = state["transfer_weight_backward_finished_event"]

        idx = state["backward_clk"]
        state["backward_clk"] ^= 1

        with torch.cuda.stream(ts):
            ts.wait_event(cs_start)
            wbw[idx] = w_diff_cpu.to(device, non_blocking=True)
            wga[idx] = w_diff_cpu.grad.to(device, non_blocking=True) if w_diff_cpu.grad is not None else None
            bga[idx] = b_diff_cpu.grad.to(device, non_blocking=True) if (b_diff_cpu is not None and b_diff_cpu.grad is not None) else None
            tb_event.record()
        torch.cuda.current_stream().wait_event(tb_event)
        cs_start.record()

        # ND-safe grad_input; keep in grad_out.dtype
        w_org_gpu = w_org_cpu.to(device, non_blocking=True, dtype=grad_out.dtype)
        grad_input = grad_out @ w_org_gpu
        w_diff_gpu = wbw[idx].to(dtype=grad_out.dtype)
        grad_input.add_(grad_out @ w_diff_gpu, alpha=s)

        # Weight grad in compute dtype, then cast to param dtype
        go_2d = grad_out.flatten(0, -2)
        x_2d  = x.to(grad_out.dtype).flatten(0, -2)
        wg[idx] = (go_2d.T @ x_2d)
        wg[idx].mul_(s)

        # Bias grad: compute in compute dtype (or fp32), then cast to param dtype
        if b_diff_cpu is not None:
            reduce_dims = tuple(range(grad_out.ndim - 1))
            bg[idx] = grad_out.sum(dim=reduce_dims)
            bg[idx].mul_(s)
        else:
            bg[idx] = None

        # Cast grads to parameter dtypes before hooks/accum
        target_w_dtype = w_diff_cpu.dtype
        wg[idx] = wg[idx].to(target_w_dtype)
        if wga[idx] is not None:
            wg[idx].add_(wga[idx].to(target_w_dtype))

        if b_diff_cpu is not None:
            target_b_dtype = b_diff_cpu.dtype
            bg[idx] = bg[idx].to(target_b_dtype)
            if bga[idx] is not None:
                bg[idx].add_(bga[idx].to(target_b_dtype))

        cs_done.record()

        # Hook invocation and CPU assignment
        with torch.cuda.stream(tgs):
            tgs.wait_event(cs_done)

            # Optional: ZeRO-2 hooks on GPU grads (match ramtorch order)
            if _invoke_zero_2_tensor_hooks is not None:
                wg[idx] = _invoke_zero_2_tensor_hooks(w_diff_cpu, wg[idx])
                if b_diff_cpu is not None:
                    bg[idx] = _invoke_zero_2_tensor_hooks(b_diff_cpu, bg[idx])

            # Backward hooks then post-accum hooks
            wg[idx] = _invoke_tensor_hooks(w_diff_cpu, wg[idx])
            w_diff_cpu.ramtorch_grad = wg[idx]
            _invoke_post_accum_tensor_hooks(w_diff_cpu)
            del w_diff_cpu.ramtorch_grad

            if b_diff_cpu is not None:
                bg[idx] = _invoke_tensor_hooks(b_diff_cpu, bg[idx])
                b_diff_cpu.ramtorch_grad = bg[idx]
                _invoke_post_accum_tensor_hooks(b_diff_cpu)
                del b_diff_cpu.ramtorch_grad

            # Just before final CPU assignment in backward
            if w_diff_cpu.is_cuda:
                raise RuntimeError("ramtorch diff param moved to CUDA; ensure FullModule._apply keeps it on CPU")

            # Final CPU assignment; dtypes now match parameter dtypes
            w_diff_cpu.grad = wg[idx].to("cpu", non_blocking=True)
            if b_diff_cpu is not None:
                b_diff_cpu.grad = bg[idx].to("cpu", non_blocking=True)

            twb_event.record()

        return grad_input, None, None, None, None, None, None


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
        ggpo_beta: Optional[float] = None,
        ggpo_sigma: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            lora_name, org_module, multiplier, dropout, rank_dropout,
            module_dropout, rank_dropout_scale, bypass_mode, ggpo_beta, ggpo_sigma, **kwargs
        )
        if self.module_type not in self.support_module:
            raise ValueError(f"{self.module_type} is not supported in Full algo.")

        self._use_ramtorch_linear = (
            CPUBouncingLinear is not None and _get_device_state is not None
            and isinstance(org_module, CPUBouncingLinear)
            and self.module_type == "linear"
        )

        if self.is_quant and not self._use_ramtorch_linear:
            raise ValueError("Quantized Linear is not supported in Full algo.")
        if self.bypass_mode:
            raise ValueError("Bypass mode is not supported in Full algo.")

        if self._use_ramtorch_linear:
            self.diff_weight = nn.Parameter(torch.zeros_like(org_module.weight, device="cpu").pin_memory())
            self.diff_weight.is_ramtorch = True
            if org_module.bias is not None:
                self.diff_bias = nn.Parameter(torch.zeros_like(org_module.bias, device="cpu").pin_memory())
                self.diff_bias.is_ramtorch = True
            else:
                self.register_parameter("diff_bias", None)
        else:
            self.weight = nn.Parameter(torch.zeros_like(org_module.weight))
            if org_module.bias is not None:
                self.bias = nn.Parameter(torch.zeros_like(org_module.bias))
            else:
                self.register_parameter("bias", None)

        self.is_diff = True
        self._org_weight = [self.org_module[0].weight.data.cpu().clone()]
        if self.org_module[0].bias is not None:
            self.org_bias = [self.org_module[0].bias.data.cpu().clone()]
        else:
            self.org_bias = None

    @classmethod
    def make_module_from_state_dict(cls, lora_name, orig_module, diff, diff_b):
        module = cls(lora_name, orig_module, 1)
        if module._use_ramtorch_linear:
            module.diff_weight.data.copy_(diff)
            if diff_b is not None and module.diff_bias is not None:
                module.diff_bias.data.copy_(diff_b)
        else:
            module.weight.data.copy_(orig_module.weight.data + diff)
            if diff_b is not None and module.bias is not None:
                module.bias.data.copy_(orig_module.bias.data + diff_b)
        return module

    def _apply(self, fn):
        # Prevent diff params from being moved off CPU by .to()/prepare()
        if getattr(self, "_use_ramtorch_linear", False):
            # Probe target transform
            probe = torch.empty(0, device="cpu", dtype=self.diff_weight.dtype)
            out = fn(probe)
            target_dtype = out.dtype  # target device is ignored for diff params

            # Allow dtype change on CPU-pinned tensors
            if self.diff_weight is not None and self.diff_weight.dtype != target_dtype:
                self.diff_weight.data = self.diff_weight.data.to(dtype=target_dtype)
            if getattr(self, "diff_bias", None) is not None and self.diff_bias.dtype != target_dtype:
                self.diff_bias.data = self.diff_bias.data.to(dtype=target_dtype)

            # Apply transform to child modules/buffers (there shouldn't be parameters to move here)
            for m in self.children():
                m._apply(fn)
            for name, buf in self._buffers.items():
                if buf is not None:
                    self._buffers[name] = fn(buf)
            return self
        # Non-ramtorch path: default behavior
        return super()._apply(fn)

    def apply_to(self, **kwargs):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward
        if not self._use_ramtorch_linear:
            self.weight.data.add_(self.org_module[0].weight.data)
            if self.org_module[0].bias is not None and self.bias is not None:
                self.bias.data.add_(self.org_module[0].bias.data)
            self.is_diff = False
            delattr(self.org_module[0], "weight")
            if self.org_module[0].bias is not None:
                delattr(self.org_module[0], "bias")

    def restore(self):
        self.org_module[0].forward = self.org_forward
        if not self._use_ramtorch_linear:
            self.org_module[0].weight = nn.Parameter(self._org_weight[0].to(self.weight.device))
            if self.org_bias is not None and self.bias is not None:
                self.org_module[0].bias = nn.Parameter(self.org_bias[0].to(self.bias.device))

    def custom_state_dict(self):
        if self._use_ramtorch_linear:
            return {"diff": self.diff_weight.data.cpu(), "diff_b": self.diff_bias.data.cpu() if self.diff_bias is not None else None}
        else:
            diff = self.weight.data.cpu() - self._org_weight[0].cpu()
            diff_b = None
            if self.bias is not None and self.org_bias is not None:
                diff_b = self.bias.data.cpu() - self.org_bias[0].cpu()
            return {"diff": diff, "diff_b": diff_b}

    def load_weight_prehook(self, state_dict, prefix, *args, **kwargs):
        if self._use_ramtorch_linear:
            diff = state_dict.pop(f"{prefix}diff")
            self.diff_weight.data.copy_(diff)
            if f"{prefix}diff_b" in state_dict:
                diff_b = state_dict.pop(f"{prefix}diff_b")
                if self.diff_bias is not None: self.diff_bias.data.copy_(diff_b)
        else:
            diff = state_dict.pop(f"{prefix}diff")
            self.weight.data.copy_(self._org_weight[0].to(diff.device) + diff)
            if f"{prefix}diff_b" in state_dict:
                diff_b = state_dict.pop(f"{prefix}diff_b")
                if self.bias is not None: self.bias.data.copy_(self.org_bias[0].to(diff_b.device) + diff_b)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if self.module_dropout and self.training and torch.rand(1) < self.module_dropout:
            return self.org_forward(x, *args, **kwargs)

        if self._use_ramtorch_linear:
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
            # Non-ramtorch path for conv layers
            weight, bias = self.get_merged_weight(self.multiplier, device=x.device)
            kw_dict = self.kw_dict | {"weight": weight, "bias": bias}
            return self.op(x, **kw_dict)

    def get_merged_weight(self, multiplier=1.0, shape=None, device=None):
        if self.is_diff:
            diff_w = self.weight * multiplier
            diff_b = self.bias * multiplier if self.bias is not None else None
            org_w = self._org_weight[0].to(device, dtype=diff_w.dtype)
            weight = org_w + diff_w
            bias = None
            if self.org_bias is not None:
                org_b = self.org_bias[0].to(device)
                if diff_b is not None: bias = org_b + diff_b
                else: bias = org_b
            return weight, bias
        else:
            org_w = self._org_weight[0].to(self.weight.device, dtype=self.weight.dtype)
            diff = self.weight - org_w
            weight = org_w + diff * multiplier
            bias = None
            if self.bias is not None and self.org_bias is not None:
                org_b = self.org_bias[0].to(self.bias.device, dtype=self.bias.dtype)
                diff_b = self.bias - org_b
                bias = org_b + diff_b * multiplier
            return weight, bias