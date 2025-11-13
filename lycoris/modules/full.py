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
    Mixed streaming for Full on CPUBouncingLinear:
    - Reuse pinned CPU params (no clones)
    - Use two ping-pong slots per call (idx, other)
    - Overlap W_diff copy with base GEMM
    - Stage backward copies on transfer streams
    - Preserve ramtorch hook order
    """
    def forward(ctx, x, w_org_cpu, b_org_cpu, w_diff_cpu, b_diff_cpu, scale, device="cuda"):
        autocast_enabled = torch.is_autocast_enabled()
        compute_dtype = torch.get_autocast_gpu_dtype() if autocast_enabled else x.dtype
        x_c = x.to(compute_dtype) if x.dtype != compute_dtype else x

        state = _get_device_state(device)
        ts = state["transfer_stream"]
        cs_start = state["compute_forward_start_event"]
        w_bufs = state["w_buffers"]
        b_bufs = state["b_buffers"]

        # Per-call events to avoid reuse races
        base_done = torch.cuda.Event()
        diff_done = torch.cuda.Event()

        def _ensure_slot(slot_t, like_cpu, dev):
            if slot_t is None or slot_t.shape != like_cpu.shape or slot_t.dtype != like_cpu.dtype or slot_t.device.index != torch.cuda.current_device():
                return torch.empty_like(like_cpu, device=dev)
            return slot_t

        # Use only one global slot per call to avoid cross-layer contention
        idx = state["forward_clk"]
        state["forward_clk"] ^= 1

        # Base transfer on transfer stream, gated by cs_start
        with torch.cuda.stream(ts):
            ts.wait_event(cs_start)
            w_bufs[idx] = _ensure_slot(w_bufs[idx], w_org_cpu, device)
            w_bufs[idx].copy_(w_org_cpu, non_blocking=True)
            if b_org_cpu is not None:
                b_bufs[idx] = _ensure_slot(b_bufs[idx], b_org_cpu, device)
                b_bufs[idx].copy_(b_org_cpu, non_blocking=True)
            else:
                b_bufs[idx] = None
            base_done.record()

        # Compute waits only for base transfer; record cs_start to release others
        comp = torch.cuda.current_stream()
        comp.wait_event(base_done)
        cs_start.record()
        w_base = w_bufs[idx] if w_bufs[idx].dtype == compute_dtype else w_bufs[idx].to(compute_dtype)
        b_base = None if b_bufs[idx] is None else (b_bufs[idx] if b_bufs[idx].dtype == compute_dtype else b_bufs[idx].to(compute_dtype))
        y = torch.nn.functional.linear(x_c, w_base, b_base)

        # Diff transfer reuses the same slot after base compute started
        with torch.cuda.stream(ts):
            ts.wait_event(cs_start)
            w_bufs[idx].copy_(w_diff_cpu, non_blocking=True)
            if b_diff_cpu is not None:
                if b_bufs[idx] is None or b_bufs[idx].shape != b_diff_cpu.shape or b_bufs[idx].dtype != b_diff_cpu.dtype:
                    b_bufs[idx] = torch.empty_like(b_diff_cpu, device=device)
                b_bufs[idx].copy_(b_diff_cpu, non_blocking=True)
            else:
                b_bufs[idx] = None
            diff_done.record()

        # Compute waits for diff transfer; record cs_start again for downstream
        comp.wait_event(diff_done)
        cs_start.record()
        w_diff = w_bufs[idx] if w_bufs[idx].dtype == compute_dtype else w_bufs[idx].to(compute_dtype)
        b_diff = None if b_bufs[idx] is None else (b_bufs[idx] if b_bufs[idx].dtype == compute_dtype else b_bufs[idx].to(compute_dtype))
        y_delta = torch.nn.functional.linear(x_c, w_diff, b_diff)
        y.add_(y_delta, alpha=float(scale))

        # Save for backward
        b_org_s = x.new_zeros(0) if b_org_cpu is None else b_org_cpu
        b_diff_s = x.new_zeros(0) if b_diff_cpu is None else b_diff_cpu
        ctx.save_for_backward(x, w_org_cpu, b_org_s, w_diff_cpu, b_diff_s)
        ctx.device = device
        ctx.scale = float(scale)
        ctx.compute_dtype = compute_dtype
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, w_org_cpu, b_org_s, w_diff_cpu, b_diff_s = ctx.saved_tensors
        b_org_cpu = None if b_org_s.numel() == 0 else b_org_s
        b_diff_cpu = None if b_diff_s.numel() == 0 else b_diff_s

        device, s = ctx.device, ctx.scale
        state = _get_device_state(device)
        ts = state["transfer_stream"]
        tgs = state["transfer_grad_stream"]
        cs_start = state["compute_backward_start_event"]
        cs_done = state["compute_backward_finished_event"]

        wbw = state["w_bwd_buffers"]
        wg_bufs = state["w_grad_buffers"]
        bg_bufs = state["b_grad_buffers"]
        wga = state["w_grad_accum_buffers"]
        bga = state["b_grad_accum_buffers"]

        base_bwd_done = torch.cuda.Event()
        diff_bwd_done = torch.cuda.Event()

        def _ensure(shape, dtype, buf):
            if buf is None or tuple(buf.shape) != tuple(shape) or buf.dtype != dtype or buf.device.index != torch.cuda.current_device():
                return torch.empty(shape, dtype=dtype, device=device)
            return buf

        # Single slot for staging
        idx = state["backward_clk"]
        state["backward_clk"] ^= 1

        # Stage W_org first
        with torch.cuda.stream(ts):
            ts.wait_event(cs_start)
            wbw[idx] = _ensure(w_org_cpu.shape, grad_out.dtype, wbw[idx])
            wbw[idx].copy_(w_org_cpu.to(dtype=grad_out.dtype), non_blocking=True)
            base_bwd_done.record()

        comp = torch.cuda.current_stream()
        comp.wait_event(base_bwd_done)
        cs_start.record()

        # Start building grad_input with W_org
        grad_input = grad_out.matmul(wbw[idx])

        # Stage W_diff into same slot (overwrite) and finish grad_input
        with torch.cuda.stream(ts):
            ts.wait_event(cs_start)
            wbw[idx].copy_(w_diff_cpu.to(dtype=grad_out.dtype), non_blocking=True)
            diff_bwd_done.record()

        comp.wait_event(diff_bwd_done)
        cs_start.record()
        grad_input.add_(grad_out.matmul(wbw[idx]), alpha=s)

        # Compute diff grads in FP32 for stability, then cast back
        go_2d = grad_out.flatten(0, -2).to(torch.float32)
        x_2d = x.flatten(0, -2).to(torch.float32)
        w_grad_32 = go_2d.t().matmul(x_2d)
        if s != 1.0:
            w_grad_32.mul_(s)

        wg_bufs[idx] = _ensure(w_diff_cpu.shape, torch.float32, wg_bufs[idx])
        wg_bufs[idx].copy_(w_grad_32, non_blocking=True)

        if b_diff_cpu is not None:
            reduce_dims = tuple(range(grad_out.ndim - 1))
            b_grad_32 = grad_out.sum(dim=reduce_dims).to(torch.float32)
            if s != 1.0:
                b_grad_32.mul_(s)
            bg_bufs[idx] = _ensure((b_grad_32.shape[-1],), torch.float32, bg_bufs[idx])
            bg_bufs[idx].copy_(b_grad_32, non_blocking=True)
        else:
            bg_bufs[idx] = None

        # Cast to param dtype and add any staged accum of matching shape
        tw_dtype = w_diff_cpu.dtype
        if wg_bufs[idx].dtype != tw_dtype:
            wg_bufs[idx] = wg_bufs[idx].to(tw_dtype)
        if wga[idx] is not None and tuple(wga[idx].shape) == tuple(wg_bufs[idx].shape):
            wg_bufs[idx].add_(wga[idx].to(tw_dtype))

        if b_diff_cpu is not None:
            tb_dtype = b_diff_cpu.dtype
            if bg_bufs[idx].dtype != tb_dtype:
                bg_bufs[idx] = bg_bufs[idx].to(tb_dtype)
            if bga[idx] is not None and tuple(bga[idx].shape) == tuple(bg_bufs[idx].shape):
                bg_bufs[idx].add_(bga[idx].to(tb_dtype))

        cs_done.record()

        # Hook order and async CPU assignment on grad stream
        with torch.cuda.stream(tgs):
            tgs.wait_event(cs_done)

            if _invoke_zero_2_tensor_hooks is not None:
                wg_bufs[idx] = _invoke_zero_2_tensor_hooks(w_diff_cpu, wg_bufs[idx])
                if b_diff_cpu is not None:
                    bg_bufs[idx] = _invoke_zero_2_tensor_hooks(b_diff_cpu, bg_bufs[idx])

            wg_bufs[idx] = _invoke_tensor_hooks(w_diff_cpu, wg_bufs[idx])
            w_diff_cpu.ramtorch_grad = wg_bufs[idx]
            _invoke_post_accum_tensor_hooks(w_diff_cpu)
            del w_diff_cpu.ramtorch_grad

            if b_diff_cpu is not None:
                bg_bufs[idx] = _invoke_tensor_hooks(b_diff_cpu, bg_bufs[idx])
                b_diff_cpu.ramtorch_grad = bg_bufs[idx]
                _invoke_post_accum_tensor_hooks(b_diff_cpu)
                del b_diff_cpu.ramtorch_grad

            if w_diff_cpu.is_cuda:
                raise RuntimeError("ramtorch diff param moved to CUDA; keep diff params on CPU")

            w_diff_cpu.grad = wg_bufs[idx].to("cpu", non_blocking=True)
            if b_diff_cpu is not None:
                b_diff_cpu.grad = bg_bufs[idx].to("cpu", non_blocking=True)

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
        super().__init__(lora_name, org_module, multiplier, dropout, rank_dropout,
                         module_dropout, rank_dropout_scale, bypass_mode, ggpo_beta, ggpo_sigma, **kwargs)
        if self.module_type not in self.support_module:
            raise ValueError(f"{self.module_type} is not supported in Full algo.")

        self._use_ramtorch_linear = (
            CPUBouncingLinear is not None and _get_device_state is not None
            and isinstance(org_module, CPUBouncingLinear) and self.module_type == "linear"
        )

        if self.is_quant and not self._use_ramtorch_linear:
            raise ValueError("Quantized Linear is not supported in Full algo.")
        if self.bypass_mode:
            raise ValueError("Bypass mode is not supported in Full algo.")

        if self._use_ramtorch_linear:
            # Pinned diff params on CPU; mark for ramtorch
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

        # IMPORTANT: keep direct references to ramtorch's pinned base params, not CPU clones
        if self._use_ramtorch_linear:
            self._org_weight = [self.org_module[0].weight]  # CPUBouncingLinear param on pinned CPU
            self.org_bias = [self.org_module[0].bias] if self.org_module[0].bias is not None else None
        else:
            self.is_diff = False  # non-ramtorch path will merge into weight/bias
            self._org_weight = [self.org_module[0].weight.data.cpu().clone()]
            self.org_bias = [self.org_module[0].bias.data.cpu().clone()] if self.org_module[0].bias is not None else None

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
        # Prevent diff params from moving off CPU; allow dtype changes
        if getattr(self, "_use_ramtorch_linear", False):
            probe = torch.empty(0, device="cpu", dtype=self.diff_weight.dtype)
            out = fn(probe)
            target_dtype = out.dtype

            if self.diff_weight is not None and self.diff_weight.dtype != target_dtype:
                self.diff_weight.data = self.diff_weight.data.to(dtype=target_dtype)
            if getattr(self, "diff_bias", None) is not None and self.diff_bias.dtype != target_dtype:
                self.diff_bias.data = self.diff_bias.data.to(dtype=target_dtype)

            # children/buffers transform only
            for m in self.children():
                m._apply(fn)
            for name, buf in self._buffers.items():
                if buf is not None:
                    self._buffers[name] = fn(buf)
            return self
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