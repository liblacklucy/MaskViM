import warnings
import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

from mmcv.cnn import ConvModule

from mmengine.model import ModuleList, Sequential, BaseModule

from mmseg.registry import MODELS as MODELS_MMSEG
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.decode_heads.psp_head import PPM
from mmseg.models.losses import accuracy, CrossEntropyLoss
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.utils import add_prefix


DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

try:
    "sscore acts the same as mamba_ssm"
    SSMODE = "sscore"
    import selective_scan_cuda_core
except Exception as e:
    print(e, flush=True)
    "you should install mamba_ssm to use this"
    SSMODE = "mamba_ssm"
    import selective_scan_cuda
    # from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref


# fvcore flops =======================================

def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    assert not with_complex
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


# this is only for selective_scan_ref...
def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # below code flops = 0

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try:
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


# cross selective scan ===============================

class SelectiveScan(torch.autograd.Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
        assert nrows in [1, 2, 3, 4], f"{nrows}"  # 8+ is too slow to compile
        assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows
        # all in float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None and D.stride(-1) != 1:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True

        if SSMODE == "mamba_ssm":
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        else:
            out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        if SSMODE == "mamba_ssm":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
                False  # option to recompute out_z, not used here
            )
        else:
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
                # u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows,
            )

        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)


def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False, with_Group=True)
    return flops


# =====================================================

class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


############################################################### EXP1 ###############################################################
#TODO: SS in sequence -> SS around(done)
############################################################### EXP1 ###############################################################
# V1: no stride
class AroundScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        x_around = F.unfold(x, kernel_size=3, padding=1).contiguous().view(B, -1, 9, H*W).permute(0, 2, 1, 3).contiguous()
        return x_around  # (B, 9, C, L)

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        y = F.fold(ys.permute(0, 2, 1, 3).contiguous().view(B, -1, H * W), output_size=(H, W), kernel_size=3, padding=1)
        return y  # (B, C, H, W)


class AroundMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        y = F.fold(ys.contiguous().view(B, K, D, -1).permute(0, 2, 1, 3).contiguous().view(B, -1, H * W), output_size=(H, W), kernel_size=3, padding=1).contiguous().view(B, D, -1)
        return y  # (B, C, L)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        x_around = F.unfold(x.contiguous().view(B, C, H, W), kernel_size=3, padding=1).contiguous().view(B, -1, 9, H * W).permute(0, 2, 1, 3).contiguous().view(B, 9, C, H, W)
        return x_around  # (B, 9, C, H, W)


def around_selective_scan(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        out_norm: torch.nn.Module = None,
        nrows=-1,
        delta_softplus=True,
        to_dtype=True,
        force_fp32=True,
):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    if nrows < 1:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1

    xs = AroundScan.apply(x)

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.contiguous().view(B, -1, L)  # V1
    dts = dts.contiguous().view(B, -1, L)  # V1
    As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)  # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
    ).view(B, K, -1, H, W)  # V1

    y: torch.Tensor = AroundMerge.apply(ys)
    y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
    y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)


class SSExp1(nn.Module):
    """Modified based on VSSM code"""
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            simple_init=False,
            # ======================
            forward_type="v2",
            # ======================
            **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv

        # disable z act ======================================
        self.disable_z_act = forward_type[-len("nozact"):] == "nozact"
        if self.disable_z_act:
            forward_type = forward_type[:-len("nozact")]

        # softmax | sigmoid | norm ===========================
        if forward_type[-len("softmax"):] == "softmax":
            forward_type = forward_type[:-len("softmax")]
            self.out_norm = nn.Softmax(dim=1)
        elif forward_type[-len("sigmoid"):] == "sigmoid":
            forward_type = forward_type[:-len("sigmoid")]
            self.out_norm = nn.Sigmoid()
        else:
            self.out_norm = nn.LayerNorm(d_inner)

        # forward_type =======================================
        self.forward_core = dict(
            v0=self.forward_corev0,
            v0_seq=self.forward_corev0_seq,
            v1=self.forward_corev2,
            v2=self.forward_corev2,
            share_ssm=self.forward_corev0_share_ssm,
            share_a=self.forward_corev0_share_a,
        ).get(forward_type, self.forward_corev2)
        # self.K = 4 if forward_type not in ["share_ssm"] else 1
        # self.K2 = self.K if forward_type not in ["share_a"] else 1
        self.K = 9 if forward_type not in ["share_ssm"] else 1  # SS around
        self.K2 = self.K if forward_type not in ["share_a"] else 1

        # in proj =======================================
        self.in_proj = nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True)  # (K * D, N) | (K * inner, N)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True)  # (K * D)

        # out proj =======================================
        self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if simple_init:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
            self.A_logs = nn.Parameter(
                torch.randn((self.K2 * d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    # only used to run previous version
    def forward_corev0(self, x: torch.Tensor, to_dtype=False, channel_first=False):
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float()  # (b, k, d_state, l)
        Cs = Cs.float()  # (b, k, d_state, l)

        As = -torch.exp(self.A_logs.float())  # (k * d, d_state)
        Ds = self.Ds.float()  # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1

        out_y = selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        # assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        return (y.to(x.dtype) if to_dtype else y)

    # only has speed difference with v0
    def forward_corev0_seq(self, x: torch.Tensor, to_dtype=False, channel_first=False):
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float()  # (b, k, d, l)
        dts = dts.contiguous().float()  # (b, k, d, l)
        Bs = Bs.float()  # (b, k, d_state, l)
        Cs = Cs.float()  # (b, k, d_state, l)

        As = -torch.exp(self.A_logs.float()).view(K, -1, self.d_state)  # (k, d, d_state)
        Ds = self.Ds.float().view(K, -1)  # (k, d)
        dt_projs_bias = self.dt_projs_bias.float().view(K, -1)  # (k, d)

        # assert len(xs.shape) == 4 and len(dts.shape) == 4 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 3 and len(Ds.shape) == 2 and len(dt_projs_bias.shape) == 2

        out_y = []
        for i in range(4):
            yi = selective_scan(
                xs[:, i], dts[:, i],
                As[i], Bs[:, i], Cs[:, i], Ds[i],
                delta_bias=dt_projs_bias[i],
                delta_softplus=True,
            ).view(B, -1, L)
            out_y.append(yi)
        out_y = torch.stack(out_y, dim=1)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        return (y.to(x.dtype) if to_dtype else y)

    def forward_corev0_share_ssm(self, x: torch.Tensor, channel_first=False):
        """
        we may conduct this ablation later, but not with v0.
        """
        ...

    def forward_corev0_share_a(self, x: torch.Tensor, channel_first=False):
        """
        we may conduct this ablation later, but not with v0.
        """
        ...

    def forward_corev2(self, x: torch.Tensor, nrows=-1, channel_first=False):
        nrows = 1
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        x = around_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None),
            nrows=nrows, delta_softplus=True, force_fp32=self.training,
        )
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def forward(self, x: torch.Tensor, **kwargs):
        xz = self.in_proj(x)
        if self.d_conv > 1:
            x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act(self.conv2d(x))  # (b, d, h, w)
        else:
            if self.disable_z_act:
                x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
                x = self.act(x)
            else:
                xz = self.act(xz)
                x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
        y = self.forward_core(x, channel_first=(self.d_conv > 1))
        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class BlockExp1(nn.Module):
    """Modified based on VSSM code"""
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_simple_init=False,
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SSExp1(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                simple_init=ssm_simple_init,
                # ==========================
                forward_type=forward_type,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                           drop=mlp_drop_rate, channels_first=False)

    def _forward(self, input: torch.Tensor):
        if self.ssm_branch:
            x = input + self.drop_path(self.op(self.norm(input)))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


class MambaExp1(nn.Module):
    """Modified based on VSSM code"""
    def __init__(
            self,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            depths=[2, 2, 9, 2],
            dims=[96, 192, 384, 768],
            # =========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_simple_init=False,
            forward_type="v2",
            # =========================
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            # =========================
            drop_path_rate=0.1,
            patch_norm=True,
            norm_layer="LN",
            downsample_version: str = "v2",  # "v1", "v2", "v3"
            patchembed_version: str = "v1",  # "v1", "v2"
            use_checkpoint=False,
            **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )

        # if norm_layer.lower() in ["ln"]:
        #     norm_layer: nn.Module = _NORMLAYERS[norm_layer.lower()]

        if ssm_act_layer.lower() in ["silu", "gelu", "relu"]:
            ssm_act_layer: nn.Module = _ACTLAYERS[ssm_act_layer.lower()]

        if mlp_act_layer.lower() in ["silu", "gelu", "relu"]:
            mlp_act_layer: nn.Module = _ACTLAYERS[mlp_act_layer.lower()]

        _make_patch_embed = dict(
            v1=self._make_patch_embed,
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer)

        _make_downsample = dict(
            v1=PatchMerging2D,
            v2=self._make_downsample,
            v3=self._make_downsample_v3,
            none=(lambda *_, **_k: None),
        ).get(downsample_version, None)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer],
                self.dims[i_layer + 1],
                norm_layer=norm_layer,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim=self.dims[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_simple_init=ssm_simple_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
            ))

        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features),  # B,H,W,C
            permute=Permute(0, 3, 1, 2),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # used in building optimizer
    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {}

    # used in building optimizer
    # @torch.jit.ignore
    # def no_weight_decay_keywords(self):
    #     return {}

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            Permute(0, 2, 3, 1),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm):
        assert patch_size == 4
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            (Permute(0, 2, 3, 1) if patch_norm else nn.Identity()),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (Permute(0, 3, 1, 2) if patch_norm else nn.Identity()),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            Permute(0, 2, 3, 1),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
            dim=96,
            drop_path=[0.1, 0.1],
            use_checkpoint=False,
            norm_layer=nn.LayerNorm,
            downsample=nn.Identity(),
            # ===========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_simple_init=False,
            forward_type="v2",
            # ===========================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(BlockExp1(
                hidden_dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_simple_init=ssm_simple_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                use_checkpoint=use_checkpoint,
            ))

        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks, ),
            downsample=downsample,
        ))

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return x

    def flops(self, shape=(3, 224, 224)):
        # shape = self.__input_shape__[1:]
        supported_ops = {
            "aten::silu": None,  # as relu is in _IGNORED_OPS
            "aten::neg": None,  # as relu is in _IGNORED_OPS
            "aten::exp": None,  # as relu is in _IGNORED_OPS
            "aten::flip": None,  # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScan": selective_scan_flop_jit,
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"

    # used to load ckpt from previous training code
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):

        def check_name(src, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    return True
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        return True
            return False

        def change_name(src, dst, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    state_dict[prefix + dst] = state_dict[prefix + src]
                    state_dict.pop(prefix + src)
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        new_k = prefix + dst + k[len(key):]
                        state_dict[new_k] = state_dict[k]
                        state_dict.pop(k)

        change_name("patch_embed.proj", "patch_embed.0")
        change_name("patch_embed.norm", "patch_embed.2")
        for i in range(100):
            for j in range(100):
                change_name(f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm")
                change_name(f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op")
        change_name("norm", "classifier.norm")
        change_name("head", "classifier.head")

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                             error_msgs)


class BackboneExp1(MambaExp1):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer=nn.LayerNorm, *args, **kwargs):
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)

        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return

        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x)  # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                out = out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        if len(self.out_indices) == 0:
            return x

        return outs





#============================================================================== EXP2 ==============================================================================
#TODO: HRNet-Like Mamba(done) Pre-training cost almost 3 months, while without pre-trained init, HR-LIKE Mamba perform worse. Compared VMamba+upernet without pretrained, HR-Like VMamba perform 26.90 (lower than 29.96 by -3.06 mIoU)
#============================================================================== EXP2 ==============================================================================
def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Upsample(nn.Module):

    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=None):
        super().__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        # (b, h, w, c)
        x = x.permute(0, 3, 1, 2).contiguous()
        if not self.size:
            size = [int(t * self.scale_factor) for t in x.shape[-2:]]
        else:
            size = self.size
        return resize(x, size, None, self.mode, self.align_corners).permute(0, 2, 3, 1).contiguous()


class ScanExp2(torch.autograd.Function):
    """pre-deal before SS"""
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        """high to low resolution"""
        B, C, H, W = x.shape  # no pad
        ctx.shape = (B, C, H, W)

        def _scan_fwd_v1(x: torch.Tensor):
            if (W % 2 != 0) or (H % 2 != 0):
                x = F.pad(x, (0, W % 2, 0, H % 2))
            _, _, _H, _W = x.shape  # pad
            xs = x.new_empty((B, 4, C, int(_H * _W / 4)))
            x0 = x[..., 0::2, 0::2]  # ... _H/2 _W/2
            x1 = x[..., 1::2, 0::2]  # ... _H/2 _W/2
            x2 = x[..., 0::2, 1::2]  # ... _H/2 _W/2
            x3 = x[..., 1::2, 1::2]  # ... _H/2 _W/2
            # x = torch.cat([x0, x1, x2, x3], 1)  # B 4*C _H/2 _W/2
            xs[:, 0] = x0.contiguous().view(B, C, -1)
            xs[:, 1] = x1.contiguous().view(B, C, -1)
            xs[:, 2] = x2.contiguous().view(B, C, -1)
            xs[:, 3] = x3.contiguous().view(B, C, -1)
            return xs, (_H, _W)

        xs, ctx.pad_shape = _scan_fwd_v1(x)

        return xs  # (B, 4, C, L) L=(_H*_W)/4

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        """low to high resolution"""
        B, C, H, W = ctx.shape
        _H, _W = ctx.pad_shape  # after padding
        h, w = int(_H/2), int(_W/2)

        def _scan_bwd_v1(ys: torch.Tensor):
            y = ys.new_empty((B, C, _H, _W))
            y[..., 0::2, 0::2] = ys[:, 0].contiguous().view(B, C, h, w)
            y[..., 1::2, 0::2] = ys[:, 1].contiguous().view(B, C, h, w)
            y[..., 0::2, 1::2] = ys[:, 2].contiguous().view(B, C, h, w)
            y[..., 1::2, 1::2] = ys[:, 3].contiguous().view(B, C, h, w)
            return y

        y = _scan_bwd_v1(ys)[..., :H, :W]  # remove pad

        return y  # (B, C, H, W)


class MergeExp2(torch.autograd.Function):
    """post-deal after SS with no downsample"""
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        """low to high resolution"""
        B, K, D, H, W = ys.shape  # H, W refer to with pad+down2X
        ctx.shape = (H, W)

        def _merge_fwd_v1(ys: torch.Tensor):
            y = ys.new_empty((B, D, int(2*H), int(2*W)))
            y[..., 0::2, 0::2] = ys[:, 0]
            y[..., 1::2, 0::2] = ys[:, 1]
            y[..., 0::2, 1::2] = ys[:, 2]
            y[..., 1::2, 1::2] = ys[:, 3]
            return y

        y = _merge_fwd_v1(ys).contiguous().view(B, D, -1)  # TODO:deal with pad part(done)

        return y  # (B, C, L) L=H*W*4

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        """high to low resolution"""
        H, W = ctx.shape  # H, W refer to with pad+down2X
        B, C, L = x.shape  # L=H*W*4

        def _merge_bwd_v1(x: torch.Tensor):
            x = x.contiguous().view(B, C, int(2*H), int(2*W))
            xs = x.new_empty((B, 4, C, H, W))
            x0 = x[..., 0::2, 0::2]  # ... H W
            x1 = x[..., 1::2, 0::2]  # ... H W
            x2 = x[..., 0::2, 1::2]  # ... H W
            x3 = x[..., 1::2, 1::2]  # ... H W
            # x = torch.cat([x0, x1, x2, x3], 1)  # B 4*C H/2 W/2
            xs[:, 0] = x0.contiguous().view(B, C, H, W)
            xs[:, 1] = x1.contiguous().view(B, C, H, W)
            xs[:, 2] = x2.contiguous().view(B, C, H, W)
            xs[:, 3] = x3.contiguous().view(B, C, H, W)
            return xs

        xs = _merge_bwd_v1(x)

        return xs  # (B, 4, C, H, W)


class DownSampleExp2(nn.Module):
    """post-deal after SS with downsample"""
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm, forward_type="v1"):
        super().__init__()
        self.dim = dim  # refer to d_innner
        # self.reduction_v1 = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        # self.norm_v1 = norm_layer(4 * dim)
        # self.reduction_v2 = nn.Linear(dim, dim if out_dim < 0 else out_dim, bias=False)
        # self.norm_v2 = norm_layer(dim)
        self.forward_type = forward_type
        if forward_type == 'v1':
            self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
            self.norm = norm_layer(4 * dim)
        elif forward_type == 'v2':
            self.reduction = nn.Linear(dim, dim if out_dim < 0 else out_dim, bias=False)
            self.norm = norm_layer(dim)

    def forward_v1(self, x):
        """concat"""
        B, K, C, H, W = x.shape  # H, W refer to with pad+down2X
        assert C == self.dim, "Input size must be d_inner!"
        assert K == 4, "Error pre-deal before SS! Please check function 'ScanExp2'..."
        x = torch.cat([x[:, 0], x[:, 1], x[:, 2], x[:, 3]], 1).permute(0, 2, 3, 1).contiguous()  # (b, h, w, 4c)
        x = self.norm(x)
        x = self.reduction(x).permute(0, 3, 1, 2).contiguous().view(B, -1, H*W)

        return x

    def forward_v2(self, x):
        """sum"""
        B, K, C, H, W = x.shape  # H, W refer to with pad+down2X
        assert C == self.dim, "Input size must be d_inner!"
        assert K == 4, "Error pre-deal before SS! Please check function 'ScanExp2'..."
        x = torch.sum(x, dim=1).permute(0, 2, 3, 1).contiguous()  # (b, h, w, c)
        x = self.norm(x)
        x = self.reduction(x).permute(0, 3, 1, 2).contiguous().view(B, -1, H*W)

        return x

    def forward(self, x):
        # TODO: use v2(done)
        if self.forward_type == "v1":
            return self.forward_v1(x)
        elif self.forward_type == "v2":
            return self.forward_v2(x)


# TODO:post-deal after SS with upsample using in stage-fuse layer


def selective_scan_exp2(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        out_norm: torch.nn.Module = None,
        nrows=-1,
        delta_softplus=True,
        to_dtype=True,
        force_fp32=True,
        downsample=None  # 'DownSampleExp2' object
):
    """SS block with w/wo downsample"""
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

    B, C, H, W = x.shape  # D=2*96,2*192...
    _H, _W = H, W  # with pad
    if (W % 2 != 0):
        _W = W + 1
    if (H % 2 != 0):
        _H = H + 1
    D, N = A_logs.shape  # D=4*2*96,4*2*192...
    K, D, R = dt_projs_weight.shape  # D=2*96,4*96...
    L = _H * _W

    if nrows < 1:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1

    xs = ScanExp2.apply(x)

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.contiguous().view(B, -1, int(L/4))
    dts = dts.contiguous().view(B, -1, int(L/4))
    As = -torch.exp(A_logs.to(torch.float))  # (K * C, d_state)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)  # (K * C)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
    ).view(B, K, -1, int(_H/2), int(_W/2))

    if downsample is None:  # wo downsample
        y: torch.Tensor = MergeExp2.apply(ys)
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = out_norm(y).view(B, _H, _W, -1)[:, :H, :W]  # remove pad
    else:  # w downsample
        y: torch.Tensor = downsample(ys)  # TODO:deal-with pad when upsample(done)
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L/4, C)
        y = out_norm(y).view(B, int(_H/2), int(_W/2), -1)  # (B, _, _, C)

    return (y.to(x.dtype) if to_dtype else y)


class SSExp2(nn.Module):
    """realize two kinds of SS: 1. with no downsample; 2. with downsample"""
    def __init__(
            self,
            # for two kinds of block ===========
            block_type="v1",  # type of blocks
            in_dim=96,  # input channels of blocks
            out_dim=96,  # output channels of blocks
            with_ffn=False,  # weather or not adopt linear layer to map in_dim to inner_dim
            # basic dims ===========
            # d_model=96,  # change to out_dim
            d_state=16,  # TODO: check this for performance
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            dt_rank="auto",  # TODO: check this for performance
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            simple_init=False,
            # ======================
            forward_type="v2",
            # ======================
            **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * out_dim)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * out_dim) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(out_dim / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(out_dim / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv

        # disable z act ======================================
        self.disable_z_act = forward_type[-len("nozact"):] == "nozact"
        if self.disable_z_act:
            forward_type = forward_type[:-len("nozact")]

        # softmax | sigmoid | norm ===========================
        if forward_type[-len("softmax"):] == "softmax":
            forward_type = forward_type[:-len("softmax")]
            self.out_norm = nn.Softmax(dim=1)
        elif forward_type[-len("sigmoid"):] == "sigmoid":
            forward_type = forward_type[:-len("sigmoid")]
            self.out_norm = nn.Sigmoid()
        else:
            self.out_norm = nn.LayerNorm(d_inner)

        # forward_type =======================================
        self.forward_core = dict(
            v0=self.forward_corev0,
            v0_seq=self.forward_corev0_seq,
            v1=self.forward_corev2,
            v2=self.forward_corev2,
            share_ssm=self.forward_corev0_share_ssm,
            share_a=self.forward_corev0_share_a,
        ).get(forward_type, self.forward_corev2)
        self.K = 4 if forward_type not in ["share_ssm"] else 1
        self.K2 = self.K if forward_type not in ["share_a"] else 1

        # in proj =======================================
        self.prefix_ffn = nn.Identity()
        self.in_proj = nn.Linear(in_dim, d_expand * 2, bias=bias, **factory_kwargs)
        if with_ffn:
            self.prefix_ffn = Mlp(in_features=in_dim, out_features=out_dim)
            self.in_proj = nn.Linear(out_dim, d_expand * 2, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True)  # (K * D, N) | (K * inner, N)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True)  # (K * D)

        # out proj =======================================
        self.out_proj = nn.Linear(d_expand, out_dim, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if simple_init:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
            self.A_logs = nn.Parameter(
                torch.randn((self.K2 * d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

        # block_type =======================================
        self.block_type = block_type
        self.downsample_module = None
        if self.block_type == "v2":
            self.downsample_module = DownSampleExp2(dim=d_inner, out_dim=d_inner)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    # only used to run previous version
    def forward_corev0(self, x: torch.Tensor, to_dtype=False, channel_first=False):
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float()  # (b, k, d_state, l)
        Cs = Cs.float()  # (b, k, d_state, l)

        As = -torch.exp(self.A_logs.float())  # (k * d, d_state)
        Ds = self.Ds.float()  # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1

        out_y = selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        # assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        return (y.to(x.dtype) if to_dtype else y)

    # only has speed difference with v0
    def forward_corev0_seq(self, x: torch.Tensor, to_dtype=False, channel_first=False):
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float()  # (b, k, d, l)
        dts = dts.contiguous().float()  # (b, k, d, l)
        Bs = Bs.float()  # (b, k, d_state, l)
        Cs = Cs.float()  # (b, k, d_state, l)

        As = -torch.exp(self.A_logs.float()).view(K, -1, self.d_state)  # (k, d, d_state)
        Ds = self.Ds.float().view(K, -1)  # (k, d)
        dt_projs_bias = self.dt_projs_bias.float().view(K, -1)  # (k, d)

        # assert len(xs.shape) == 4 and len(dts.shape) == 4 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 3 and len(Ds.shape) == 2 and len(dt_projs_bias.shape) == 2

        out_y = []
        for i in range(4):
            yi = selective_scan(
                xs[:, i], dts[:, i],
                As[i], Bs[:, i], Cs[:, i], Ds[i],
                delta_bias=dt_projs_bias[i],
                delta_softplus=True,
            ).view(B, -1, L)
            out_y.append(yi)
        out_y = torch.stack(out_y, dim=1)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        return (y.to(x.dtype) if to_dtype else y)

    def forward_corev0_share_ssm(self, x: torch.Tensor, channel_first=False):
        """
        we may conduct this ablation later, but not with v0.
        """
        ...

    def forward_corev0_share_a(self, x: torch.Tensor, channel_first=False):
        """
        we may conduct this ablation later, but not with v0.
        """
        ...

    def forward_corev2(self, x: torch.Tensor, nrows=-1, channel_first=False):
        # (b, c, h, w)
        nrows = 1
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        x = selective_scan_exp2(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None),
            nrows=nrows, delta_softplus=True, force_fp32=self.training, downsample=self.downsample_module
        )  # TODO: w/wo downsample(done)
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def forward(self, x: torch.Tensor, **kwargs):  # TODO: self.d_conv < 1 and self.disable_z_act is True
        # (b, h, w, c)
        x = self.prefix_ffn(x)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y = self.forward_core(x, channel_first=(self.d_conv > 1))  # (b, h, w, c)
        if self.downsample_module:
            z = F.interpolate(z.permute(0, 3, 1, 2).contiguous(), size=y.shape[1:3]).permute(0, 2, 3, 1).contiguous()  # (b, h, w, c)
        z = self.act(z)
        y = y * z  # TODO: downsample z: 1. use bilinear interpolation; 2. use DownSampleExp2
        out = self.dropout(self.out_proj(y))
        return out  # (b, h, w, c)


class BlockExp2(nn.Module):
    """two kinds of blocks"""
    def __init__(
            self,
            block_type="v1",  # type of blocks
            with_ffn=False,  # weather or not adopt linear layer to map in_dim to inner_dim
            in_dim: int=0,
            out_dim: int=0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_simple_init=False,
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint

        # input and output have not same channels and size
        self.block_type = block_type
        self.pre_mlp = nn.Identity()  # pre-deal before block to map in_dim to out_dim, TODO: middle-deal in block: respectively map input to out_dim in SS and outside SS, use with_ffn to map out_dim in SS.
        if in_dim != out_dim:
            mlp_hidden_dim = int(out_dim * mlp_ratio)
            self.pre_mlp = nn.Sequential(
                norm_layer(in_dim),
                Mlp(in_features=in_dim, hidden_features=mlp_hidden_dim, out_features=out_dim, act_layer=mlp_act_layer,
                    drop=mlp_drop_rate, channels_first=False)
            )

        if self.ssm_branch:
            self.norm = norm_layer(out_dim)
            self.op = SSExp2(
                block_type=block_type,
                with_ffn=with_ffn,
                in_dim=out_dim,
                out_dim=out_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                simple_init=ssm_simple_init,
                # ==========================
                forward_type=forward_type,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            self.norm2 = norm_layer(out_dim)
            mlp_hidden_dim = int(out_dim * mlp_ratio)
            self.mlp = Mlp(in_features=out_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                           drop=mlp_drop_rate, channels_first=False)

    def _forward(self, input: torch.Tensor):
        # (b, h/n, w/n, cn)
        input = self.pre_mlp(input)  # (b, h/n, w/n, cn)  # TODO: need or not norm before pre_mlp(done)
        if self.ssm_branch:
            x = self.drop_path(self.op(self.norm(input)))  # (b, h/n, w/n, cn)
            if self.block_type == 'v2':  # downsample block
                input = F.interpolate(input.permute(0, 3, 1, 2).contiguous(), size=x.shape[1:3]).permute(0, 2, 3, 1).contiguous()
            x += input
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x  # (b, h/n, w/n, cn)

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


class HRModule(BaseModule):
    """High-Resolution Module for HRNet.

    In this module, every branch has 4 BasicBlocks/Bottlenecks. Fusion/Exchange
    is in this module.
    """

    def __init__(self,
                 num_branches,
                 num_blocks,
                 in_channels,  # same as num_channels
                 num_channels,
                 multiscale_output=True,
                 init_cfg=None,
                 # SS settings ======================
                 drop_path = [0.0, 0.0],
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 # =============================
                 ssm_d_state: int = 16,
                 ssm_ratio=2.0,
                 ssm_rank_ratio=2.0,
                 ssm_dt_rank: Any = "auto",
                 ssm_act_layer=nn.SiLU,
                 ssm_conv: int = 3,
                 ssm_conv_bias=True,
                 ssm_drop_rate: float = 0,
                 ssm_simple_init=False,
                 forward_type="v2",
                 # =============================
                 mlp_ratio=4.0,
                 mlp_act_layer=nn.GELU,
                 mlp_drop_rate: float = 0.0,
                 # =============================
                 use_checkpoint: bool = False,
                 ):
        super().__init__(init_cfg)

        self._check_branches(num_branches, num_blocks, in_channels, num_channels)
        self.in_channels = in_channels
        self.num_branches = num_branches
        self.multiscale_output = multiscale_output
        self.branches = self._make_branches(num_branches, num_blocks, num_channels,
                                            drop_path, norm_layer, ssm_d_state, ssm_ratio,
                                            ssm_rank_ratio, ssm_dt_rank, ssm_act_layer,
                                            ssm_conv, ssm_conv_bias, ssm_drop_rate,
                                            ssm_simple_init, forward_type, mlp_ratio,
                                            mlp_act_layer, mlp_drop_rate, use_checkpoint)
        self.fuse_layers = self._make_fuse_layers(norm_layer, ssm_d_state, ssm_ratio,
                                            ssm_rank_ratio, ssm_dt_rank, ssm_act_layer,
                                            ssm_conv, ssm_conv_bias, ssm_drop_rate,
                                            ssm_simple_init, forward_type, mlp_ratio,
                                            mlp_act_layer, mlp_drop_rate, use_checkpoint)
        # self.relu = nn.ReLU(inplace=False)
        # self.gelu = nn.GELU()

    def _check_branches(self, num_branches, num_blocks, in_channels,
                        num_channels):
        """Check branches configuration."""
        if num_branches != len(num_blocks):
            error_msg = f'NUM_BRANCHES({num_branches}) <> NUM_BLOCKS(' \
                        f'{len(num_blocks)})'
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) <> NUM_CHANNELS(' \
                        f'{len(num_channels)})'
            raise ValueError(error_msg)

        if num_branches != len(in_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) <> NUM_INCHANNELS(' \
                        f'{len(in_channels)})'
            raise ValueError(error_msg)

    def _make_one_branch(self,
                         branch_index,
                         num_blocks,
                         num_channels,
                         drop_path, norm_layer, ssm_d_state, ssm_ratio, ssm_rank_ratio,
                         ssm_dt_rank, ssm_act_layer, ssm_conv, ssm_conv_bias, ssm_drop_rate,
                         ssm_simple_init, forward_type, mlp_ratio, mlp_act_layer, mlp_drop_rate, use_checkpoint):
        """Build one branch."""
        layers = []
        layers.append(
            BlockExp2(
                block_type="v1",
                with_ffn=False,  # in stage, no use ffn as pre-deal before SS
                in_dim=self.in_channels[branch_index],
                out_dim=num_channels[branch_index],
                drop_path=drop_path[0],
                norm_layer=norm_layer,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_simple_init=ssm_simple_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                use_checkpoint=use_checkpoint
            )
        )
        # self.in_channels[branch_index] = num_channels[branch_index]  # TODO: if or not expand channels when input in 1-st block
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                BlockExp2(
                    block_type="v1",
                    with_ffn=False,  # in stage, no use ffn as pre-deal before SS
                    in_dim=self.in_channels[branch_index],
                    out_dim=num_channels[branch_index],
                    drop_path=drop_path[i:i+1][0],  # slice list still be list
                    norm_layer=norm_layer,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_rank_ratio=ssm_rank_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_simple_init=ssm_simple_init,
                    forward_type=forward_type,
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=mlp_act_layer,
                    mlp_drop_rate=mlp_drop_rate,
                    use_checkpoint=use_checkpoint
                )
            )

        return Sequential(*layers)

    def _make_branches(self, num_branches, num_blocks, num_channels,
                       drop_path, norm_layer, ssm_d_state, ssm_ratio, ssm_rank_ratio, ssm_dt_rank,
                       ssm_act_layer, ssm_conv, ssm_conv_bias, ssm_drop_rate, ssm_simple_init, forward_type,
                       mlp_ratio, mlp_act_layer, mlp_drop_rate, use_checkpoint):
        """Build multiple branch."""
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, num_blocks, num_channels,
                                      drop_path, norm_layer, ssm_d_state, ssm_ratio, ssm_rank_ratio, ssm_dt_rank,
                                      ssm_act_layer, ssm_conv, ssm_conv_bias, ssm_drop_rate, ssm_simple_init, forward_type,
                                      mlp_ratio, mlp_act_layer, mlp_drop_rate, use_checkpoint))

        return ModuleList(branches)

    def _make_fuse_layers(self, norm_layer, ssm_d_state, ssm_ratio,
                        ssm_rank_ratio, ssm_dt_rank, ssm_act_layer,
                        ssm_conv, ssm_conv_bias, ssm_drop_rate,
                        ssm_simple_init, forward_type, mlp_ratio,
                        mlp_act_layer, mlp_drop_rate, use_checkpoint, drop_path: float=0.0):
        """Build fuse layer. no use DropPath"""
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:  # upsample module, TODO:analyze two kinds of upsample model: up->BlockExp2 | BlockExp2->up(done)
                    fuse_layer.append(
                        nn.Sequential(
                            # # we set align_corners=False for HRNet
                            # Upsample(
                            #     scale_factor=2 ** (j - i),
                            #     mode='bilinear',
                            #     align_corners=False),
                            BlockExp2(
                                block_type="v1",
                                with_ffn=False,
                                in_dim=in_channels[j],
                                out_dim=in_channels[i],
                                drop_path=drop_path,
                                norm_layer=norm_layer,
                                ssm_d_state=ssm_d_state,
                                ssm_ratio=ssm_ratio,
                                ssm_rank_ratio=ssm_rank_ratio,
                                ssm_dt_rank=ssm_dt_rank,
                                ssm_act_layer=ssm_act_layer,
                                ssm_conv=ssm_conv,
                                ssm_conv_bias=ssm_conv_bias,
                                ssm_drop_rate=ssm_drop_rate,
                                ssm_simple_init=ssm_simple_init,
                                forward_type=forward_type,
                                mlp_ratio=mlp_ratio,
                                mlp_act_layer=mlp_act_layer,
                                mlp_drop_rate=mlp_drop_rate,
                                use_checkpoint=use_checkpoint
                            ),
                            # # # we set align_corners=False for HRNet
                            Upsample(
                                scale_factor=2**(j - i),
                                mode='bilinear',
                                align_corners=False)
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:  # downsample module
                    downsample_modules = []
                    for k in range(i - j):
                        if k == i - j - 1:  # the last downsample module,TODO: if or not extra deal
                            downsample_modules.append(
                                nn.Sequential(
                                    BlockExp2(
                                        block_type="v2",
                                        with_ffn=False,
                                        in_dim=in_channels[j],
                                        out_dim=in_channels[i],
                                        drop_path=drop_path,
                                        norm_layer=norm_layer,
                                        ssm_d_state=ssm_d_state,
                                        ssm_ratio=ssm_ratio,
                                        ssm_rank_ratio=ssm_rank_ratio,
                                        ssm_dt_rank=ssm_dt_rank,
                                        ssm_act_layer=ssm_act_layer,
                                        ssm_conv=ssm_conv,
                                        ssm_conv_bias=ssm_conv_bias,
                                        ssm_drop_rate=ssm_drop_rate,
                                        ssm_simple_init=ssm_simple_init,
                                        forward_type=forward_type,
                                        mlp_ratio=mlp_ratio,
                                        mlp_act_layer=mlp_act_layer,
                                        mlp_drop_rate=mlp_drop_rate,
                                        use_checkpoint=use_checkpoint
                                    )
                                )
                            )
                        else:
                            downsample_modules.append(
                                nn.Sequential(
                                    BlockExp2(
                                        block_type="v2",
                                        with_ffn=False,
                                        in_dim=in_channels[j],
                                        out_dim=in_channels[j],
                                        drop_path=drop_path,
                                        norm_layer=norm_layer,
                                        ssm_d_state=ssm_d_state,
                                        ssm_ratio=ssm_ratio,
                                        ssm_rank_ratio=ssm_rank_ratio,
                                        ssm_dt_rank=ssm_dt_rank,
                                        ssm_act_layer=ssm_act_layer,
                                        ssm_conv=ssm_conv,
                                        ssm_conv_bias=ssm_conv_bias,
                                        ssm_drop_rate=ssm_drop_rate,
                                        ssm_simple_init=ssm_simple_init,
                                        forward_type=forward_type,
                                        mlp_ratio=mlp_ratio,
                                        mlp_act_layer=mlp_act_layer,
                                        mlp_drop_rate=mlp_drop_rate,
                                        use_checkpoint=use_checkpoint
                                    )
                                )
                            )
                    fuse_layer.append(nn.Sequential(*downsample_modules))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        """Forward function."""
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = 0
            for j in range(self.num_branches):
                if i == j:
                    y += x[j]  # (b, h, w, c)
                elif j > i:
                    y = y + resize(
                        self.fuse_layers[i][j](x[j]).permute(0, 3, 1, 2).contiguous(),
                        size=x[i].shape[1:3],
                        mode='bilinear',
                        align_corners=False).permute(0, 2, 3, 1).contiguous()
                else:
                    y += self.fuse_layers[i][j](x[j])
            # x_fuse.append(self.relu(y))  # TODO: check diff act layer(done)
            # x_fuse.append(self.gelu(y))
            x_fuse.append(y)
        return x_fuse


@MODELS_MMSEG.register_module()
class Exp2(nn.Module):
    """HR-Like Mamba"""
    def __init__(
            self,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            extra={},  # HR-Like settings
            multiscale_output=True,
            # SS settings =========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_simple_init=False,
            forward_type="v2",
            drop_path_rate=0.1,
            # patch embedding settings =========================
            patch_norm=True,
            norm_layer="LN",
            downsample_version: str = "v2",  # "v1", "v2", "v3"
            patchembed_version: str = "v1",  # "v1", "v2"
            # pre-train settings =========================
            use_checkpoint=False,
            # mlp settings =========================
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            pretrained=None,
            out_indices=(0, 1, 2, 3),
            **kwargs,
    ):
        super().__init__()
        # Assert configurations of 4 stages are in extra
        assert 'stage1' in extra and 'stage2' in extra \
               and 'stage3' in extra and 'stage4' in extra
        for i in range(4):
            cfg = extra[f'stage{i + 1}']
            assert len(cfg['num_blocks']) == cfg['num_branches'] and \
                   len(cfg['num_channels']) == cfg['num_branches']

        self.extra = extra
        self.num_classes = num_classes
        depths = self.extra['depths']  # transition and downsample layer not set dpr
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            bn=nn.BatchNorm2d,
        )
        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )
        if norm_layer.lower() in ["ln"]:
            norm_layer: nn.Module = _NORMLAYERS[norm_layer.lower()]
        if ssm_act_layer.lower() in ["silu", "gelu", "relu"]:
            ssm_act_layer: nn.Module = _ACTLAYERS[ssm_act_layer.lower()]
        if mlp_act_layer.lower() in ["silu", "gelu", "relu"]:
            mlp_act_layer: nn.Module = _ACTLAYERS[mlp_act_layer.lower()]

        # stage 1
        self.stage1_cfg = self.extra['stage1']
        num_channels = self.stage1_cfg['num_channels'][0]
        num_blocks = self.stage1_cfg['num_blocks'][0]
        stage1_out_channels = num_channels
        _make_patch_embed = dict(
            v1=self._make_patch_embed,
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)
        self.layer1 = _make_patch_embed(in_chans, stage1_out_channels, patch_size, patch_norm, norm_layer)

        # stage 2
        self.stage2_cfg = self.extra['stage2']
        num_channels = self.stage2_cfg['num_channels']
        self.transition1 = self._make_transition_layer([stage1_out_channels], num_channels, norm_layer, ssm_d_state,
                                                       ssm_ratio, ssm_rank_ratio, ssm_dt_rank, ssm_act_layer, ssm_conv, ssm_conv_bias,
                                                       ssm_drop_rate, ssm_simple_init, forward_type, mlp_ratio, mlp_act_layer,
                                                       mlp_drop_rate, use_checkpoint)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels, dpr[sum(depths[:0]):sum(depths[:1])], norm_layer, ssm_d_state, ssm_ratio, ssm_rank_ratio,
            ssm_dt_rank, ssm_act_layer, ssm_conv, ssm_conv_bias, ssm_drop_rate, ssm_simple_init, forward_type,
            mlp_ratio, mlp_act_layer, mlp_drop_rate, use_checkpoint)

        # stage 3
        self.stage3_cfg = self.extra['stage3']
        num_channels = self.stage3_cfg['num_channels']
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels, norm_layer, ssm_d_state,
                                                       ssm_ratio, ssm_rank_ratio, ssm_dt_rank, ssm_act_layer, ssm_conv, ssm_conv_bias,
                                                       ssm_drop_rate, ssm_simple_init, forward_type, mlp_ratio, mlp_act_layer,
                                                       mlp_drop_rate, use_checkpoint)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels, dpr[sum(depths[:1]):sum(depths[:2])], norm_layer, ssm_d_state, ssm_ratio, ssm_rank_ratio,
            ssm_dt_rank, ssm_act_layer, ssm_conv, ssm_conv_bias, ssm_drop_rate, ssm_simple_init, forward_type,
            mlp_ratio, mlp_act_layer, mlp_drop_rate, use_checkpoint)

        # stage 4
        self.stage4_cfg = self.extra['stage4']
        num_channels = self.stage4_cfg['num_channels']
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels, norm_layer, ssm_d_state,
                                                       ssm_ratio, ssm_rank_ratio, ssm_dt_rank, ssm_act_layer, ssm_conv, ssm_conv_bias,
                                                       ssm_drop_rate, ssm_simple_init, forward_type, mlp_ratio, mlp_act_layer,
                                                       mlp_drop_rate, use_checkpoint)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, dpr[sum(depths[:2]):sum(depths[:3])], norm_layer, ssm_d_state, ssm_ratio, ssm_rank_ratio,
            ssm_dt_rank, ssm_act_layer, ssm_conv, ssm_conv_bias, ssm_drop_rate, ssm_simple_init, forward_type,
            mlp_ratio, mlp_act_layer, mlp_drop_rate, use_checkpoint, multiscale_output=multiscale_output)

        last_inp_channels = int(sum(pre_stage_channels))

        # TODO: chose fit last_layer
        # self.last_layer = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=last_inp_channels,
        #         out_channels=last_inp_channels,
        #         kernel_size=1,
        #         stride=1,
        #         padding=0),
        #     norm_layer(last_inp_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(
        #         in_channels=last_inp_channels,
        #         out_channels=self.num_classes,
        #         kernel_size=1,
        #         stride=1,
        #         padding=0)
        # )

        # self.last_layer = nn.Sequential(OrderedDict(
        #     norm=norm_layer(last_inp_channels),  # B,H,W,C
        #     permute=Permute(0, 3, 1, 2),
        #     avgpool=nn.AdaptiveAvgPool2d(1),
        #     flatten=nn.Flatten(1),
        #     head=nn.Linear(last_inp_channels, num_classes),
        # ))
        for i in range(4):
            layer = norm_layer(num_channels[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        self.apply(self._init_weights)
        self.load_pretrained(pretrained)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            Permute(0, 2, 3, 1),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm):
        assert patch_size == 4
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            (Permute(0, 2, 3, 1) if patch_norm else nn.Identity()),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (Permute(0, 3, 1, 2) if patch_norm else nn.Identity()),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer, norm_layer, ssm_d_state, ssm_ratio,
                               ssm_rank_ratio, ssm_dt_rank, ssm_act_layer, ssm_conv, ssm_conv_bias, ssm_drop_rate, ssm_simple_init,
                               forward_type, mlp_ratio, mlp_act_layer, mlp_drop_rate, use_checkpoint, drop_path: float=0.0):
        """Make transition layer. In transition, no use DropPath"""
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:  # the cur branch wo downsample
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:  # only change channels
                    transition_layers.append(
                        BlockExp2(
                            block_type="v1",
                            with_ffn=False,
                            in_dim=num_channels_pre_layer[i],
                            out_dim=num_channels_cur_layer[i],
                            drop_path=drop_path,
                            norm_layer=norm_layer,
                            ssm_d_state=ssm_d_state,
                            ssm_ratio=ssm_ratio,
                            ssm_rank_ratio=ssm_rank_ratio,
                            ssm_dt_rank=ssm_dt_rank,
                            ssm_act_layer=ssm_act_layer,
                            ssm_conv=ssm_conv,
                            ssm_conv_bias=ssm_conv_bias,
                            ssm_drop_rate=ssm_drop_rate,
                            ssm_simple_init=ssm_simple_init,
                            forward_type=forward_type,
                            mlp_ratio=mlp_ratio,
                            mlp_act_layer=mlp_act_layer,
                            mlp_drop_rate=mlp_drop_rate,
                            use_checkpoint=use_checkpoint
                        )
                    )
                else:
                    transition_layers.append(None)
            else:  # the cur branch w downsample, only once loop
                downsample_modules = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else in_channels
                    downsample_modules.append(
                        BlockExp2(
                            block_type="v2",
                            with_ffn=False,
                            in_dim=in_channels,
                            out_dim=out_channels,
                            drop_path=drop_path,
                            norm_layer=norm_layer,
                            ssm_d_state=ssm_d_state,
                            ssm_ratio=ssm_ratio,
                            ssm_rank_ratio=ssm_rank_ratio,
                            ssm_dt_rank=ssm_dt_rank,
                            ssm_act_layer=ssm_act_layer,
                            ssm_conv=ssm_conv,
                            ssm_conv_bias=ssm_conv_bias,
                            ssm_drop_rate=ssm_drop_rate,
                            ssm_simple_init=ssm_simple_init,
                            forward_type=forward_type,
                            mlp_ratio=mlp_ratio,
                            mlp_act_layer=mlp_act_layer,
                            mlp_drop_rate=mlp_drop_rate,
                            use_checkpoint=use_checkpoint
                        )
                    )
                transition_layers.append(nn.Sequential(*downsample_modules))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, layer_config, in_channels, drop_path, norm_layer, ssm_d_state, ssm_ratio, ssm_rank_ratio, ssm_dt_rank,
                    ssm_act_layer, ssm_conv, ssm_conv_bias, ssm_drop_rate, ssm_simple_init, forward_type, mlp_ratio, mlp_act_layer,
                    mlp_drop_rate, use_checkpoint, multiscale_output=True):
        """Make each stage. Use DropPath"""
        num_modules = layer_config['num_modules']  # default set to 1
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']

        hr_modules = []
        for i in range(num_modules):
            # multi_scale_output is only used for the last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            _drop_path = drop_path[i*num_blocks[0]: (i+1)*num_blocks[0]]
            hr_modules.append(
                HRModule(
                    num_branches=num_branches,
                    num_blocks=num_blocks,
                    in_channels=in_channels,
                    num_channels=num_channels,
                    multiscale_output=reset_multiscale_output,
                    drop_path=_drop_path,  # list
                    norm_layer=norm_layer,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_rank_ratio=ssm_rank_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_simple_init=ssm_simple_init,
                    forward_type=forward_type,
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=mlp_act_layer,
                    mlp_drop_rate=mlp_drop_rate,
                    use_checkpoint=use_checkpoint
                    )
            )

        return Sequential(*hr_modules), in_channels

    def forward(self, x):
        """Forward function."""
        # TODO: check size(done)
        # (b, 3, h, w)
        x = self.layer1(x)
        # (b, h/4, w/4, c1)

        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        outs = []
        for i in range(len(y_list)):  # (b, h, w, c) -> (b, c, h, w)  # TODO: use outnorm(done)
            norm_layer = getattr(self, f'outnorm{i}')
            out = norm_layer(y_list[i])
            out = out.permute(0, 3, 1, 2).contiguous()
            outs.append(out)

        return outs

        # if or not use decoder
        # decoder = True  # TODO: not use decoder
        # if decoder:
        #     return outs
        # else:
        #     ALIGN_CORNERS = False
        #     x0_h, x0_w = y_list[0].size(2), y_list[0].size(3)
        #     x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        #     x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        #     x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        #     x = torch.cat([x[0], x1, x2, x3], 1)
        #     x = self.last_layer(x)

        # return x

    def flops(self, shape=(3, 224, 224)):
        # shape = self.__input_shape__[1:]
        supported_ops = {
            "aten::silu": None,  # as relu is in _IGNORED_OPS
            "aten::neg": None,  # as relu is in _IGNORED_OPS
            "aten::exp": None,  # as relu is in _IGNORED_OPS
            "aten::flip": None,  # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScan": selective_scan_flop_jit,
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        # return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"

    def load_pretrained(self, ckpt=None, key="state_dict"):
        if ckpt is None:
            return

        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            # print(_ckpt.key)
            state_dict = OrderedDict()
            _ckpt = _ckpt[key]
            metadata = getattr(_ckpt, '_metadata', OrderedDict())
            for k in _ckpt.keys():
                if k.startswith('backbone.'):
                    new_k = k.replace('backbone.', '')
                    state_dict[new_k] = _ckpt[k]
            state_dict._metadata = metadata
            # ckpt_keys = set(_ckpt['state_dict'].keys())
            # model_keys = set(self.state_dict().keys())
            # print(ckpt_keys)
            # print(model_keys)
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(state_dict, strict=False)
            print(incompatibleKeys)
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")





#============================================================================== EXP3 ==============================================================================
#TODO: Mamba-based decoder: Mask-attention UNet-Like Mamba-based decoder(done)
#============================================================================== EXP3 ==============================================================================
class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs  # (B, 4, C, L)

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)  # (B, C, H, W)


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y  # (B, C, L)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs, None, None  # (B, 4, C, H, W)


def cross_selective_scan(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        out_norm: torch.nn.Module = None,
        nrows=-1,
        delta_softplus=True,
        to_dtype=True,
        force_fp32=True,
):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    if nrows < 1:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1

    xs = CrossScan.apply(x)

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)  # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
    ).view(B, K, -1, H, W)

    y: torch.Tensor = CrossMerge.apply(ys)
    y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
    y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)


class SS2D(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            simple_init=False,
            # ======================
            forward_type="v2",
            # ======================
            **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv

        # disable z act ======================================
        self.disable_z_act = forward_type[-len("nozact"):] == "nozact"
        if self.disable_z_act:
            forward_type = forward_type[:-len("nozact")]

        # softmax | sigmoid | norm ===========================
        if forward_type[-len("softmax"):] == "softmax":
            forward_type = forward_type[:-len("softmax")]
            self.out_norm = nn.Softmax(dim=1)
        elif forward_type[-len("sigmoid"):] == "sigmoid":
            forward_type = forward_type[:-len("sigmoid")]
            self.out_norm = nn.Sigmoid()
        else:
            self.out_norm = nn.LayerNorm(d_inner)

        # forward_type =======================================
        self.forward_core = dict(
            v0=self.forward_corev0,
            v0_seq=self.forward_corev0_seq,
            v1=self.forward_corev2,
            v2=self.forward_corev2,
            share_ssm=self.forward_corev0_share_ssm,
            share_a=self.forward_corev0_share_a,
        ).get(forward_type, self.forward_corev2)
        self.K = 4 if forward_type not in ["share_ssm"] else 1
        self.K2 = self.K if forward_type not in ["share_a"] else 1

        # in proj =======================================
        self.in_proj = nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True)  # (K * D, N) | (K * inner, N)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True)  # (K * D)

        # out proj =======================================
        self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if simple_init:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
            self.A_logs = nn.Parameter(
                torch.randn((self.K2 * d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    # only used to run previous version
    def forward_corev0(self, x: torch.Tensor, to_dtype=False, channel_first=False):
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float()  # (b, k, d_state, l)
        Cs = Cs.float()  # (b, k, d_state, l)

        As = -torch.exp(self.A_logs.float())  # (k * d, d_state)
        Ds = self.Ds.float()  # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1

        out_y = selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        # assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        return (y.to(x.dtype) if to_dtype else y)

    # only has speed difference with v0
    def forward_corev0_seq(self, x: torch.Tensor, to_dtype=False, channel_first=False):
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float()  # (b, k, d, l)
        dts = dts.contiguous().float()  # (b, k, d, l)
        Bs = Bs.float()  # (b, k, d_state, l)
        Cs = Cs.float()  # (b, k, d_state, l)

        As = -torch.exp(self.A_logs.float()).view(K, -1, self.d_state)  # (k, d, d_state)
        Ds = self.Ds.float().view(K, -1)  # (k, d)
        dt_projs_bias = self.dt_projs_bias.float().view(K, -1)  # (k, d)

        # assert len(xs.shape) == 4 and len(dts.shape) == 4 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 3 and len(Ds.shape) == 2 and len(dt_projs_bias.shape) == 2

        out_y = []
        for i in range(4):
            yi = selective_scan(
                xs[:, i], dts[:, i],
                As[i], Bs[:, i], Cs[:, i], Ds[i],
                delta_bias=dt_projs_bias[i],
                delta_softplus=True,
            ).view(B, -1, L)
            out_y.append(yi)
        out_y = torch.stack(out_y, dim=1)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        return (y.to(x.dtype) if to_dtype else y)

    def forward_corev0_share_ssm(self, x: torch.Tensor, channel_first=False):
        """
        we may conduct this ablation later, but not with v0.
        """
        ...

    def forward_corev0_share_a(self, x: torch.Tensor, channel_first=False):
        """
        we may conduct this ablation later, but not with v0.
        """
        ...

    def forward_corev2(self, x: torch.Tensor, nrows=-1, channel_first=False):
        nrows = 1
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        x = cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None),
            nrows=nrows, delta_softplus=True, force_fp32=self.training,
        )
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def forward(self, x: torch.Tensor, **kwargs):
        xz = self.in_proj(x)
        if self.d_conv > 1:
            x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act(self.conv2d(x))  # (b, d, h, w)
        else:
            if self.disable_z_act:
                x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
                x = self.act(x)
            else:
                xz = self.act(xz)
                x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
        y = self.forward_core(x, channel_first=(self.d_conv > 1))
        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            in_dim: int = 0,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_simple_init=False,
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.pre_mlp = nn.Identity()  # pre-deal before block to map in_dim to out_dim
        if in_dim != hidden_dim:
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.pre_mlp = nn.Sequential(
                # ConvModule(
                #     in_dim,
                #     hidden_dim,
                #     1,
                #     norm_cfg=dict(type='SyncBN', requires_grad=True),
                #     act_cfg=dict(type='ReLU'),
                #     inplace=False
                # ),
                # norm_layer(in_dim),  # encoder output execute norm
                nn.Linear(in_dim, hidden_dim),  # TODO: use conv1*1 to change dim(done, same as linear_layer)
                # Mlp(in_features=in_dim, hidden_features=mlp_hidden_dim, out_features=hidden_dim, act_layer=mlp_act_layer,
                #     drop=mlp_drop_rate, channels_first=False)  # TODO: only use linear layer(done)
            )

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                simple_init=ssm_simple_init,
                # ==========================
                forward_type=forward_type,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                           drop=mlp_drop_rate, channels_first=False)

    def _forward(self, input: torch.Tensor):
        input = self.pre_mlp(input)  # TODO: in encoder use mask
        # input = self.pre_mlp(input.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()  # for conv-based pre_mlp
        if self.ssm_branch:
            x = input + self.drop_path(self.op(self.norm(input)))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


@MODELS_MMSEG.register_module()
class VSSM(nn.Module):
    def __init__(
            self,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            depths=[2, 2, 9, 2],
            dims=[96, 192, 384, 768],
            # =========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_simple_init=False,
            forward_type="v2",
            # =========================
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            # =========================
            drop_path_rate=0.1,
            patch_norm=True,
            norm_layer="LN",
            downsample_version: str = "v2",  # "v1", "v2", "v3"
            patchembed_version: str = "v1",  # "v1", "v2"
            use_checkpoint=False,
            pretrained=None,
            out_indices=(0, 1, 2, 3),
            **kwargs,
    ):
        super().__init__()
        self.out_indices = out_indices
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )

        # unused for seg task
        if norm_layer.lower() in ["ln"]:
            norm_layer: nn.Module = _NORMLAYERS[norm_layer.lower()]

        if ssm_act_layer.lower() in ["silu", "gelu", "relu"]:
            ssm_act_layer: nn.Module = _ACTLAYERS[ssm_act_layer.lower()]

        if mlp_act_layer.lower() in ["silu", "gelu", "relu"]:
            mlp_act_layer: nn.Module = _ACTLAYERS[mlp_act_layer.lower()]

        _make_patch_embed = dict(
            v1=self._make_patch_embed,
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer)

        _make_downsample = dict(
            v1=PatchMerging2D,
            v2=self._make_downsample,
            v3=self._make_downsample_v3,
            none=(lambda *_, **_k: None),
        ).get(downsample_version, None)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer],
                self.dims[i_layer + 1],
                norm_layer=norm_layer,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim=self.dims[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_simple_init=ssm_simple_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
            ))

        # self.classifier = nn.Sequential(OrderedDict(
        #     norm=norm_layer(self.num_features),  # B,H,W,C
        #     permute=Permute(0, 3, 1, 2),
        #     avgpool=nn.AdaptiveAvgPool2d(1),
        #     flatten=nn.Flatten(1),
        #     head=nn.Linear(self.num_features, num_classes),
        # ))

        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        self.apply(self._init_weights)
        self.load_pretrained(pretrained)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # used in building optimizer
    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {}

    # used in building optimizer
    # @torch.jit.ignore
    # def no_weight_decay_keywords(self):
    #     return {}

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            Permute(0, 2, 3, 1),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm):
        assert patch_size == 4
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            (Permute(0, 2, 3, 1) if patch_norm else nn.Identity()),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (Permute(0, 3, 1, 2) if patch_norm else nn.Identity()),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            Permute(0, 2, 3, 1),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
            dim=96,
            drop_path=[0.1, 0.1],
            use_checkpoint=False,
            norm_layer=nn.LayerNorm,
            downsample=nn.Identity(),
            # ===========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_simple_init=False,
            forward_type="v2",
            # ===========================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                in_dim=dim,
                hidden_dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_simple_init=ssm_simple_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                use_checkpoint=use_checkpoint,
            ))

        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks, ),
            downsample=downsample,
        ))

    # def forward(self, x: torch.Tensor):
    #     """for classification"""
    #     x = self.patch_embed(x)
    #     for layer in self.layers:
    #         x = layer(x)
    #     x = self.classifier(x)
    #     return x

    def forward(self, x):
        """for segmentation"""
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x)  # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                out = out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return outs

    def flops(self, shape=(3, 224, 224)):
        # shape = self.__input_shape__[1:]
        supported_ops = {
            "aten::silu": None,  # as relu is in _IGNORED_OPS
            "aten::neg": None,  # as relu is in _IGNORED_OPS
            "aten::exp": None,  # as relu is in _IGNORED_OPS
            "aten::flip": None,  # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScan": selective_scan_flop_jit,
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"

    # used to load ckpt from previous training code
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):

        def check_name(src, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    return True
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        return True
            return False

        def change_name(src, dst, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    state_dict[prefix + dst] = state_dict[prefix + src]
                    state_dict.pop(prefix + src)
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        new_k = prefix + dst + k[len(key):]
                        state_dict[new_k] = state_dict[k]
                        state_dict.pop(k)

        change_name("patch_embed.proj", "patch_embed.0")
        change_name("patch_embed.norm", "patch_embed.2")
        for i in range(100):
            for j in range(100):
                change_name(f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm")
                change_name(f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op")
        change_name("norm", "classifier.norm")
        change_name("head", "classifier.head")

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                             unexpected_keys,
                                             error_msgs)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return

        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")


@MODELS_MMSEG.register_module()
class DecoderExp3(BaseDecodeHead):
    """Mask-attention UperNet-Like Mamba-based decoder"""
    def __init__(self,
                 # BaseDecodeHead base settings ===========
                 # num_classes=19,  # set in BaseDecodeHead
                 # in_index=[0, 1, 2, 3],  # set in BaseDecodeHead
                 # in_channels=[96, 192, 384, 768],  # set in BaseDecodeHead
                 # channels=512,  # set in BaseDecodeHead
                 # align_corners=False,  # set in BaseDecodeHead
                 # token mask settings =========
                 gamma=4,
                 input_resolution=(512, 1024),  # same as crop
                 patch_size=4,  # same as encoder
                 t_mask=False,
                 pool_scales=(1, 2, 3, 6),
                 # vss block settings ==========
                 depths=[1, 1, 1],  # from high resolution to low
                 ssm_d_state=16,
                 ssm_ratio=2.0,
                 ssm_rank_ratio=2.0,
                 ssm_dt_rank="auto",
                 ssm_act_layer="silu",
                 ssm_conv=3,
                 ssm_conv_bias=True,
                 ssm_drop_rate=0.0,
                 ssm_simple_init=False,
                 forward_type="v2",
                 mlp_ratio=4.0,
                 mlp_act_layer="gelu",
                 mlp_drop_rate=0.0,
                 drop_path_rate=0.1,
                 norm_layer="LN",
                 use_checkpoint=False,
                 **kwargs,
                 ):
        super().__init__(input_transform='multiple_select', **kwargs)
        self.num_layers = len(depths)
        assert len(self.in_index) - 1 == self.num_layers, "length of depths must be same as num_layers!"
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            bn=nn.BatchNorm2d,
        )
        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )
        if norm_layer.lower() in ["ln"]:
            norm_layer: nn.Module = _NORMLAYERS[norm_layer.lower()]
        if ssm_act_layer.lower() in ["silu", "gelu", "relu"]:
            ssm_act_layer: nn.Module = _ACTLAYERS[ssm_act_layer.lower()]
        if mlp_act_layer.lower() in ["silu", "gelu", "relu"]:
            mlp_act_layer: nn.Module = _ACTLAYERS[mlp_act_layer.lower()]

        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners
        )
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i_layer, in_channels in enumerate(self.in_channels[:-1]):  # skip the top layer
            # linear_layer = ConvModule(
            #     in_channels,
            #     self.channels,
            #     1,
            #     conv_cfg=self.conv_cfg,
            #     norm_cfg=self.norm_cfg,
            #     act_cfg=self.act_cfg,
            #     inplace=False
            # )  # TODO: change mask-attention vss block
            l_conv = self._make_layer(
                in_dim=in_channels,
                dim=self.channels,  # TODO: change channels to low dim(done, lower gpu memory)
                drop_path=dpr[sum(depths[::-1][:self.num_layers-i_layer-1]):sum(depths[::-1][:self.num_layers-i_layer])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                # linear_layer = linear_layer,  # TODO: delete linear_layer for debug(done)
                ln=norm_layer(self.channels),
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_simple_init=ssm_simple_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
            )
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

        self.t_mask = t_mask
        if self.t_mask:
            # token mask
            self.spatial_size = []
            self.token_mask_lst = []
            for i in range(len(depths)):
                self.spatial_size.append(tuple(map(lambda x: x // (patch_size * 2**i), input_resolution)))
                init_n = int(self.spatial_size[i][0] * self.spatial_size[i][1])
                self.token_mask = torch.zeros(1, init_n, 2)  # (1, l, 2)
                self.token_mask[:, :, 0].fill_(gamma)
                self.token_mask[:, :, 1].fill_(-gamma)
                self.token_mask = nn.Parameter(self.token_mask, requires_grad=True).cuda()
                self.token_mask_lst.append(self.token_mask)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _make_layer(
            in_dim=96,
            dim=96,
            drop_path=[0.1, 0.1],
            use_checkpoint=False,
            norm_layer=nn.LayerNorm,
            # upsample=nn.Identity(),
            # linear_layer=nn.Identity(),
            ln=nn.Identity(),
            # ===========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_simple_init=False,
            forward_type="v2",
            # ===========================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                in_dim=in_dim,
                hidden_dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_simple_init=ssm_simple_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                use_checkpoint=use_checkpoint,
            ))

        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks, ),  # TODO: delete block(done)
            ln=ln,
            # linear_layer=linear_layer,
        ))

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i].permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward_token_mask(self, inputs):
        """token mask operation"""
        inputs_mask = []
        hard_keep_decision_list = []
        for i, input in enumerate(inputs[:-1]):  # discard the last layer
            B, C, H, W = input.shape
            # token_mask = self.token_mask[i]
            # resize input size
            token_mask = F.interpolate(
                self.token_mask_lst[i].reshape(1, int(self.spatial_size[i][0]), int(self.spatial_size[i][1]), 2).permute(0, 3, 1, 2).contiguous(), size=(H, W), mode='bilinear'
            ).contiguous().view(1, H*W, 2)  # TODO : set self.spatial_size(done)
            token_masks = token_mask.expand(B, -1, -1)  # (B, l, 2)
            if self.training:
                mask_thresholded = F.gumbel_softmax(token_masks, tau=1, hard=True)
                mask_thresholded = mask_thresholded[..., 0]
            else:
                mask_thresholded = token_masks[..., 0] > token_masks[..., 1]
            mask_thresholded = mask_thresholded.reshape(B, mask_thresholded.shape[1], 1)  # (b, l, 1) for calculate token mask loss
            hard_keep_decision = mask_thresholded
            hard_keep_decision_list.append(hard_keep_decision.reshape(B, H, W, 1))
            mask_thresholded = mask_thresholded.permute(0, 2, 1).contiguous().view(B, 1, H, W)  # (b, 1, h, w)
            # inputs[i] = inputs[i] * mask_thresholded.to(inputs[i].dtype)  # token mask, TODO: design channel-aware token mask
            inputs_mask.append(inputs[i] * mask_thresholded.to(inputs[i].dtype))
            # inputs_mask.append(inputs[i])
        inputs_mask.append(inputs[-1])

        return inputs_mask, hard_keep_decision_list

    def forward(self, inputs):  # TODO: mask_loss: recode loss_by_feat(done)
        """
        mask-attention
        inputs: list, member size: (b, c, h, w), len=4 for each branch
        mask: list, member size: (1, l, 2), len=3 for each branch (apart from last layer), 2 for mask or not
        """
        inputs_mask = inputs  # for auxiliary_head, not inplace operation to inputs
        hard_keep_decision = None
        if self.t_mask:
            inputs_mask, hard_keep_decision = self.forward_token_mask(inputs)  # TODO: for auxiliary_head, inputs change or not(done)

        output = self._forward_feature(inputs_mask)
        output = self.cls_seg(output)

        # if not self.training:
        #     import pickle
        #     print('EXP3:saving encoder_mask_lst')
        #     with open('/data/ljh/data/MVM/city/city2others/tiny/decoder_mask_lst.pkl', 'wb') as f:
        #         pickle.dump(hard_keep_decision, f)
        #     print('EXP3:save decoder_mask_lst done')

        return output, hard_keep_decision

    def predict(self, inputs, batch_img_metas, test_cfg):
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        seg_logits, hard_keep_decision = self.forward(inputs)

        return self.predict_by_feat(seg_logits, batch_img_metas)

    def loss(self, inputs, batch_data_samples, train_cfg):
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits, hard_keep_decision = self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, batch_data_samples, hard_keep_decision=hard_keep_decision)
        return losses

    def loss_by_feat(self, seg_logits, batch_data_samples, hard_keep_decision=None):  # TODO:mask loss
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.
            hard_keep_decision (None or List[Tensor]): len 3, tensor size (b, l, 1)

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                if loss_decode.loss_name == 'loss_mask' and hard_keep_decision is not None:
                    loss[loss_decode.loss_name] = loss_decode(
                        hard_keep_decision,
                        ignore_index=self.ignore_index
                    )
                else:
                    loss[loss_decode.loss_name] = loss_decode(
                        seg_logits,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
            else:
                if loss_decode.loss_name == 'loss_mask' and hard_keep_decision is not None:
                    loss[loss_decode.loss_name] += loss_decode(
                        hard_keep_decision,
                        ignore_index=self.ignore_index
                    )
                else:
                    loss[loss_decode.loss_name] += loss_decode(
                        seg_logits,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss


@MODELS_MMSEG.register_module()
class MaskTokenLoss(nn.Module):
    """token mask loss"""
    def __init__(self,
                 keep_ratio=0.9,
                 loss_weight=1.0,
                 loss_name='loss_mask',
                 avg_non_ignore=False):
        super().__init__()
        self._loss_name = loss_name
        self.keep_ratio = keep_ratio
        self.loss_weight = loss_weight

    def forward(self,
                hard_keep_decision,
                ignore_index=-100,
                **kwargs):
        """Forward function."""
        pred_loss = 0.0
        for keep_mask in hard_keep_decision:
            keep_mask = keep_mask.mean()
            pred_loss = pred_loss + ((keep_mask - self.keep_ratio) ** 2).mean()

        pred_loss = pred_loss * self.loss_weight

        return pred_loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


@MODELS_MMSEG.register_module()
class SegExp3(EncoderDecoder):
    """token mask seg"""
    def _forward(self, inputs, data_samples=None):
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(inputs)
        return self.decode_head.forward(x)[0]





#============================================================================== EXP4 ==============================================================================
#TODO: MVM and L_mask(done)
#============================================================================== EXP4 ==============================================================================
class MaskVSSBlock(nn.Module):  # TODO: mask before SSM
    def __init__(
            self,
            # for mask ================
            t_mask=False,
            spatial_size=(16, 16),
            gamma=4,
            in_dim: int = 0,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_simple_init=False,
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.pre_mlp = nn.Identity()  # pre-deal before block to map in_dim to out_dim
        if in_dim != hidden_dim:
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.pre_mlp = nn.Sequential(
                # ConvModule(
                #     in_dim,
                #     hidden_dim,
                #     1,
                #     norm_cfg=dict(type='SyncBN', requires_grad=True),
                #     act_cfg=dict(type='ReLU'),
                #     inplace=False
                # ),
                # norm_layer(in_dim),  # encoder output execute norm
                nn.Linear(in_dim, hidden_dim),  # TODO: use conv1*1 to change dim(done, same as linear_layer)
                # Mlp(in_features=in_dim, hidden_features=mlp_hidden_dim, out_features=hidden_dim, act_layer=mlp_act_layer,
                #     drop=mlp_drop_rate, channels_first=False)  # TODO: only use linear layer(done)
            )

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                simple_init=ssm_simple_init,
                # ==========================
                forward_type=forward_type,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                           drop=mlp_drop_rate, channels_first=False)

        self.t_mask = t_mask
        if self.t_mask:
            # token mask
            self.spatial_size = spatial_size
            init_n = int(self.spatial_size[0] * self.spatial_size[1])
            self.token_mask = torch.zeros(1, init_n, 2)  # (1, l, 2)
            # self.token_mask[:, :, 0].fill_(gamma)
            # self.token_mask[:, :, 1].fill_(-gamma)
            self.token_mask = nn.Parameter(self.token_mask, requires_grad=True)
            nn.init.uniform_(self.token_mask.data, -gamma, gamma)

    def forward_token_mask(self, input):
        """token mask operation"""
        if not self.t_mask:
            return input, None
        B, H, W, C = input.shape
        token_mask = F.interpolate(
            self.token_mask.reshape(1, int(self.spatial_size[0]), int(self.spatial_size[1]), 2).permute(0, 3, 1, 2).contiguous(),
            size=(H, W), mode='bilinear'
        ).contiguous().view(1, H * W, 2)
        token_mask = token_mask.expand(B, -1, -1)  # (B, l, 2)
        if self.training:
            mask_thresholded = F.gumbel_softmax(token_mask, tau=1, hard=True)
            mask_thresholded = mask_thresholded[..., 0]
        else:
            mask_thresholded = token_mask[..., 0] > token_mask[..., 1]
        mask_thresholded = mask_thresholded.reshape(B, mask_thresholded.shape[1], 1)
        mask = mask_thresholded.reshape(B, H, W, 1) # (b, h, w, 1) for calculate token mask loss
        mask_thresholded = mask_thresholded.contiguous().view(B, H, W, 1)  # (b, h, w, 1)
        input = input * mask_thresholded.to(input.dtype) + input

        return input, mask

    def _forward(self, input):
        input, mask = self.forward_token_mask(input)  # for mask
        input = self.pre_mlp(input)  # TODO: in encoder use mask(done)
        # input = self.pre_mlp(input.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()  # for conv-based pre_mlp
        if self.ssm_branch:
            x = input + self.drop_path(self.op(self.norm(input)))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN

        return x, mask

    def forward(self, input):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


@MODELS_MMSEG.register_module()
class MaskVSSM(nn.Module):
    def __init__(
            self,
            # for mask ===========
            t_mask=False,
            input_resolution=(512, 1024),
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            depths=[2, 2, 9, 2],
            dims=[96, 192, 384, 768],
            # =========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_simple_init=False,
            forward_type="v2",
            # =========================
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            # =========================
            drop_path_rate=0.1,
            patch_norm=True,
            norm_layer="LN",
            downsample_version: str = "v2",  # "v1", "v2", "v3"
            patchembed_version: str = "v1",  # "v1", "v2"
            use_checkpoint=False,
            pretrained=None,
            out_indices=(0, 1, 2, 3),
            **kwargs,
    ):
        super().__init__()
        self.t_mask = t_mask
        self.out_indices = out_indices
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )

        # unused for seg task
        if norm_layer.lower() in ["ln"]:
            norm_layer: nn.Module = _NORMLAYERS[norm_layer.lower()]

        if ssm_act_layer.lower() in ["silu", "gelu", "relu"]:
            ssm_act_layer: nn.Module = _ACTLAYERS[ssm_act_layer.lower()]

        if mlp_act_layer.lower() in ["silu", "gelu", "relu"]:
            mlp_act_layer: nn.Module = _ACTLAYERS[mlp_act_layer.lower()]

        _make_patch_embed = dict(
            v1=self._make_patch_embed,
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer)

        _make_downsample = dict(
            v1=PatchMerging2D,
            v2=self._make_downsample,
            v3=self._make_downsample_v3,
            none=(lambda *_, **_k: None),
        ).get(downsample_version, None)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer],
                self.dims[i_layer + 1],
                norm_layer=norm_layer,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            spatial_size = tuple(map(lambda x: x // (patch_size * 2**i_layer), input_resolution))

            self.layers.append(self._make_layer(
                t_mask=True if t_mask and (i_layer < self.num_layers - 1) else False,
                spatial_size=spatial_size,
                dim=self.dims[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_simple_init=ssm_simple_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
            ))

        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        self.apply(self._init_weights)
        self.load_pretrained(pretrained)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # used in building optimizer
    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {}

    # used in building optimizer
    # @torch.jit.ignore
    # def no_weight_decay_keywords(self):
    #     return {}

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            Permute(0, 2, 3, 1),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm):
        assert patch_size == 4
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            (Permute(0, 2, 3, 1) if patch_norm else nn.Identity()),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (Permute(0, 3, 1, 2) if patch_norm else nn.Identity()),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            Permute(0, 2, 3, 1),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
            t_mask=False,
            spatial_size=(16, 16),
            dim=96,
            drop_path=[0.1, 0.1],
            use_checkpoint=False,
            norm_layer=nn.LayerNorm,
            downsample=nn.Identity(),
            # ===========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_simple_init=False,
            forward_type="v2",
            # ===========================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(MaskVSSBlock(
                t_mask=t_mask,
                spatial_size=spatial_size,
                in_dim=dim,
                hidden_dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_simple_init=ssm_simple_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                use_checkpoint=use_checkpoint,
            ))

        # return nn.Sequential(OrderedDict(
        #     blocks=nn.Sequential(*blocks, ),
        #     downsample=downsample,
        # ))

        return nn.Sequential(OrderedDict(
            blocks=nn.ModuleList(blocks),
            downsample=downsample,
        ))

    def forward(self, x):
        """for segmentation"""
        encoder_mask_lst = []  # 179622

        # def layer_forward(l, x):
        #     x = l.blocks(x)
        #     y = l.downsample(x)
        #     return x, y

        def layer_forward(l, x, i_layer):
            for blk in l.blocks:
                x, mask = blk(x)
                if mask is not None:
                    encoder_mask_lst.append(mask)
            y = l.downsample(x)
            return x, y

        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x, i)  # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                out = out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        # if not self.training and self.t_mask:
        #     print(len(encoder_mask_lst))
        #     for keep_mask in encoder_mask_lst[:2]:
        #         keep_mask = torch.sum(keep_mask.float())
        #         print('stage1 encoder mask ratio:', keep_mask)  # 116935  0.65
        #     for keep_mask in encoder_mask_lst[2:4]:
        #         keep_mask = torch.sum(keep_mask.float())
        #         print('stage2 encoder mask ratio:', keep_mask)  # 29321  0.16
        #     for keep_mask in encoder_mask_lst[4:13]:
        #         keep_mask = torch.sum(keep_mask.float())
        #         print('stage3 encoder mask ratio:', keep_mask)  # 33366  0.19

        return outs, encoder_mask_lst

    def flops(self, shape=(3, 224, 224)):
        # shape = self.__input_shape__[1:]
        supported_ops = {
            "aten::silu": None,  # as relu is in _IGNORED_OPS
            "aten::neg": None,  # as relu is in _IGNORED_OPS
            "aten::exp": None,  # as relu is in _IGNORED_OPS
            "aten::flip": None,  # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScan": selective_scan_flop_jit,
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"

    # used to load ckpt from previous training code
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):

        def check_name(src, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    return True
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        return True
            return False

        def change_name(src, dst, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    state_dict[prefix + dst] = state_dict[prefix + src]
                    state_dict.pop(prefix + src)
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        new_k = prefix + dst + k[len(key):]
                        state_dict[new_k] = state_dict[k]
                        state_dict.pop(k)

        change_name("patch_embed.proj", "patch_embed.0")
        change_name("patch_embed.norm", "patch_embed.2")
        for i in range(100):
            for j in range(100):
                change_name(f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm")
                change_name(f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op")
        change_name("norm", "classifier.norm")
        change_name("head", "classifier.head")

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                             unexpected_keys,
                                             error_msgs)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return

        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")


@MODELS_MMSEG.register_module()
class SegExp4(EncoderDecoder):
    """token mask seg"""
    def extract_feat(self, inputs):
        """Extract features from images."""
        x, encoder_mask_lst = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x, encoder_mask_lst

    def encode_decode(self, inputs,
                      batch_img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x, encoder_mask_lst = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict(x, batch_img_metas,
                                              self.test_cfg)

        return seg_logits

    def _decode_head_forward_train(self, inputs, data_samples, encoder_mask_lst=None):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg, encoder_mask_lst=encoder_mask_lst)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def loss(self, inputs, data_samples) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x, encoder_mask_lst = self.extract_feat(inputs)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, data_samples, encoder_mask_lst=encoder_mask_lst)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses

    def _forward(self, inputs, data_samples=None):
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x, encoder_mask_lst = self.extract_feat(inputs)
        return self.decode_head.forward(x)[0]


@MODELS_MMSEG.register_module()
class DecoderExp4(BaseDecodeHead):
    """Mamba-based decoder"""
    def __init__(self,
                 # BaseDecodeHead base settings ===========
                 # num_classes=19,  # set in BaseDecodeHead
                 # in_index=[0, 1, 2, 3],  # set in BaseDecodeHead
                 # in_channels=[96, 192, 384, 768],  # set in BaseDecodeHead
                 # channels=512,  # set in BaseDecodeHead
                 # align_corners=False,  # set in BaseDecodeHead
                 # token mask settings =========
                 input_resolution=(512, 1024),  # same as crop
                 patch_size=4,  # same as encoder
                 t_mask=False,
                 pool_scales=(1, 2, 3, 6),
                 # vss block settings ==========
                 depths=[1, 1, 1],  # from high resolution to low
                 ssm_d_state=16,
                 ssm_ratio=2.0,
                 ssm_rank_ratio=2.0,
                 ssm_dt_rank="auto",
                 ssm_act_layer="silu",
                 ssm_conv=3,
                 ssm_conv_bias=True,
                 ssm_drop_rate=0.0,
                 ssm_simple_init=False,
                 forward_type="v2",
                 mlp_ratio=4.0,
                 mlp_act_layer="gelu",
                 mlp_drop_rate=0.0,
                 drop_path_rate=0.1,
                 norm_layer="LN",
                 use_checkpoint=False,
                 **kwargs,
                 ):
        super().__init__(input_transform='multiple_select', **kwargs)
        self.t_mask = t_mask
        self.num_layers = len(depths)  # mask layer num
        assert len(self.in_index) - 1 == self.num_layers, "length of depths must be same as num_layers!"
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            bn=nn.BatchNorm2d,
        )
        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )
        if norm_layer.lower() in ["ln"]:
            norm_layer: nn.Module = _NORMLAYERS[norm_layer.lower()]
        if ssm_act_layer.lower() in ["silu", "gelu", "relu"]:
            ssm_act_layer: nn.Module = _ACTLAYERS[ssm_act_layer.lower()]
        if mlp_act_layer.lower() in ["silu", "gelu", "relu"]:
            mlp_act_layer: nn.Module = _ACTLAYERS[mlp_act_layer.lower()]

        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners
        )
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i_layer, in_channels in enumerate(self.in_channels[:-1]):  # skip the top layer
            # linear_layer = ConvModule(
            #     in_channels,
            #     self.channels,
            #     1,
            #     conv_cfg=self.conv_cfg,
            #     norm_cfg=self.norm_cfg,
            #     act_cfg=self.act_cfg,
            #     inplace=False
            # )  # TODO: change mask-attention vss block(done)
            spatial_size = tuple(map(lambda x: x // (patch_size * 2 ** i_layer), input_resolution))
            l_conv = self._make_layer(
                t_mask=t_mask,
                spatial_size=spatial_size,
                in_dim=in_channels,
                dim=self.channels,  # TODO: change channels to low dim(done, lower gpu memory)
                drop_path=dpr[sum(depths[::-1][:self.num_layers-i_layer-1]):sum(depths[::-1][:self.num_layers-i_layer])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                # linear_layer = linear_layer,  # TODO: delete linear_layer for debug(done)
                ln=norm_layer(self.channels),
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_simple_init=ssm_simple_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
            )
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _make_layer(
            t_mask=False,
            spatial_size=(16, 16),
            in_dim=96,
            dim=96,
            drop_path=[0.1, 0.1],
            use_checkpoint=False,
            norm_layer=nn.LayerNorm,
            # upsample=nn.Identity(),
            # linear_layer=nn.Identity(),
            ln=nn.Identity(),
            # ===========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_simple_init=False,
            forward_type="v2",
            # ===========================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(MaskVSSBlock(
                t_mask=t_mask,
                spatial_size=spatial_size,
                in_dim=in_dim,
                hidden_dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_simple_init=ssm_simple_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                use_checkpoint=use_checkpoint,
            ))

        return nn.Sequential(OrderedDict(
            blocks=nn.ModuleList(blocks),  # TODO: delete block(done)
            ln=ln,
        ))

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        # laterals = [
        #     lateral_conv(inputs[i].permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        #     for i, lateral_conv in enumerate(self.lateral_convs)
        # ]
        decoder_mask_lst = []
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            x = inputs[i].permute(0, 2, 3, 1).contiguous()  # B, H, W, C
            for blk in lateral_conv.blocks:
                x, mask = blk(x)
                if mask is not None:
                    decoder_mask_lst.append(mask)
            laterals.append(lateral_conv.ln(x).permute(0, 3, 1, 2).contiguous())

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats, decoder_mask_lst

    def forward(self, inputs):  # TODO: mask_loss: recode loss_by_feat(done)
        """
        mask-attention
        inputs: list, member size: (b, c, h, w), len=4 for each branch
        mask: list, member size: (1, l, 2), len=3 for each branch (apart from last layer), 2 for mask or not
        """
        # output = self._forward_feature(inputs)
        output, decoder_mask_lst = self._forward_feature(inputs)
        output = self.cls_seg(output)

        # if not self.training:
        #     import pickle
        #     print('saving encoder_mask_lst')
        #     with open('/data/ljh/data/MVM/city/city2others/tiny/decoder_mask_lst.pkl', 'wb') as f:
        #         pickle.dump(decoder_mask_lst, f)
        #     print('save decoder_mask_lst done')

        return output, decoder_mask_lst

    def predict(self, inputs, batch_img_metas, test_cfg):
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        seg_logits, decoder_mask_lst = self.forward(inputs)

        return self.predict_by_feat(seg_logits, batch_img_metas)

    def loss(self, inputs, batch_data_samples, train_cfg, encoder_mask_lst=None):
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits, decoder_mask_lst = self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, batch_data_samples, encoder_mask_lst=encoder_mask_lst, decoder_mask_lst=decoder_mask_lst)
        return losses

    def loss_by_feat(self, seg_logits, batch_data_samples, encoder_mask_lst=None, decoder_mask_lst=None):  # TODO:mask loss(done)
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.
            hard_keep_decision (None or List[Tensor]): len 3, tensor size (b, l, 1)

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        # print(seg_label.shape)  # B, H, W
        # print(seg_logits.shape)  # B, C, H, W
        def generate_mask_label():
            seg_mask = torch.argmax(seg_logits, dim=1)
            assert seg_label.shape == seg_mask.shape, 'shape error!'
            mask_label = (seg_mask == seg_label).to(seg_logits.dtype)
            return mask_label

        mask_label = generate_mask_label()

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                if loss_decode.loss_name == 'loss_mask' and self.t_mask:
                    loss[loss_decode.loss_name] = loss_decode(
                        encoder_mask_lst=encoder_mask_lst,
                        decoder_mask_lst=decoder_mask_lst,
                        mask_label=mask_label,
                        ignore_index=self.ignore_index
                    )
                else:
                    loss[loss_decode.loss_name] = loss_decode(
                        seg_logits,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
            else:
                if loss_decode.loss_name == 'loss_mask' and self.t_mask:
                    loss[loss_decode.loss_name] += loss_decode(
                        encoder_mask_lst=encoder_mask_lst,
                        decoder_mask_lst=decoder_mask_lst,
                        mask_label=mask_label,
                        ignore_index=self.ignore_index
                    )
                else:
                    loss[loss_decode.loss_name] += loss_decode(
                        seg_logits,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss


@MODELS_MMSEG.register_module()
class MaskLoss(nn.Module):
    """mask loss"""
    def __init__(self,
                 keep_ratio=0.5,
                 loss_e_weight=1.0,
                 loss_d_weight=1.0,
                 loss_name='loss_mask',
                 avg_non_ignore=False):
        super().__init__()
        self._loss_name = loss_name
        self.keep_ratio = keep_ratio
        self.loss_e_weight = loss_e_weight
        self.loss_d_weight = loss_d_weight
        self.l2_loss = nn.MSELoss()

    def forward(self,
                encoder_mask_lst,
                decoder_mask_lst,
                mask_label,
                ignore_index=-100,
                **kwargs):
        """Forward function."""
        mask_loss_encoder = 0
        for keep_mask in encoder_mask_lst:
            keep_mask = keep_mask.mean()
            mask_loss_encoder = mask_loss_encoder + ((keep_mask - self.keep_ratio) ** 2).mean()

        mask_label = mask_label.unsqueeze(dim=1)  # TODO: resize:label to keep_mask or keep_mask to label
        for i, keep_mask in enumerate(decoder_mask_lst):  # TODO: mask loss for decoder(done)
            _, H, W, _ = keep_mask.shape
            mask = F.interpolate(mask_label, size=(H, W), mode='bilinear')
            if i == 0:
                mask_loss_decoder = self.l2_loss(keep_mask.permute(0, 3, 1, 2).contiguous(), mask)
            else:
                mask_loss_decoder += self.l2_loss(keep_mask.permute(0, 3, 1, 2).contiguous(), mask)

        mask_loss_encoder = mask_loss_encoder * self.loss_e_weight
        mask_loss_decoder = mask_loss_decoder * self.loss_d_weight
        mask_loss = mask_loss_encoder + mask_loss_decoder

        return mask_loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name