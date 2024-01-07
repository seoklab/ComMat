# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import reduce
import importlib
import math
import sys
from operator import mul

import torch
import torch.nn as nn
from typing import Optional, Tuple, Sequence

from openfold.model.primitives import Linear, LayerNorm, ipa_point_weights_init_
from openfold.np.residue_constants import (
    restype_rigid_group_default_frame,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
)
from openfold.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
from openfold.utils.precision_utils import is_fp16_enabled
from openfold.utils.rigid_utils import Rotation, Rigid
from openfold.utils.tensor_utils import (
    dict_multimap,
    permute_final_dims,
    flatten_final_dims,
)

attn_core_inplace_cuda = importlib.import_module("attn_core_inplace_cuda")


class AngleResnetBlock(nn.Module):
    def __init__(self, c_hidden):
        """
        Args:
            c_hidden:
                Hidden channel dimension
        """
        super(AngleResnetBlock, self).__init__()

        self.c_hidden = c_hidden

        self.linear_1 = Linear(self.c_hidden, self.c_hidden, init="relu")
        self.linear_2 = Linear(self.c_hidden, self.c_hidden, init="final")

        self.relu = nn.ReLU()

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        s_initial = a

        a = self.relu(a)
        a = self.linear_1(a)
        a = self.relu(a)
        a = self.linear_2(a)

        return a + s_initial


class AngleResnet(nn.Module):
    """
    Implements Algorithm 20, lines 11-14
    """

    def __init__(self, c_in, c_hidden, no_blocks, no_angles, epsilon):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Hidden channel dimension
            no_blocks:
                Number of resnet blocks
            no_angles:
                Number of torsion angles to generate
            epsilon:
                Small constant for normalization
        """
        super(AngleResnet, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.no_angles = no_angles
        self.eps = epsilon

        self.linear_in = Linear(self.c_in, self.c_hidden)
        self.linear_initial = Linear(self.c_in, self.c_hidden)

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = AngleResnetBlock(c_hidden=self.c_hidden)
            self.layers.append(layer)

        self.linear_out = Linear(self.c_hidden, self.no_angles * 2)

        self.relu = nn.ReLU()

    def forward(
        self, s: torch.Tensor, s_initial: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, C_hidden] single embedding
            s_initial:
                [*, C_hidden] single embedding as of the start of the
                StructureModule
        Returns:
            [*, no_angles, 2] predicted angles
        """
        # NOTE: The ReLU's applied to the inputs are absent from the supplement
        # pseudocode but present in the source. For maximal compatibility with
        # the pretrained weights, I'm going with the source.

        # [*, C_hidden]
        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial

        for l in self.layers:
            s = l(s)

        s = self.relu(s)

        # [*, no_angles * 2]
        s = self.linear_out(s)

        # [*, no_angles, 2]
        s = s.view(s.shape[:-1] + (-1, 2))

        unnormalized_s = s
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(s**2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        s = s / norm_denom

        return unnormalized_s, s


class AngleResnet_backbone(nn.Module):
    """
    Implements Algorithm 20, lines 11-14
    """

    def __init__(self, c_in, c_hidden, no_blocks, no_angles, epsilon):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Hidden channel dimension
            no_blocks:
                Number of resnet blocks
            no_angles:
                Number of torsion angles to generate
            epsilon:
                Small constant for normalization
        """
        super(AngleResnet_backbone, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.no_angles = no_angles
        self.eps = epsilon

        self.linear_in = Linear(self.c_in, self.c_hidden)
        self.linear_initial = Linear(self.c_in, self.c_hidden)

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = AngleResnetBlock(c_hidden=self.c_hidden)
            self.layers.append(layer)

        self.linear_out = Linear(self.c_hidden, self.no_angles * 2)

        self.relu = nn.ReLU()

    def forward(
        self, s: torch.Tensor, s_initial: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, C_hidden] single embedding
            s_initial:
                [*, C_hidden] single embedding as of the start of the
                StructureModule
        Returns:
            [*, no_angles, 2] predicted angles
        """
        # NOTE: The ReLU's applied to the inputs are absent from the supplement
        # pseudocode but present in the source. For maximal compatibility with
        # the pretrained weights, I'm going with the source.

        # [*, C_hidden]
        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial

        for l in self.layers:
            s = l(s)

        s = self.relu(s)

        # [*, no_angles * 2]
        s = self.linear_out(s)

        # [*, no_angles, 2]
        s = s.view(s.shape[:-1] + (-1, 2))

        unnormalized_s = s
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(s**2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        s = s / norm_denom

        return unnormalized_s, s


class InvariantPointAttention(nn.Module):
    """
    Implements Algorithm 22.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_qk_points: int,
        no_v_points: int,
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super(InvariantPointAttention, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        hpv = self.no_heads * self.no_v_points * 3

        self.linear_b = Linear(self.c_z, self.no_heads)

        self.head_weights = nn.Parameter(torch.zeros((no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.no_heads * (
            self.c_z + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
        self,
        s: torch.Tensor,
        z: Optional[torch.Tensor],
        r: Rigid,
        mask: torch.Tensor,
        inplace_safe: bool = False,
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        if _offload_inference and inplace_safe:
            z = _z_reference_list
        else:
            z = [z]
        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)
        # print ("1!", q.shape,k.shape,v.shape)
        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3))

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)

        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )
        # print ("2!", q_pts.shape,k_pts.shape,v_pts.shape)

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z[0])

        if _offload_inference:
            assert sys.getrefcount(z[0]) == 2
            z[0] = z[0].cpu()

        # [*, H, N_res, N_res]
        if is_fp16_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                a = torch.matmul(
                    permute_final_dims(q.float(), (1, 0, 2)),  # [*, H, N_res, C_hidden]
                    permute_final_dims(k.float(), (1, 2, 0)),  # [*, H, C_hidden, N_res]
                )
        else:
            a = torch.matmul(
                permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
                permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
            )

        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1))

        # [*, N_res, N_res, H, P_q, 3]
        pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        if inplace_safe:
            pt_att *= pt_att
        else:
            pt_att = pt_att**2

        # [*, N_res, N_res, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        if inplace_safe:
            pt_att *= head_weights
        else:
            pt_att = pt_att * head_weights
        # print ("6!", pt_att.shape)
        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)
        # print ("7",square_mask.shape)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        if inplace_safe:
            a += pt_att
            del pt_att
            a += square_mask.unsqueeze(-3)
            # in-place softmax
            attn_core_inplace_cuda.forward_(
                a,
                reduce(mul, a.shape[:-1]),
                a.shape[-1],
            )
        else:
            a = a + pt_att
            a = a + square_mask.unsqueeze(-3)
            a = self.softmax(a)
        # print ("9",a.shape)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(a, v.transpose(-2, -3).to(dtype=a.dtype)).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)
        # print ("10",o.shape)

        # [*, H, 3, N_res, P_v]
        if inplace_safe:
            v_pts = permute_final_dims(v_pts, (1, 3, 0, 2))
            o_pt = [torch.matmul(a, v.to(a.dtype)) for v in torch.unbind(v_pts, dim=-3)]
            o_pt = torch.stack(o_pt, dim=-3)
        else:
            o_pt = torch.sum(
                (
                    a[..., None, :, :, None]
                    * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
                ),
                dim=-2,
            )

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)
        # print ("11",o_pt.shape)

        # [*, N_res, H * P_v]
        o_pt_norm = flatten_final_dims(
            torch.sqrt(torch.sum(o_pt**2, dim=-1) + self.eps), 2
        )

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)
        # print ("12",o_pt.shape)

        if _offload_inference:
            z[0] = z[0].to(o_pt.device)

        # [*, N_res, H, C_z]
        o_pair = torch.matmul(a.transpose(-2, -3), z[0].to(dtype=a.dtype))
        # print ("13",o_pt.shape)

        # [*, N_res, H * C_z]
        o_pair = flatten_final_dims(o_pair, 2)
        # print ("14",o_pt.shape)

        # [*, N_res, C_s]
        s = self.linear_out(
            torch.cat((o, *torch.unbind(o_pt, dim=-1), o_pt_norm, o_pair), dim=-1).to(
                dtype=z[0].dtype
            )
        )

        return s


class InvariantPointAttention_all_frames(nn.Module):
    """
    Implements Algorithm 22.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_qk_points: int,
        no_v_points: int,
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super(InvariantPointAttention_all_frames, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        hpv = self.no_heads * self.no_v_points * 3

        self.linear_b = Linear(self.c_z, self.no_heads)

        self.head_weights = nn.Parameter(torch.zeros((no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.no_heads * (
            self.c_z + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()
        self.layer_expand_sidechain = nn.Linear(self.c_s, self.c_s * 5)
        self.layer_norm_ipa = LayerNorm(self.c_s)
        self.weight = nn.Linear(self.c_s, 1)

    def forward(
        self,
        s: torch.Tensor,
        z: Optional[torch.Tensor],
        r: Rigid,
        mask: torch.Tensor,
        inplace_safe: bool = False,
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        if _offload_inference and inplace_safe:
            z = _z_reference_list
        else:
            z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################
        ulr_mask = mask[..., 0]  # [*, N_res]
        fg_mask = mask[..., 1:]  # [*, N_res, 4]
        weight = self.weight(s)  # [*, N_res, 1]
        weight = self.softmax(weight)  # [*, N_res, 1]
        s = self.layer_expand_sidechain(s)  # [*, N_res, C_s * 5]
        s = s.reshape(*s.shape[:-1], 5, self.c_s)  # [*, N_res, 5, C_s]
        s = self.layer_norm_ipa(s)

        q = self.linear_q(s)  # [*, N_res, 5, c_s, H * C_hidden]
        kv = self.linear_kv(s)  # [*, N_res, 5, c_s, 2 * H * C_hidden]

        # [*, N_res, 5, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        bb_q = q[..., :1, :, :]  # [*, N_res, 1, H, C_hidden]
        fg_q = q[..., 1:, :, :]  # [*, N_res, 4, H, C_hidden]
        fg_q = fg_q * fg_mask[..., None, None]
        fg_q = fg_q.sum(dim=-3, keepdim=True)
        fg_q = fg_q / (fg_mask.sum(dim=-1, keepdim=True)[..., None, None] + 1e-8)
        weight_q = weight[..., None, None].expand_as(bb_q)
        q = weight_q * bb_q + (1 - weight_q) * fg_q
        q = q.reshape(*q.shape[:-3], *q.shape[-2:])  # [*, N_res, H, C_hidden]
        # [*, N_res, 5, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))
        bb_kv = kv[..., :1, :, :]  # [*, N_res, 1, H, 2 * C_hidden]
        fg_kv = kv[..., 1:, :, :]  # [*, N_res, 4, H, 2 * C_hidden]
        fg_kv = fg_kv * fg_mask[..., None, None]
        fg_kv = fg_kv.sum(dim=-3, keepdim=True)
        fg_kv = fg_kv / (fg_mask.sum(dim=-1, keepdim=True)[..., None, None] + 1e-8)
        weight_kv = weight[..., None, None].expand_as(bb_kv)
        kv = weight_kv * bb_kv + (1 - weight_kv) * fg_kv
        kv = kv.reshape(*kv.shape[:-3], *kv.shape[-2:])  # [*, N_res, H, C_hidden]
        # [*, N_res, 5, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)
        # [*, N_res, 5, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, 5, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        # r = [*, N_res, 5]
        q_pts = r[..., None].apply(q_pts)

        # [*, N_res, 5, H, P_q, 3]
        q_pts = q_pts.view(q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3))
        bb_q_pts = q_pts[..., :1, :, :, :]  # [*, N_res, 1, H, P_q, 3]
        fg_q_pts = q_pts[..., 1:, :, :, :]  # [*, N_res, 4, H, P_q, 3]
        fg_q_pts = fg_q_pts * fg_mask[..., None, None, None]
        fg_q_pts = fg_q_pts.sum(dim=-4, keepdim=True)  # [*, N_res, 4, H, P_q, 3]
        fg_q_pts = fg_q_pts / (
            fg_mask.sum(dim=-1, keepdim=True)[..., None, None, None] + 1e-8
        )  # [*, N_res, 1, H, P_q, 3]
        weight_q = weight[..., None, None, None].expand_as(bb_q_pts)
        q_pts = (
            weight_q * bb_q_pts + (1 - weight_q) * fg_q_pts
        )  # [*, N_res, 1, H, P_q, 3]
        q_pts = q_pts.reshape(
            *q_pts.shape[:-4], *q_pts.shape[-3:]
        )  # [*, N_res, H,  P_q, 3]
        # [*, N_res, 5, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)
        # [*, N_res, 5, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)
        # [*, N_res, 5, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))
        # [*, N_res, 5, 12, 12, 3]
        bb_kv_pts = kv_pts[..., :1, :, :, :]  # [*, N_res, 1, H, P_q, 3]
        fg_kv_pts = kv_pts[..., 1:, :, :, :]  # [*, N_res, 4, H, P_q, 3]
        fg_kv_pts = fg_kv_pts * fg_mask[..., None, None, None]
        fg_kv_pts = fg_kv_pts.sum(dim=-4, keepdim=True)  # [*, N_res, 4, H, P_q, 3]
        fg_kv_pts = fg_kv_pts / (
            fg_mask.sum(dim=-1, keepdim=True)[..., None, None, None] + 1e-8
        )  # [*, N_res, 1, H, P_q, 3]
        weight_kv = weight[..., None, None, None].expand_as(bb_kv_pts)
        kv_pts = weight_kv * bb_kv_pts + (1 - weight_kv) * fg_kv_pts
        # [*, N_res, 1, H, P_q/P_v, 3]
        kv_pts = kv_pts.reshape(
            *kv_pts.shape[:-4], *kv_pts.shape[-3:]
        )  # [*, N_res, H,  P_q, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )
        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z[0])

        if _offload_inference:
            assert sys.getrefcount(z[0]) == 2
            z[0] = z[0].cpu()

        # [*, H, N_res, N_res]
        if is_fp16_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                a = torch.matmul(
                    permute_final_dims(q.float(), (1, 0, 2)),  # [*, H, N_res, C_hidden]
                    permute_final_dims(k.float(), (1, 2, 0)),  # [*, H, C_hidden, N_res]
                )
        else:
            a = torch.matmul(
                permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
                permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
            )

        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1))

        # [*, N_res, N_res, H, P_q, 3]
        pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        if inplace_safe:
            pt_att *= pt_att
        else:
            pt_att = pt_att**2

        # [*, N_res, N_res, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        if inplace_safe:
            pt_att *= head_weights
        else:
            pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, N_res]
        square_mask = mask[..., 0].unsqueeze(-1) * mask[..., 0].unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        if inplace_safe:
            a += pt_att
            del pt_att
            a += square_mask.unsqueeze(-3)
            # in-place softmax
            attn_core_inplace_cuda.forward_(
                a,
                reduce(mul, a.shape[:-1]),
                a.shape[-1],
            )
        else:
            a = a + pt_att
            a = a + square_mask.unsqueeze(-3)
            a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(a, v.transpose(-2, -3).to(dtype=a.dtype)).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, H, 3, N_res, P_v]
        if inplace_safe:
            v_pts = permute_final_dims(v_pts, (1, 3, 0, 2))
            o_pt = [torch.matmul(a, v.to(a.dtype)) for v in torch.unbind(v_pts, dim=-3)]
            o_pt = torch.stack(o_pt, dim=-3)
        else:
            o_pt = torch.sum(
                (
                    a[..., None, :, :, None]
                    * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
                ),
                dim=-2,
            )

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., 0][..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_norm = flatten_final_dims(
            torch.sqrt(torch.sum(o_pt**2, dim=-1) + self.eps), 2
        )

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        if _offload_inference:
            z[0] = z[0].to(o_pt.device)

        # [*, N_res, H, C_z]
        o_pair = torch.matmul(a.transpose(-2, -3), z[0].to(dtype=a.dtype))

        # [*, N_res, H * C_z]
        o_pair = flatten_final_dims(o_pair, 2)

        # [*, N_res, C_s]
        s = self.linear_out(
            torch.cat((o, *torch.unbind(o_pt, dim=-1), o_pt_norm, o_pair), dim=-1).to(
                dtype=z[0].dtype
            )
        )

        return s


class BackboneUpdate(nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self, c_s):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super(BackboneUpdate, self).__init__()

        self.c_s = c_s

        self.linear = Linear(self.c_s, 6, init="final")

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector
        """
        # [*, 6]
        update = self.linear(s)

        return update


class StructureModuleTransitionLayer(nn.Module):
    def __init__(self, c):
        super(StructureModuleTransitionLayer, self).__init__()

        self.c = c

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        s = s + s_initial

        return s


class StructureModuleTransition(nn.Module):
    def __init__(self, c, num_layers, dropout_rate):
        super(StructureModuleTransition, self).__init__()

        self.c = c
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            l = StructureModuleTransitionLayer(self.c)
            self.layers.append(l)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = LayerNorm(self.c)

    def forward(self, s):
        for l in self.layers:
            s = l(s)

        s = self.dropout(s)
        s = self.layer_norm(s)

        return s


class StructureModule(nn.Module):
    def __init__(
        self,
        c_s,
        c_z,
        c_ipa,
        c_resnet,
        no_heads_ipa,
        no_qk_points,
        no_v_points,
        dropout_rate,
        no_blocks,
        no_transition_layers,
        no_resnet_blocks,
        no_angles,
        trans_scale_factor,
        epsilon,
        inf,
        **kwargs,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_ipa:
                IPA hidden channel dimension
            c_resnet:
                Angle resnet (Alg. 23 lines 11-14) hidden channel dimension
            no_heads_ipa:
                Number of IPA heads
            no_qk_points:
                Number of query/key points to generate during IPA
            no_v_points:
                Number of value points to generate during IPA
            dropout_rate:
                Dropout rate used throughout the layer
            no_blocks:
                Number of structure module blocks
            no_transition_layers:
                Number of layers in the single representation transition
                (Alg. 23 lines 8-9)
            no_resnet_blocks:
                Number of blocks in the angle resnet
            no_angles:
                Number of angles to generate in the angle resnet
            trans_scale_factor:
                Scale of single representation transition hidden dimension
            epsilon:
                Small number used in angle resnet normalization
            inf:
                Large number used for attention masking
        """
        super(StructureModule, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_ipa = c_ipa
        self.c_resnet = c_resnet
        self.no_heads_ipa = no_heads_ipa
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.dropout_rate = dropout_rate
        self.no_blocks = no_blocks
        self.no_transition_layers = no_transition_layers
        self.no_resnet_blocks = no_resnet_blocks
        self.no_angles = no_angles
        self.trans_scale_factor = trans_scale_factor
        self.epsilon = epsilon
        self.inf = inf

        # Buffers to be lazily initialized later
        # self.default_frames
        # self.group_idx
        # self.atom_mask
        # self.lit_positions

        self.layer_norm_s = LayerNorm(self.c_s)
        self.layer_norm_z = LayerNorm(self.c_z)

        self.linear_in = Linear(self.c_s, self.c_s)

        self.ipa = InvariantPointAttention(
            self.c_s,
            self.c_z,
            self.c_ipa,
            self.no_heads_ipa,
            self.no_qk_points,
            self.no_v_points,
            inf=self.inf,
            eps=self.epsilon,
        )

        self.ipa_dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm_ipa = LayerNorm(self.c_s)

        self.transition = StructureModuleTransition(
            self.c_s,
            self.no_transition_layers,
            self.dropout_rate,
        )

        self.bb_update = BackboneUpdate(self.c_s)

        self.angle_resnet = AngleResnet(
            self.c_s,
            self.c_resnet,
            self.no_resnet_blocks,
            self.no_angles,
            self.epsilon,
        )

    def forward(
        self,
        evoformer_output_dict,
        aatype,
        mask=None,
        inplace_safe=False,
        _offload_inference=False,
    ):
        """
        Args:
            evoformer_output_dict:
                Dictionary containing:
                    "single":
                        [*, N_res, C_s] single representation
                    "pair":
                        [*, N_res, N_res, C_z] pair representation
            aatype:
                [*, N_res] amino acid indices
            mask:
                Optional [*, N_res] sequence mask
        Returns:
            A dictionary of outputs
        """
        s = evoformer_output_dict["single"]

        if mask is None:
            # [*, N]
            mask = s.new_ones(s.shape[:-1])

        # [*, N, C_s]
        s = self.layer_norm_s(s)

        # [*, N, N, C_z]
        z = self.layer_norm_z(evoformer_output_dict["pair"])

        z_reference_list = None
        if _offload_inference:
            assert sys.getrefcount(evoformer_output_dict["pair"]) == 2
            evoformer_output_dict["pair"] = evoformer_output_dict["pair"].cpu()
            z_reference_list = [z]
            z = None

        # [*, N, C_s]
        s_initial = s
        s = self.linear_in(s)

        # [*, N]
        rigids = Rigid.identity(
            s.shape[:-1],
            s.dtype,
            s.device,
            self.training,
            fmt="quat",
        )
        outputs = []
        for i in range(self.no_blocks):
            # [*, N, C_s]
            s = s + self.ipa(
                s,
                z,
                rigids,
                mask,
                inplace_safe=inplace_safe,
                _offload_inference=_offload_inference,
                _z_reference_list=z_reference_list,
            )
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)

            # [*, N]
            rigids = rigids.compose_q_update_vec(self.bb_update(s))

            # To hew as closely as possible to AlphaFold, we convert our
            # quaternion-based transformations to rotation-matrix ones
            # here
            backb_to_global = Rigid(
                Rotation(rot_mats=rigids.get_rots().get_rot_mats(), quats=None),
                rigids.get_trans(),
            )

            backb_to_global = backb_to_global.scale_translation(self.trans_scale_factor)

            # [*, N, 7, 2]
            unnormalized_angles, angles = self.angle_resnet(s, s_initial)

            all_frames_to_global = self.torsion_angles_to_frames(
                backb_to_global,
                angles,
                aatype,
            )

            pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
                all_frames_to_global,
                aatype,
            )

            scaled_rigids = rigids.scale_translation(self.trans_scale_factor)

            preds = {
                "frames": scaled_rigids.to_tensor_7(),
                "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
                "unnormalized_angles": unnormalized_angles,
                "angles": angles,
                "positions": pred_xyz,
                "states": s,
            }

            outputs.append(preds)

            rigids = rigids.stop_rot_gradient()

        del z, z_reference_list

        if _offload_inference:
            evoformer_output_dict["pair"] = evoformer_output_dict["pair"].to(s.device)

        outputs = dict_multimap(torch.stack, outputs)
        outputs["single"] = s

        return outputs

    def _init_residue_constants(self, float_dtype, device):
        if not hasattr(self, "default_frames"):
            self.register_buffer(
                "default_frames",
                torch.tensor(
                    restype_rigid_group_default_frame,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "group_idx"):
            self.register_buffer(
                "group_idx",
                torch.tensor(
                    restype_atom14_to_rigid_group,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "atom_mask"):
            self.register_buffer(
                "atom_mask",
                torch.tensor(
                    restype_atom14_mask,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                torch.tensor(
                    restype_atom14_rigid_group_positions,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )

    def torsion_angles_to_frames(self, r, alpha, f):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype, alpha.device)
        # Separated purely to make testing less annoying
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(
        self, r, f  # [*, N, 8]  # [*, N]
    ):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(r.get_rots().dtype, r.get_rots().device)
        return frames_and_literature_positions_to_atom14_pos(
            r,
            f,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )


class FgToBb(nn.Module):
    def __init__(
        self,
        c_s,
        c_hidden,
        no_points,
        mode="fg",  # fg,both,sum
    ):
        super(FgToBb, self).__init__()
        self.c_s = c_s
        self.c_hidden = c_hidden
        self.no_points = no_points
        self.mode = mode
        if mode == "sum":
            pass
        elif mode in ["fg", "both"]:
            hpq = self.no_points * self.c_hidden * 3
            self.linear_points = Linear(self.c_s, hpq, init="relu")
            self.linear_out = Linear(hpq, self.c_s, init="final")
            self.act_fn = nn.ReLU()
        else:
            raise Exception("wrong FgToBb mode type")

    def forward(self, s_bb, s_fg, r_bb, r_fg, fg_mask):
        if self.mode == "sum":
            return s_fg * fg_mask[..., None]
        p_fg = self.linear_points(s_fg)
        p_fg = torch.split(p_fg, p_fg.shape[-1] // 3, dim=-1)
        p_fg = torch.stack(p_fg, dim=-1)
        p_fg = r_fg[..., None].apply(p_fg)  # fg is on global
        p_fg = r_bb[..., None].invert_apply(p_fg)
        p_fg = p_fg * fg_mask[..., None, None]
        if self.mode == "both":
            p_bb = self.linear_points(s_fg)
            p_bb = torch.split(p_bb, p_bb.shape[-1] // 3, dim=-1)
            p_bb = torch.stack(p_bb, dim=-1)
            p_fg = p_bb + p_fg
        p_fg = flatten_final_dims(p_fg, 2)
        p_fg = self.linear_out(p_fg)
        return p_fg


class ScToBb(nn.Module):
    def __init__(
        self,
        c_s,
        c_hidden,
        no_points,
        mode="fg",  # fg,both,sum
    ):
        super(ScToBb, self).__init__()
        self.c_s = c_s
        self.c_hidden = c_hidden
        self.no_points = no_points
        self.mode = mode
        if mode == "sum":
            pass
        elif mode in ["fg", "both"]:
            hpq = self.no_points * self.c_hidden * 3
            self.linear_points = Linear(self.c_s, hpq, init="relu")
            self.linear_out = Linear(hpq, self.c_s, init="final")
            self.act_fn = nn.ReLU()
        else:
            raise Exception("wrong FgToBb mode type")

    def forward(self, s_frame, bb_fg_frames, fg_exist_mask, length):
        # s_frame [L_seed, L_sel, c_s]
        # bb_fg_frames [L_seed, L_sel]
        # fg_exist_mask [L_seed, L_sel]
        # length [L]

        # 1. s_frame scatter to L*5 frame
        orig_s_frame = torch.zeros(
            *s_frame.shape[:-2], 5 * length, s_frame.shape[-1]
        ).to(s_frame.device)

        fg_exist_mask_s = fg_exist_mask.unsqueeze(-1).expand(
            *fg_exist_mask.shape, s_frame.shape[-1]
        )
        orig_s_frame = orig_s_frame.scatter(
            -2, fg_exist_mask_s, s_frame
        )  # [L_seed, L*5, c_s]
        orig_s_frame = orig_s_frame.reshape(
            *orig_s_frame.shape[:-2], length, 5, s_frame.shape[-1]
        )  # [L_seed, L, 5, c_s]

        s_bb = orig_s_frame[..., :1, :]

        s_fg = orig_s_frame[..., 1:, :]

        # 2. bb_fg_frame scatter to L*5 frame
        bb_fg_frames_4x4 = Rigid.to_tensor_4x4(bb_fg_frames)  # [L_seed, L_sel, 4, 4]
        bb_fg_frames_4x4_zeros = torch.zeros(
            *bb_fg_frames_4x4.shape[:-3], 5 * length, 4, 4
        ).to(bb_fg_frames.device)

        fg_exist_mask_4x4 = fg_exist_mask[..., None, None]
        fg_exist_mask_4x4 = fg_exist_mask_4x4.expand(*fg_exist_mask.shape, 4, 4)

        bb_fg_frames_4x4 = bb_fg_frames_4x4_zeros.scatter(
            -3, fg_exist_mask_4x4, bb_fg_frames_4x4
        )  # 32, 500, 4, 4

        bb_fg_frames_4x4 = bb_fg_frames_4x4.reshape(
            *bb_fg_frames_4x4.shape[:-3], -1, 5, *bb_fg_frames_4x4.shape[-2:]
        )  # [L_seed, L, 5, 4, 4]
        bb_fg_frames = Rigid.from_tensor_4x4(bb_fg_frames_4x4)  # [L_seed, L, 5]
        r_bb = bb_fg_frames[..., :1]  # [L_seed, L, 1]
        r_fg = bb_fg_frames[..., 1:]  # [L_seed, L, 4]

        # orig_bb_fg_frames = orig_bb_fg_frames.s

        p_fg = self.linear_points(s_fg)  # [L_seed, L, 4, 768]
        p_fg = torch.split(p_fg, p_fg.shape[-1] // 3, dim=-1)
        p_fg = torch.stack(p_fg, dim=-1)  # [L_seed, L, 4, 256, 3]

        p_fg = r_fg[..., None].apply(p_fg)  # fg is on global
        # [32, 100, 4, 256, 3]
        p_fg = r_bb[..., None].invert_apply(p_fg)
        # [32, 100, 4, 256, 3]

        fg_mask_zeros = torch.zeros(*fg_exist_mask.shape[:-1], 5 * length).to(
            s_frame.device
        )
        fg_mask_ones = torch.ones(fg_exist_mask.shape).to(s_frame.device)
        fg_mask = fg_mask_zeros.scatter(-1, fg_exist_mask, fg_mask_ones)
        fg_mask = fg_mask.reshape(*fg_mask.shape[:-1], length, 5)
        fg_mask = fg_mask[..., 1:]  # [N_seed, L, 4]

        p_fg = p_fg * fg_mask[..., None, None]  # [32, 100, 4, 256, 3]
        p_fg = p_fg.sum(dim=-3)  # [32, 100, 256, 3]
        fg_mask = fg_mask.sum(dim=-1)  # [N_seed, L]
        fg_mask = fg_mask[..., None, None, None]
        p_fg = p_fg.unsqueeze(-3) / (fg_mask + 1e-8)  # [32, 100, 1, 256, 3]
        p_fg = p_fg.reshape(*p_fg.shape[:-3], -1, 3)
        p_bb = self.linear_points(s_bb)
        p_bb = torch.split(p_bb, p_bb.shape[-1] // 3, dim=-1)
        p_bb = torch.stack(p_bb, dim=-1)  # [32, 100, 1, 256, 3]
        p_bb = p_bb.reshape(*p_bb.shape[:-3], -1, 3)  # [32, 100, 256, 3]
        p_fg = p_bb + p_fg
        # [32, 100, 256, 3]
        p_fg = flatten_final_dims(p_fg, 2)  # [32, 100, 768]
        p_fg = self.linear_out(p_fg)  # [32, 100, c_s]
        return p_fg


class InvariantPointAttention_frames(nn.Module):
    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_qk_points: int,
        no_v_points: int,
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        super(InvariantPointAttention_frames, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        hpv = self.no_heads * self.no_v_points * 3

        self.linear_b = Linear(self.c_z, self.no_heads)

        self.head_weights = nn.Parameter(torch.zeros((no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.no_heads * (
            self.c_z + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()
        self.emb_rel_pos = Linear(74, self.c_z)
        self.linear_z_rel = Linear(self.c_z, self.c_z)

    def forward(
        self,
        s: torch.Tensor,
        z: Optional[torch.Tensor],
        r: Rigid,
        mask: torch.Tensor,
        rel_pos: torch.Tensor,
        inplace_safe: bool = False,
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        # input embedding for distinguish backbone and functional group already done
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3))
        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)
        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)
        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))
        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )
        ##
        # q, k, v = [32, 200, 12, 16]
        # q_pts, k_pts, #[32, 200, 12, 4, 3]
        # v_pts #[32, 200, 12, 8, 3]

        with torch.no_grad():
            seed_size = s.shape[0]
            n_frame = s.shape[1]
            pdist = r._trans[..., None, :] - r._trans[..., None, :, :]
            pdist = (pdist**2).sum(dim=-1)
            pdist = pdist ** (0.5)
            n_neigh = 50
            n_idx = torch.topk(pdist, n_neigh, largest=False, sorted=False)[1][
                ..., None
            ]
            p_n_idx = (
                torch.arange(n_frame)
                .cuda()[None, :, None]
                .expand(seed_size, -1, n_neigh)
            )  # pair is okay?
            p_n_idx = torch.cat([p_n_idx[..., None], n_idx], dim=-1)
            ##
            square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
            square_mask = square_mask[
                torch.arange(square_mask.shape[0])[:, None, None],
                p_n_idx[..., 0],
                p_n_idx[..., 1],
            ]
            square_mask = self.inf * (square_mask - 1)
            ##
            rel_pos = rel_pos[
                torch.arange(rel_pos.shape[0])[:, None, None],
                p_n_idx[..., 0],
                p_n_idx[..., 1],
            ]
            rel_pos = torch.cat(
                [
                    torch.nn.functional.one_hot(rel_pos[..., 0], num_classes=66),
                    torch.nn.functional.one_hot(rel_pos[..., 1], num_classes=10),
                    torch.nn.functional.one_hot(rel_pos[..., 2], num_classes=4),
                ],
                dim=-1,
            )

        k = k[torch.arange(k.shape[0])[:, None, None], n_idx[:, :, :, 0]]
        v = v[torch.arange(v.shape[0])[:, None, None], n_idx[:, :, :, 0]]
        k_pts = k_pts[torch.arange(k_pts.shape[0])[:, None, None], n_idx[:, :, :, 0]]
        v_pts = v_pts[torch.arange(v_pts.shape[0])[:, None, None], n_idx[:, :, :, 0]]
        z = z[
            torch.arange(z.shape[0])[:, None, None],
            p_n_idx[:, :, :, 0] % int(n_frame / 2),
            p_n_idx[:, :, :, 1] % int(n_frame / 2),
        ]
        # z = z+ self.emb_rel_pos(rel_pos)
        # z = self.linear_z_rel(z)
        # q_a torch.Size([32, 200, 12, 16])
        # k_a torch.Size([32, 200, 50, 12, 16])
        # v_a torch.Size([32, 200, 50, 12, 16])
        # q_pts torch.Size([32, 200, 12, 4, 3])
        # k_pts_a torch.Size([32, 200, 50, 12, 4, 3])
        # v_pts_a torch.Size([32, 200, 50, 12, 8, 3])
        # [*, N_res, N_res]

        ##
        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z)

        # [*, H, N_res, N_res]
        # a = torch.matmul(
        #    permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
        #    permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        # )
        ###
        # S, H, N_res, C
        # S, H, N_res, N_neigh, C
        # S, H, N_res, N_neigh

        a = (q[..., None, :, :] * k).sum(dim=-1)  # Seed_size, N_res, N_neigh, H
        # [32, 200, 50, 12]
        a = permute_final_dims(a, (2, 0, 1))  # S, H, N_res, N_neigh
        # [32, 12, 200, 50]
        a *= math.sqrt(1.0 / (3 * self.c_hidden))

        ##b: S, N_res, N_neigh, H
        a += math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1))

        # [*, N_res, N_res, H, P_q, 3]
        # S, N_res, N_neigh, H, P_q, 3
        # [*, N_res, N_neigh, H, P_q, 3]
        pt_att = q_pts.unsqueeze(-4) - k_pts
        pt_att = pt_att**2

        # [*, N_res, N_neigh, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        pt_att = pt_att * head_weights

        # [*, N_res, N_neigh, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)

        # [*, H, N_res, N_neigh]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        a = a + pt_att
        a = a + square_mask.unsqueeze(-3)  # S,N_frames, P, 3
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # a : S, H, N_res, N_neigh

        # S, N_res N_neigh H, C_hidden
        o = permute_final_dims(a, (1, 2, 0))[..., None] * v
        o = o.sum(dim=-3)  # S, N_res, H, C_hidden
        o = flatten_final_dims(o, 2)

        # S, N_res, H, P_v, 3 -> S, H, 3, N_res, P_v-> S, H, 3, 1, N_res, P_v
        # a -> S, H, 1, N_res, N_res, 1
        # a*vpts => S, H, 3, N_res, N_res, P_v  ->  S, H, 3, N_res, P_v -> S, N_res, H, P_v, 3

        # S, N_res, N_neigh, H, P_v, 3
        # S, H, 3, N_res, N_neigh, P_v

        # S, H, 1, N_res, N_neigh, 1

        # S, H, 3, N_res, N_neigh, P_v
        # S, H, 3, N_res, P_v

        o_pt = torch.sum(
            (a[..., None, :, :, None] * permute_final_dims(v_pts, (2, 4, 0, 1, 3))),
            dim=-2,
        )
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)
        o_pt_norm = flatten_final_dims(
            torch.sqrt(torch.sum(o_pt**2, dim=-1) + self.eps), 2
        )
        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        # S, H, N_res, N_neigh-> S, N_res, H, N_neigh
        # S, N_res, N_neigh, C_z

        # S, N_res, H, C_z
        o_pair = (
            a.transpose(-2, -3)[..., None] * z[:, :, None, :, :].to(dtype=a.dtype)
        ).sum(dim=-2)
        # [*, N_res, H * C_z]
        o_pair = flatten_final_dims(o_pair, 2)
        # [*, N_res, C_s]
        s = self.linear_out(
            torch.cat((o, *torch.unbind(o_pt, dim=-1), o_pt_norm, o_pair), dim=-1).to(
                dtype=z.dtype
            )
        )
        # [N_seed, 2*L, c_s]
        return s


class InvariantPointAttention_all_frames_neigh(nn.Module):
    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_qk_points: int,
        no_v_points: int,
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        super(InvariantPointAttention_all_frames_neigh, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        hpv = self.no_heads * self.no_v_points * 3

        self.linear_b = Linear(self.c_z, self.no_heads)

        self.head_weights = nn.Parameter(torch.zeros((no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.no_heads * (
            self.c_z + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()
        self.emb_rel_pos = Linear(74, self.c_z)
        self.linear_z_rel = Linear(self.c_z, self.c_z)

    def forward(
        self,
        s: torch.Tensor,
        z: Optional[torch.Tensor],
        r: Rigid,
        mask: torch.Tensor,
        rel_pos: torch.Tensor,
        inplace_safe: bool = False,
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        # input embedding for distinguish backbone and functional group already done
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3))
        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)
        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)
        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))
        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )
        ##
        # q, k, v = [32, 200, 12, 16]
        # q_pts, k_pts, #[32, 200, 12, 4, 3]
        # v_pts #[32, 200, 12, 8, 3]

        with torch.no_grad():
            seed_size = s.shape[0]
            n_frame = s.shape[1]
            pdist = r._trans[..., None, :] - r._trans[..., None, :, :]
            pdist = (pdist**2).sum(dim=-1)
            pdist = pdist ** (0.5)
            n_neigh = 50
            n_idx = torch.topk(pdist, n_neigh, largest=False, sorted=False)[1][
                ..., None
            ]
            p_n_idx = (
                torch.arange(n_frame)
                .cuda()[None, :, None]
                .expand(seed_size, -1, n_neigh)
            )  # pair is okay?
            p_n_idx = torch.cat([p_n_idx[..., None], n_idx], dim=-1)
            ##
            square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
            square_mask = square_mask[
                torch.arange(square_mask.shape[0])[:, None, None],
                p_n_idx[..., 0],
                p_n_idx[..., 1],
            ]
            square_mask = self.inf * (square_mask - 1)
            ##
            rel_pos = rel_pos[
                torch.arange(rel_pos.shape[0])[:, None, None],
                p_n_idx[..., 0],
                p_n_idx[..., 1],
            ]
            rel_pos = torch.cat(
                [
                    torch.nn.functional.one_hot(rel_pos[..., 0], num_classes=66),
                    torch.nn.functional.one_hot(rel_pos[..., 1], num_classes=10),
                    torch.nn.functional.one_hot(rel_pos[..., 2], num_classes=4),
                ],
                dim=-1,
            )

        k = k[torch.arange(k.shape[0])[:, None, None], n_idx[:, :, :, 0]]
        v = v[torch.arange(v.shape[0])[:, None, None], n_idx[:, :, :, 0]]
        k_pts = k_pts[torch.arange(k_pts.shape[0])[:, None, None], n_idx[:, :, :, 0]]
        v_pts = v_pts[torch.arange(v_pts.shape[0])[:, None, None], n_idx[:, :, :, 0]]
        z = z[
            torch.arange(z.shape[0])[:, None, None],
            p_n_idx[:, :, :, 0] % int(n_frame / 2),
            p_n_idx[:, :, :, 1] % int(n_frame / 2),
        ]
        # z = z+ self.emb_rel_pos(rel_pos)
        # z = self.linear_z_rel(z)
        # q_a torch.Size([32, 200, 12, 16])
        # k_a torch.Size([32, 200, 50, 12, 16])
        # v_a torch.Size([32, 200, 50, 12, 16])
        # q_pts torch.Size([32, 200, 12, 4, 3])
        # k_pts_a torch.Size([32, 200, 50, 12, 4, 3])
        # v_pts_a torch.Size([32, 200, 50, 12, 8, 3])
        # [*, N_res, N_res]

        ##
        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z)

        # [*, H, N_res, N_res]
        # a = torch.matmul(
        #    permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
        #    permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        # )
        ###
        # S, H, N_res, C
        # S, H, N_res, N_neigh, C
        # S, H, N_res, N_neigh

        a = (q[..., None, :, :] * k).sum(dim=-1)  # Seed_size, N_res, N_neigh, H
        # [32, 200, 50, 12]
        a = permute_final_dims(a, (2, 0, 1))  # S, H, N_res, N_neigh
        # [32, 12, 200, 50]
        a *= math.sqrt(1.0 / (3 * self.c_hidden))

        ##b: S, N_res, N_neigh, H
        a += math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1))

        # [*, N_res, N_res, H, P_q, 3]
        # S, N_res, N_neigh, H, P_q, 3
        # [*, N_res, N_neigh, H, P_q, 3]
        pt_att = q_pts.unsqueeze(-4) - k_pts
        pt_att = pt_att**2

        # [*, N_res, N_neigh, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        pt_att = pt_att * head_weights

        # [*, N_res, N_neigh, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)

        # [*, H, N_res, N_neigh]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        a = a + pt_att
        a = a + square_mask.unsqueeze(-3)  # S,N_frames, P, 3
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # a : S, H, N_res, N_neigh

        # S, N_res N_neigh H, C_hidden
        o = permute_final_dims(a, (1, 2, 0))[..., None] * v
        o = o.sum(dim=-3)  # S, N_res, H, C_hidden
        o = flatten_final_dims(o, 2)

        # S, N_res, H, P_v, 3 -> S, H, 3, N_res, P_v-> S, H, 3, 1, N_res, P_v
        # a -> S, H, 1, N_res, N_res, 1
        # a*vpts => S, H, 3, N_res, N_res, P_v  ->  S, H, 3, N_res, P_v -> S, N_res, H, P_v, 3

        # S, N_res, N_neigh, H, P_v, 3
        # S, H, 3, N_res, N_neigh, P_v

        # S, H, 1, N_res, N_neigh, 1

        # S, H, 3, N_res, N_neigh, P_v
        # S, H, 3, N_res, P_v

        o_pt = torch.sum(
            (a[..., None, :, :, None] * permute_final_dims(v_pts, (2, 4, 0, 1, 3))),
            dim=-2,
        )
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)
        o_pt_norm = flatten_final_dims(
            torch.sqrt(torch.sum(o_pt**2, dim=-1) + self.eps), 2
        )
        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        # S, H, N_res, N_neigh-> S, N_res, H, N_neigh
        # S, N_res, N_neigh, C_z

        # S, N_res, H, C_z
        o_pair = (
            a.transpose(-2, -3)[..., None] * z[:, :, None, :, :].to(dtype=a.dtype)
        ).sum(dim=-2)
        # [*, N_res, H * C_z]
        o_pair = flatten_final_dims(o_pair, 2)
        # [*, N_res, C_s]
        s = self.linear_out(
            torch.cat((o, *torch.unbind(o_pt, dim=-1), o_pt_norm, o_pair), dim=-1).to(
                dtype=z.dtype
            )
        )
        return s
