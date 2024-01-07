import torch, math
import torch.nn as nn
from typing import Optional, Tuple
from openfold.model.primitives import Linear, ipa_point_weights_init_, LayerNorm
from openfold.utils.rigid_utils import Rotation, Rigid
from openfold.utils.tensor_utils import (
    dict_multimap,
    permute_final_dims,
    flatten_final_dims,
)
from h3xsemble.model.openfold_template import TemplatePairStack

class Tmp_attention(nn.Module):
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
        super(Tmp_attention, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps

        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        concat_out_dim = self.no_heads * (self.c_hidden)
        self.linear_out = Linear(concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        r: Rigid,
        mask: torch.Tensor,
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

        ##########################
        # Compute attention scores
        ##########################

        # [*, H, N_res, N_res]
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a = a * math.sqrt(1.0 / (3 * self.c_hidden))
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(a, v.transpose(-2, -3)).transpose(-2, -3)
        # print (o.shape)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)
        s = self.linear_out(o)

        return s


class Tmp_attention2(nn.Module):
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
        super(Tmp_attention2, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps

        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)
        #
        self.linear_g_q = Linear(self.c_s, hc)
        self.linear_g_k = Linear(self.c_s, hc)
        #

        concat_out_dim = self.no_heads * (self.c_hidden)
        self.linear_out = Linear(concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        r: Rigid,
        mask: torch.Tensor,
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

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))
        k, v = torch.split(kv, self.c_hidden, dim=-1)
        #######################################
        g_q = self.linear_g_q(s.sum(dim=0))
        g_k = self.linear_g_k(s.sum(dim=0))
        g_q = g_q.view(g_q.shape[:-1] + (self.no_heads, -1))
        g_k = g_k.view(g_k.shape[:-1] + (self.no_heads, -1))
        ########################################
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a = a * math.sqrt(1.0 / (3 * self.c_hidden))
        a_g = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a_g = a_g * math.sqrt(1.0 / (3 * self.c_hidden))
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        a = a + square_mask.unsqueeze(-3) + a_g
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(a, v.transpose(-2, -3)).transpose(-2, -3)
        # print (o.shape)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)
        s = self.linear_out(o)

        return s


class Tmp_attention3(nn.Module):
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
        super(Tmp_attention3, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps

        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)
        #
        self.linear_g_q = Linear(self.c_s, hc)
        self.linear_g_k = Linear(self.c_s, hc)
        #
        hpv = self.no_heads * self.no_v_points * 3
        self.linear_v_points = Linear(self.c_s, hpv)
        #
        concat_out_dim = self.no_heads * (self.c_hidden + self.no_v_points * 4)
        self.linear_out = Linear(concat_out_dim, self.c_s, init="final")
        #
        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        r: Rigid,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))
        k, v = torch.split(kv, self.c_hidden, dim=-1)
        #######################################
        g_q = self.linear_g_q(s.sum(dim=0))
        g_k = self.linear_g_k(s.sum(dim=0))
        g_q = g_q.view(g_q.shape[:-1] + (self.no_heads, -1))
        g_k = g_k.view(g_k.shape[:-1] + (self.no_heads, -1))
        g_q = g_q[None, ...]
        g_k = g_k[None, ...]
        ########################################
        v_pts = self.linear_v_points(s)
        v_pts = torch.split(v_pts, v_pts.shape[-1] // 3, dim=-1)
        v_pts = torch.stack(v_pts, dim=-1)
        v_pts = r[..., None].apply(v_pts)
        v_pts = v_pts.view(v_pts.shape[:-2] + (self.no_heads, self.no_v_points, 3))
        ########################################
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a = a * math.sqrt(1.0 / (3 * self.c_hidden))
        a_g = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a_g = a_g * math.sqrt(1.0 / (3 * self.c_hidden))
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        a = a + square_mask.unsqueeze(-3) + a_g
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(a, v.transpose(-2, -3).to(dtype=a.dtype)).transpose(-2, -3)
        o = flatten_final_dims(o, 2)
        o_pt = torch.sum(
            (
                a[..., None, :, :, None]
                * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            ),
            dim=-2,
        )
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)
        o_pt_norm = flatten_final_dims(
            torch.sqrt(torch.sum(o_pt**2, dim=-1) + self.eps), 2
        )

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        # [*, N_res, H * C_hidden]

        s = self.linear_out(
            torch.cat((o, *torch.unbind(o_pt, dim=-1), o_pt_norm), dim=-1).to(
                dtype=s.dtype
            )
        )

        return s


class Tmp_attention4(nn.Module):
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
        super(Tmp_attention4, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps

        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)
        #
        self.linear_g_q = Linear(self.c_s, hc)
        self.linear_g_k = Linear(self.c_s, hc)
        #
        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)
        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)
        #
        self.head_weights = nn.Parameter(torch.zeros((no_heads)))
        ipa_point_weights_init_(self.head_weights)
        #
        concat_out_dim = self.no_heads * (self.c_hidden + self.no_v_points * 4)
        self.linear_out = Linear(concat_out_dim, self.c_s, init="final")
        #
        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        r: Rigid,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))
        k, v = torch.split(kv, self.c_hidden, dim=-1)
        #######################################
        g_q = self.linear_g_q(s.sum(dim=0))
        g_k = self.linear_g_k(s.sum(dim=0))
        g_q = g_q.view(g_q.shape[:-1] + (self.no_heads, -1))
        g_k = g_k.view(g_k.shape[:-1] + (self.no_heads, -1))
        g_q = g_q[None, ...]
        g_k = g_k[None, ...]
        ########################################
        q_pts = self.linear_q_points(s)
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)
        q_pts = q_pts.view(q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3))
        kv_pts = self.linear_kv_points(s)

        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)

        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )
        ########################################
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a = a * math.sqrt(1.0 / (3 * self.c_hidden))
        a_g = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a_g = a_g * math.sqrt(1.0 / (3 * self.c_hidden))
        pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_att**2
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        pt_att = pt_att * head_weights
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        pt_att = permute_final_dims(pt_att, (2, 0, 1))
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        a = a + square_mask.unsqueeze(-3) + a_g + pt_att
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(a, v.transpose(-2, -3).to(dtype=a.dtype)).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)
        o_pt = torch.sum(
            (
                a[..., None, :, :, None]
                * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            ),
            dim=-2,
        )
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)
        o_pt_norm = flatten_final_dims(
            torch.sqrt(torch.sum(o_pt**2, dim=-1) + self.eps), 2
        )

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        # [*, N_res, H * C_hidden]
        s = self.linear_out(
            torch.cat((o, *torch.unbind(o_pt, dim=-1), o_pt_norm), dim=-1).to(
                dtype=s.dtype
            )
        )
        return s


class CrossOver(nn.Module):
    """
    Implements Algorithm 22.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_qk_points=None,
        no_v_points=None,
        use_gloabal_feature=False,
        use_point_attention=False,
        use_triangle_attention=False,
        use_distance_bias = False,
        point_attention_weight=-0.5,
        use_non_ulr=True,
        tri_attn_config={},
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        super(CrossOver, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points

        self.use_gloabal_feature = use_gloabal_feature
        self.use_non_ulr = use_non_ulr
        self.use_point_attention = use_point_attention
        self.use_triangle_attention = use_triangle_attention
        self.use_distance_bias = use_distance_bias
        self.point_attention_weight = (
            point_attention_weight  # ....r^2, r^1,1, 1/(1+r), 1/(1+r^2) ...
        )

        self.inf = inf
        self.eps = eps

        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)
        concat_out_dim = self.c_hidden * self.no_heads
        #
        if self.use_gloabal_feature:
            self.layernorm_g = LayerNorm(self.c_s)
            self.linear_g_q = Linear(self.c_s, hc)
            self.linear_g_k = Linear(self.c_s, hc)
        #
        if self.use_point_attention:
            hpq = self.no_heads * self.no_qk_points * 3
            self.linear_q_points = Linear(self.c_s, hpq)
            hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
            self.linear_kv_points = Linear(self.c_s, hpkv)
            #
            self.head_weights = nn.Parameter(torch.zeros((no_heads)))
            ipa_point_weights_init_(self.head_weights)
            concat_out_dim = concat_out_dim + self.no_heads * (self.no_v_points * 4)
            #
        if self.use_triangle_attention:
            print('use triangle attention in cross_over!!!!!')
            self.triangle_attention = TemplatePairStack(**tri_attn_config)
            self.layernorm_z = LayerNorm(tri_attn_config["c_t"])
            self.linear_z = Linear(tri_attn_config["c_t"], self.no_heads)
        
        if self.use_distance_bias:
            print('use distance bias in cross_over!!!!!')
            self.linear_dist = Linear(1, self.no_heads)

        self.linear_out = Linear(concat_out_dim, self.c_s, init="final")
        #
        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        r: Rigid,
        mask: torch.Tensor,
        attn: list,
        use_non_ulr=True,
        ulr_mask=True,
    ) -> torch.Tensor:
        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))
        k, v = torch.split(kv, self.c_hidden, dim=-1)
        # print ("1!",q.shape,k.shape,v.shape)


        if self.use_gloabal_feature:
            # s:B,L,S,C_m
            s_g = s.mean(dim=-3)
            s_g = self.layernorm_g(s_g)
            g_q = self.linear_g_q(s_g)
            g_k = self.linear_g_k(s_g)
            g_q = g_q.view(g_q.shape[:-1] + (self.no_heads, -1))
            g_k = g_k.view(g_k.shape[:-1] + (self.no_heads, -1))
            g_q = g_q[:, None, ...]
            g_k = g_k[:, None, ...]
            # print ("2!",g_q.shape,g_k.shape)
        if self.use_point_attention:
            q_pts = self.linear_q_points(s)
            q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
            q_pts = torch.stack(q_pts, dim=-1)
            q_pts = r[..., None].apply(q_pts)
            q_pts = q_pts.view(q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3))
            kv_pts = self.linear_kv_points(s)

            kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
            kv_pts = torch.stack(kv_pts, dim=-1)
            kv_pts = r[..., None].apply(kv_pts)

            kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

            k_pts, v_pts = torch.split(
                kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
            )
            # print ("3!",q_pts.shape,k_pts.shape,v_pts.shape)
        
        if self.use_triangle_attention:
            vertical = s.unsqueeze(-2).expand(*s.shape[:-1], *s.shape[-2:])
            horizontal = s.unsqueeze(-3).expand(*s.shape[:-1], *s.shape[-2:])
            z = vertical + horizontal
            square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
            z = self.triangle_attention(z, mask=square_mask, chunk_size = None) #[B, L, S, S, c_s]
            z = self.layernorm_z(z)
            z = self.linear_z(z)
            z = z.view(*z.shape[:-3], z.shape[-1], z.shape[-2], -1)
        
        if self.use_distance_bias:
            point_coord = r.get_trans()
            distance = point_coord.unsqueeze(-2) - point_coord.unsqueeze(-3)
            distance = (distance)**2
            distance = torch.sum(distance, dim = -1)
            distance = torch.sqrt(distance)
            distance = distance.unsqueeze(-1)
            distance = self.linear_dist(distance)
            distance = distance.view(*distance.shape[:-3], distance.shape[-1], distance.shape[-2], -1)

        #######################################
        # Calculating attention matrix
        #######################################
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a = a * math.sqrt(1.0 / (3 * self.c_hidden))
        
        if self.use_gloabal_feature:
            a_g = torch.matmul(
                permute_final_dims(g_q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
                permute_final_dims(g_k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
            )
            a_g = a_g * math.sqrt(1.0 / (3 * self.c_hidden))
            # print ("5",a_g.shape)
            a = a + a_g
        
        if self.use_point_attention:
            pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
            pt_att = pt_att**2
            # print ("6",pt_att.shape)
            pt_att = sum(torch.unbind(pt_att, dim=-1))
            head_weights = self.softplus(self.head_weights).view(
                *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
            )
            head_weights = head_weights * math.sqrt(
                1.0 / (3 * (self.no_qk_points * 9.0 / 2))
            )
            pt_att = pt_att * head_weights
            pt_att = torch.sum(pt_att, dim=-1) * self.point_attention_weight
            # print ("7",pt_att.shape)
            pt_att = permute_final_dims(pt_att, (2, 0, 1))
            # print ("8",pt_att.shape)
            a = a + pt_att
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)
        # print ("8",square_mask.shape)
        a = a + square_mask.unsqueeze(-3)
        
        if self.use_triangle_attention:
            a += z

        if self.use_distance_bias:
            a += distance

        a = self.softmax(a)
        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        
        attn.append(a)
        o = torch.matmul(a, v.transpose(-2, -3).to(dtype=a.dtype)).transpose(-2, -3)
        # print ("9",o.shape)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)
        # print ("10",o.shape)
        if self.use_point_attention:
            o_pt = torch.sum(
                (
                    a[..., None, :, :, None]
                    * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
                ),
                dim=-2,
            )
            o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
            o_pt = r[..., None, None].invert_apply(o_pt)
            # print ("11", o_pt.shape)
            o_pt_norm = flatten_final_dims(
                torch.sqrt(torch.sum(o_pt**2, dim=-1) + self.eps), 2
            )
            # [*, N_res, H * P_v, 3]
            o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)
            s = self.linear_out(
                torch.cat((o, *torch.unbind(o_pt, dim=-1), o_pt_norm), dim=-1).to(
                    dtype=s.dtype
                )
            )
        else:
            s = self.linear_out(o).to(dtype=s.dtype)
        # print (s.shape)
        # print ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        return s, attn
