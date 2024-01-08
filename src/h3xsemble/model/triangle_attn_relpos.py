import sys
import torch
import torch.nn as nn
from typing import Optional, Tuple
from copy import deepcopy
from h3xsemble.model.primitives import Linear, ipa_point_weights_init_
from h3xsemble.utils import residue_constants as rc
from h3xsemble.utils.residue_constants import (
    restype_rigid_group_default_frame,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
)
from h3xsemble.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
from h3xsemble.model.utils import rbf

from h3xsemble.model.openfold_template import TemplatePairStack
from h3xsemble.model.pair_transition import PairTransition

class Sequential(nn.Sequential):
    """Sequential module with arbitrary forward args and kwargs. Used to pass graph, basis and edge features."""

    def forward(self, input, *args, **kwargs):
        for module in self:
            input = module(input, *args, **kwargs)
        return input


class WorkingZ(nn.Module):
    # From s and rigids generate new raw_z -> traingle_attn on raw_z
    def __init__(
        self,
        c_z: int,
        raw_2d_in: int,
        dgram_min_bin: int,
        dgram_max_bin: int,
        dgram_no_bins: int,
        c_in_tri_att: int,
        c_hidden_tri_att: int,
        c_hidden_tri_mul: int,
        no_head: int,
        no_blocks: int,
        pair_transition_n: int,
        chunk_size: int,
        rel_pos_dim: int,
        rel_pos_add: str,
        blocks_per_ckpt: int,
        extra_activation=True,
        bottle_neck=False,
        add_prev=False,
        use_tri_attn=True,
        partial_use_tri_attn=True,
        partial_use_tri_mul=True,
    ):
        super(WorkingZ, self).__init__()
        ##
        self.add_prev = add_prev
        in_dim = raw_2d_in
        ##
        self.featurization = From3Dto2D(
            dgram_min_bin,
            dgram_max_bin,
            dgram_no_bins,
        )
        self.use_tri_attn = use_tri_attn
        self.rel_pos_add = rel_pos_add
        ## relative positional encoding is added by concatenation
        if self.use_tri_attn:
            if rel_pos_add == "cat":
                transformer_modules = []
                in_att_dim = c_z
                if bottle_neck:
                    in_att_dim = int(c_z / 2)
                transformer_modules.append(Linear(in_dim + rel_pos_dim, in_att_dim))
                if extra_activation:
                    transformer_modules.append(nn.ReLU())
                transformer_modules.append(nn.LayerNorm(in_att_dim))
                transformer_modules.append(
                    Triattn_Z(
                        c_in_tri_att=in_att_dim,
                        c_hidden_tri_att=c_hidden_tri_att,
                        c_hidden_tri_mul=c_hidden_tri_mul,
                        no_head=no_head,
                        no_blocks=no_blocks,
                        pair_transition_n=pair_transition_n,
                        blocks_per_ckpt=blocks_per_ckpt,
                        use_tri_attn=partial_use_tri_attn,
                        use_tri_mul=partial_use_tri_mul,
                    )
                )
                if bottle_neck:
                    transformer_modules.append(Linear(in_att_dim, c_z))
                    if extra_activation:
                        transformer_modules.append(nn.ReLU())
                self.transformer_modules = Sequential(*transformer_modules)
            elif rel_pos_add == "sum":
                transformer_modules_1 = []
                transformer_modules_2 = []
                in_att_dim = c_z
                if bottle_neck:
                    in_att_dim = int(c_z / 2)
                transformer_modules_1.append(Linear(in_dim, in_att_dim))
                if extra_activation:
                    transformer_modules_1.append(nn.ReLU())
                ##
                self.linear_relpos = Linear(rel_pos_dim, in_att_dim)
                ##
                transformer_modules_2.append(nn.LayerNorm(in_att_dim))
                transformer_modules_2.append(
                    Triattn_Z(
                        c_in_tri_att=in_att_dim,
                        c_hidden_tri_att=c_hidden_tri_att,
                        c_hidden_tri_mul=c_hidden_tri_mul,
                        no_head=no_head,
                        no_blocks=no_blocks,
                        pair_transition_n=pair_transition_n,
                        blocks_per_ckpt=blocks_per_ckpt,
                        use_tri_attn=partial_use_tri_attn,
                        use_tri_mul=partial_use_tri_mul,
                    )
                )
                if bottle_neck:
                    transformer_modules_2.append(Linear(in_att_dim, c_z))
                    if extra_activation:
                        transformer_modules_2.append(nn.ReLU())
                self.transformer_modules_1 = Sequential(*transformer_modules_1)
                self.transformer_modules_2 = Sequential(*transformer_modules_2)
            else:
                print("wrong rel pos add type!!")
                sys.exit()

    def forward(self, s1, s2, z, rigids, ulr_mask, rel_pos, naive=False):
        raw_z = self.featurization(s1, s2, rigids, ulr_mask)
        # [S, L, L, 329]
        # 329 = 65 (64 distogram + 1 mask) + 128*2 + 4(ulr, non-ulr relationship)
        #       + mask(1) + unit-vector(3)
        if naive:
            return raw_z
        if self.rel_pos_add == "cat":
            raw_z = torch.cat([raw_z, rel_pos], dim=-1)
            raw_z = self.transformer_modules(raw_z)
        elif self.rel_pos_add == "sum":
            raw_z = self.transformer_modules_1(raw_z)
            raw_z = raw_z + self.linear_relpos(rel_pos)
            raw_z = self.transformer_modules_2(raw_z)
        if self.add_prev:
            if z == None:
                z = torch.zeros_like(raw_z)
            raw_z = raw_z + z
        return raw_z


class From3Dto2D(nn.Module):
    def __init__(self, min_bin, max_bin, no_bins,including_s=True):
        super(From3Dto2D, self).__init__()
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
    def forward(
        self, s1, s2, rigids, ulr_mask, mask=None, eps=1e-20, inf=1e8, use_rbf=False,including_s=True
    ):
        if mask == None:
            mask = (
                torch.zeros_like(ulr_mask).fill_(1).bool()
            )  # if not given set all true
        n_idx, ca_idx, c_idx, cb_idx = [
            rc.atom_order[a] for a in ["N", "CA", "C", "CB"]
        ]
        mask_2d = mask[..., None] * mask[..., None, :]  # pseudo-beta mask

        # Compute distogram (this seems to differ slightly from Alg. 5)
        # tpb = from_r_to_atoms(rigids)[...,cb_idx,:]
        tpb = rigids._trans
        if use_rbf:
            dgram = torch.cdist(tpb, tpb)
            d_sigma = (self.max_bin - self.min_bin) / (self.no_bins - 1)
            dgram = rbf(
                dgram, D_min=self.min_bin, D_count=self.no_bins, D_sigma=d_sigma
            )
        else:
            dgram = torch.sum(
                (tpb[..., None, :] - tpb[..., None, :, :]) ** 2, dim=-1, keepdim=True
            )  # [B, S, L, L, 1]
            lower = (
                torch.linspace(
                    self.min_bin, self.max_bin, self.no_bins, device=tpb.device
                )
                ** 2
            )  # [n_bin]
            upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
            dgram = ((dgram > lower) * (dgram < upper)).type(dgram.dtype)

        to_concat = [dgram, mask_2d[..., None]]  # 64+1
        if including_s:
            n_res = s1.shape[-2]
            # s1: S,L,C
            to_concat.append(s1[..., None, :, :].expand(*s1.shape[:-2], n_res, -1, -1))
            to_concat.append(s2[..., None, :].expand(*s2.shape[:-2], -1, n_res, -1))

        points = rigids.get_trans()[..., None, :, :]  # [S, 1, L, 3]
        rigid_vec = rigids[..., None].invert_apply(points)  # [S, L, L, 3]
        inv_distance_scalar = torch.rsqrt(eps + torch.sum(rigid_vec**2, dim=-1))
        # [S, L, L]
        inv_distance_scalar = inv_distance_scalar * mask_2d
        # [S, L, L, 1]
        unit_vector = rigid_vec * inv_distance_scalar[..., None]

        to_concat.extend(torch.unbind(unit_vector[..., None, :], dim=-1))
        to_concat.append(mask_2d[..., None])
        ## add ulr_mask_info
        # ulr_mask : B, N_RES
        tmp = ulr_mask[..., None] + 2 * ulr_mask[..., None, :]
        tmp = tmp.long()
        tmp = torch.nn.functional.one_hot(
            tmp, num_classes=4
        )  # ulr-ulr, ulr->non, non->ulr,non-non
        to_concat.append(tmp)
        ##
        act = torch.cat(to_concat, dim=-1)
        act = act * mask_2d[..., None]
        return act
class Hu_tmp_z(nn.Module):
    def __init__(
            self,
            c_s: int,
            c_z: int,
            mode = None,
            ):
        super(Hu_tmp_z, self).__init__()
        self.c_s=c_s
        self.c_z=c_z
        self.mode=mode
        ###
        dgram_min_bin=2.3125
        dgram_max_bin=21.6875
        dgram_no_bins=64
        rel_pos_dim = 32*2+1+1
        raw_2d_in= 73 
        pair_transition_n=1
        ###
        self.featurization = From3Dto2D(
            dgram_min_bin,
            dgram_max_bin,
            dgram_no_bins,
        )
        self.mode=mode
        ####
        in_dim = raw_2d_in
        self.linear_relpos=Linear(rel_pos_dim,c_z)
        if not mode =="naive":
            self.linear_s1=Linear(c_s,c_s)
            self.linear_s2=Linear(c_s,c_s)
            in_dim=raw_2d_in+2*c_s
        self.linear_z=Linear(in_dim,c_z)
        ####
        self.pair_transition = PairTransition(
            self.c_z,
            1
        )
    def forward(self, s, z, rigids, ulr_mask,rel_pos,mask):
        if not self.mode == "naive":
            s1=self.linear_s1(s)
            s2=self.linear_s2(s)
            out_z = self.linear_z(self.featurization(s1,s2,rigids,ulr_mask,mask)) + self.linear_relpos(rel_pos)
        else:
            out_z = self.linear_z(self.featurization(None,None,rigids,ulr_mask,including_s=False)) + self.linear_relpos(rel_pos)
        ####
        out_z=z+out_z
        out_z=self.pair_transition(out_z,mask=None)
        ####
        return out_z


class Triattn_Z(nn.Module):
    def __init__(
        self,
        c_in_tri_att=0,
        c_hidden_tri_att=16,
        c_hidden_tri_mul=16,
        no_blocks=2,
        no_head=4,
        pair_transition_n=2,
        drop_out_rate=0.1,
        blocks_per_ckpt=1,
        use_tri_attn=False,
        use_tri_mul=False,
    ):
        super(Triattn_Z, self).__init__()
        self.do_tri_attn = TemplatePairStack(
            c_in_tri_att,
            c_hidden_tri_att,
            c_hidden_tri_mul,
            no_blocks,
            no_head,
            pair_transition_n,
            drop_out_rate,
            blocks_per_ckpt,
            use_tri_attn=use_tri_attn,
            use_tri_mul=use_tri_mul,
        )

    def forward(self, z, mask=None):
        # z : *, N,N,C
        if mask == None:
            mask = z.new_ones(
                z.shape[:-2],
            )
        z = z.unsqueeze(-4)
        tmp_sq_mask = mask.unsqueeze(-1) + mask.unsqueeze(-2)
        tmp_sq_mask = (tmp_sq_mask == 2).long()
        tmp_sq_mask = tmp_sq_mask[:, :, None, ...]
        # sys.exit()
        z = self.do_tri_attn(t=z, mask=tmp_sq_mask, chunk_size=None)
        z = z.squeeze(-4)
        return z


