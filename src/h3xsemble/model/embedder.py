import sys
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from copy import deepcopy
from openfold.model.primitives import Linear, ipa_point_weights_init_
from openfold.np.residue_constants import (
    restype_rigid_group_default_frame,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
)
import openfold.utils.feats
from openfold.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
import openfold.utils.feats
from openfold.utils.rigid_utils import Rotation, Rigid
from openfold.utils.tensor_utils import (
    dict_multimap,
    permute_final_dims,
    flatten_final_dims,
)

from h3xsemble.model.utils import prep_rel_pos


class InputSeqFeatEmbedder(nn.Module):
    def __init__(
        self,
        tf_dim: int,
        out_dim: int,
        use_torsion_mode,  # False,str
        lang_model_stat: bool,
        lang_model_dim=1,
        use_ulr=True,
        mode = "cat",
        is_score_model=False,
    ):
        super(InputSeqFeatEmbedder, self).__init__()
        self.lang_model_stat = lang_model_stat
        self.use_torsion_mode = use_torsion_mode
        self.use_ulr=use_ulr
        self.is_score_model = is_score_model
        self.mode=mode
        #
        in_dim = tf_dim
        torsion_dim=0
        if use_torsion_mode == "bb":
            torsion_dim = 3 * 2
        elif use_torsion_mode == "all":
            torsion_dim = 7 * 2
        elif use_torsion_mode == "bb_bin":
            torsion_dim = 3 * 10
        if mode =="cat":
            mid_dim = int(out_dim / 2)
            if lang_model_stat:
                in_dim = in_dim + lang_model_dim  # 1300
            if use_ulr:
                in_dim = in_dim + torsion_dim + 1  # ulr_mask
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, mid_dim), nn.ReLU(), nn.Linear(mid_dim, out_dim)
            )
            self.norm = nn.LayerNorm(out_dim)
        elif mode =="sum":
            mid_dim = int (out_dim *2)
            self.mlp_aatype = Linear(20,out_dim)
            if lang_model_stat:
                self.mlp_langout=Linear(lang_model_dim,out_dim)
            if use_torsion_mode:
                self.mlp_torsion=Linear(torsion_dim,out_dim)
            if use_ulr:
                self.mlp_ulr=Linear(2,out_dim)


    def forward(
        self, aatype, lang_out=None, ulr_mask=None, torsion_angle=None, input_s=None
    ):
        aatype = torch.nn.functional.one_hot(aatype, num_classes=20).float()
        if self.use_torsion_mode == "bb":
            torsion_angle = torsion_angle[:, :, 0:3, :].view(
                torsion_angle.shape[0], torsion_angle.shape[1], -1
            )
        elif self.use_torsion_mode == "all":
            torsion_angle = torsion_angle[:, :, :, :].view(
                torsion_angle.shape[0], torsion_angle.shape[1], -1
            )
        ####################################################################
        if self.mode == "cat":
            if self.lang_model_stat:
                aatype = torch.cat([aatype, lang_out], dim=-1)
            if not self.use_torsion_mode:
                aatype = torch.cat([aatype, ulr_mask[..., None]], dim=-1)
            else:
                aatype = torch.cat([aatype, torsion_angle, ulr_mask[..., None]], dim=-1)
            aatype = self.mlp(aatype)
            aatype = self.norm(aatype)
        elif self.mode=="sum":
            aatype = self.mlp_aatype(aatype)
            if self.use_torsion_mode:
                aatype = aatype + self.mlp_torsion(torsion_angle)
            if self.lang_model_stat:
                aatype = aatype + self.mlp_langout(lang_out)
            if self.use_ulr:
                aatype = aatype + self.mlp_ulr( torch.nn.functional.one_hot(ulr_mask.long(),num_classes=2).float())
        return aatype


##
class InputPairFeatEmbedder(nn.Module):  # backbone torsion?
    def __init__(
        self,
        tf_dim: int,
        c_z: int,
        out_dim: int,
        max_rel_pos: int,
    ):
        super(InputPairFeatEmbedder, self).__init__()
        self.max_rel_pos = max_rel_pos
        mid_dim = int(out_dim / 2)
        #
        self.linear_tf_z_i = Linear(tf_dim, c_z)
        self.linear_tf_z_j = Linear(tf_dim, c_z)
        #
        inp_dim = (max_rel_pos * 2 + 1 + 1) + c_z
        self.mlp = nn.Sequential(
            nn.Linear(inp_dim, mid_dim), nn.ReLU(), nn.Linear(mid_dim, out_dim)
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        s,
        z,
        rigids,
        mask,
        ulr_mask,
        dic,
    ):
        aatype = torch.nn.functional.one_hot(dic["aatype"], num_classes=20)
        n_res = aatype.shape[0]
        templ = torch.zeros_like(n_res, n_res, 1).to(device=aatype.device)
        res_idx = dic["hu_residue_index"]
        chain_idx = dic["chain_idx"]
        pair_dist = dic["pair_d"]
        ###
        relpos = prep_rel_pos(res_idx, chain_idx, self.max_rel_pos)

        emb_z_i = self.linear_tf_z_i(aatype)
        emb_z_j = self.linear_tf_z_j(aatype)

        pair_repr = emb_z_i[..., None] + emb_z_j[..., None, :]
        pair_repr = torch.cat([pair_repr, relpos], dim=-1)

        pair_repr = self.mlp(pair_repr)
        pair_repr = self.norm(pair_repr)
        return pair_repr


###
class RecycleEmbedder(nn.Module):
    def __init__(
        self,
        c_s: int,
        c_z: int,
        min_bin: float,
        max_bin: float,
        no_bins: int,
        inf: float = 1e8,
        eps: float = 1e-4,
        use_dist_bin=False,
        recycle_add_mode="sum",
        **kwargs,
    ):
        super(RecycleEmbedder, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.inf = inf
        self.use_dist_bin = False
        self.recycle_add_mode = recycle_add_mode
        print(self.recycle_add_mode)
        if self.recycle_add_mode == "sum":
            self.layer_norm_s = nn.LayerNorm(self.c_s)
            self.layer_norm_z = nn.LayerNorm(self.c_z)
        elif self.recycle_add_mode in ["cat", "cat_sum"]:
            self.act_fn = nn.ReLU()
            self.linear_s = nn.Linear(self.c_s * 2, self.c_s, init="relu")
            self.linear_z = nn.Linear(self.c_z * 2, self.c_z, init="relu")
            self.layer_norm_s = nn.LayerNorm(self.c_s)
            self.layer_norm_z = nn.LayerNorm(self.c_z)
        self.linear = Linear(self.no_bins, self.c_z)

    def forward(
        self,
        s_init: torch.Tensor,
        z_init: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # [*, N, C_m]
        if self.recycle_add_mode in ["sum"]:
            s_update = self.layer_norm_s(s)
            z_update = self.layer_norm_z(z)
            return s_init + s_update, z_init + z_update
        elif self.recycle_add_mode in ["cat", "cat_sum"]:
            s_update = self.linear_s(torch.cat([s_init, s], dim=-1))
            z_update = self.linear_z(torch.cat([z_init, s], dim=-1))
            s_update = self.act_fn(s_update)
            z_update = self.act_fn(z_update)
            s_update = self.layer_norm_s(s_update)
            z_update = self.layer_norm_z(z_update)
            if self.recycle_add_mode == "cat":
                return s_update, z_update
            elif self.recycle_add_mode == "cat_sum":
                return s_update + s_init, z_update + z_init
        else:
            print("Wrong recycle add mode !!")
            sys.exit()


class TemplateAngleEmbedder(nn.Module):
    """
    Embeds the "template_angle_feat" feature.

    Implements Algorithm 2, line 7.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        **kwargs,
    ):
        """
        Args:
            c_in:
                Final dimension of "template_angle_feat"
            c_out:
                Output channel dimension
        """
        super(TemplateAngleEmbedder, self).__init__()

        self.c_out = c_out
        self.c_in = c_in

        self.linear_1 = Linear(self.c_in, self.c_out, init="relu")
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.c_out, self.c_out, init="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [*, N_templ, N_res, c_in] "template_angle_feat" features
        Returns:
            x: [*, N_templ, N_res, C_out] embedding
        """
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)

        return x
