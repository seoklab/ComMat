import torch
import torch.nn as nn

from openfold.utils.rigid_utils import Rigid
from openfold.np.residue_constants import restype_atom14_to_rigid_group, chi_angles_mask
from openfold.model.primitives import Linear

from h3xsemble.model.structure_module import (
    InvariantPointAttention,
    InvariantPointAttention_frames,
    InvariantPointAttention_all_frames_neigh,
    InvariantPointAttention_all_frames,
    StructureModuleTransition,
    FgToBb,
    ScToBb,
)

from openfold.utils.tensor_utils import dict_multimap
from h3xsemble.model.triangle_attn_relpos import Sequential,Hu_tmp_z
from h3xsemble.model.utils import prep_rel_pos, get_bb_pos
from openfold.utils.tensor_utils import (
    dict_multimap,
    permute_final_dims,
    flatten_final_dims,
)
from h3xsemble.model.structure_building import (
    Build_str,
    Build_str_all_frame,
)


def prep_sidechain_frames(
    batch,
    sidechain_frames,
    s,
    trans_scale_factor,
    z,
    ulr_mask,
    dim_change=False,
    idx_sel=True,
):
    # 0. get sidechain frames
    sidechain_frames = torch.cat(
        [sidechain_frames[..., :1, :, :], sidechain_frames[..., 4:, :, :]], dim=-3
    )  # [N_seed, L, 5, 4, 4]
    sidechain_frames_5 = sidechain_frames
    sidechain_frames = sidechain_frames.reshape(
        *sidechain_frames.shape[:-4], -1, 4, 4
    )  # [N_seed, L*5, 4, 4]

    chi_angles_mask = [
        [0.0, 0.0, 0.0, 0.0],  # ALA
        [1.0, 1.0, 1.0, 1.0],  # ARG
        [1.0, 1.0, 0.0, 0.0],  # ASN
        [1.0, 1.0, 0.0, 0.0],  # ASP
        [1.0, 0.0, 0.0, 0.0],  # CYS
        [1.0, 1.0, 1.0, 0.0],  # GLN
        [1.0, 1.0, 1.0, 0.0],  # GLU
        [0.0, 0.0, 0.0, 0.0],  # GLY
        [1.0, 1.0, 0.0, 0.0],  # HIS
        [1.0, 1.0, 0.0, 0.0],  # ILE
        [1.0, 1.0, 0.0, 0.0],  # LEU
        [1.0, 1.0, 1.0, 1.0],  # LYS
        [1.0, 1.0, 1.0, 0.0],  # MET
        [1.0, 1.0, 0.0, 0.0],  # PHE
        [1.0, 1.0, 0.0, 0.0],  # PRO
        [1.0, 0.0, 0.0, 0.0],  # SER
        [1.0, 0.0, 0.0, 0.0],  # THR
        [1.0, 1.0, 0.0, 0.0],  # TRP
        [1.0, 1.0, 0.0, 0.0],  # TYR
        [1.0, 0.0, 0.0, 0.0],  # VAL
    ]

    chi_angles_mask = torch.tensor(chi_angles_mask).to(device=batch["aatype"].device)
    chi_angles_mask = chi_angles_mask[batch["aatype"]]  # [N_seed, L, 4]
    backbone_mask = chi_angles_mask[..., 0].clone()  # [N_seed, L]
    backbone_mask = backbone_mask.fill_(1)
    fg_exist_mask = torch.cat(
        [backbone_mask.unsqueeze(-1), chi_angles_mask], dim=-1
    )  # [N_seed, L, 5]
    fg_exist_mask_5 = fg_exist_mask
    fg_exist_mask = fg_exist_mask.reshape(*fg_exist_mask.shape[:-2], -1)
    fg_exist_mask_first = fg_exist_mask.bool()[..., 0, :]  # [*, L*5]
    fg_exist_mask_index = fg_exist_mask_first.nonzero(as_tuple=True)[0]  # [*, L_sel]
    fg_exist_mask = fg_exist_mask_index[None].expand(
        *batch["aatype"].shape[:-1], *fg_exist_mask_index.shape
    )  # [*, N_seed, L_sel]

    # 2. Get relative position
    n_res = batch["hu_residue_index"]  # [N_seed, L]
    res_idx = n_res.unsqueeze(-1).expand(*n_res.shape, 5)  # [N_seed, L, 5]
    res_idx = res_idx.reshape(*res_idx.shape[:-2], -1)  # [N_seed, L*5]

    chain_idx = batch["chain_id"]
    chain_idx = chain_idx.unsqueeze(-1).expand(*chain_idx.shape, 5)  # [N_seed, L, 5]
    chain_idx = chain_idx.reshape(*chain_idx.shape[:-2], -1)  # [N_seed, L*5]

    max_rel_k = 32
    rel_pos = prep_rel_pos(res_idx, chain_idx, max_rel_k)  # [N_seed, L*5, L*5]

    # 3. Get bb_or_fg
    bb_or_fg_bb = (
        torch.zeros_like(n_res).unsqueeze(-1).to(device=batch["aatype"].device)
    )  # [N_seed, L, 1]
    bb_or_fg_sc = torch.ones_like(n_res).unsqueeze(-1)
    bb_or_fg_sc = bb_or_fg_sc.expand(*n_res.shape, 4)  # [N_seed, L, 4]
    bb_or_fg = torch.cat([bb_or_fg_bb, bb_or_fg_sc], dim=-1)  # [N_seed, L, 5]
    bb_or_fg = bb_or_fg.reshape(*bb_or_fg.shape[:-2], -1)  # [N_seed, L*5]
    bb_or_fg = bb_or_fg[..., None] + 2 * bb_or_fg[..., None, :]  # [N_seed, L*5, L*5]

    # 4. Get torsion_distance
    torsion_distance = torch.arange(5).to(device=batch["aatype"].device)
    torsion_distance = torsion_distance.repeat(*n_res.shape, 1)  # [N_seed, L, 5]
    torsion_distance = torsion_distance.reshape(
        *torsion_distance.shape[:-2], -1
    )  # [N_seed, L*5]
    torsion_distance = torsion_distance[..., None] - torsion_distance[..., None, :]
    # 5. Get final relative positions
    rel_pos = torch.stack(
        [rel_pos, bb_or_fg, torsion_distance], dim=-1
    )  # [N_seed, L*5, L*5, 3]

    # 6. Get single feature
    # s = [N_seed, L, 128*5]
    if dim_change:
        s = s.reshape(*s.shape[:-1], 5, -1)  # [N_seed, L, 5, 128]
        s = s.reshape(*s.shape[:-3], -1, s.shape[-1])  # [N_seed, L*5, 128]
    # 7. Get pairwise feature
    if dim_change:
        z = z.unsqueeze(-3).expand(*z.shape[:-2], 5, *z.shape[-2:])
        # [N_seed, L, 5, L, 96]
        z = z.reshape(*z.shape[:-4], -1, *z.shape[-2:])
        # [N_seed, L*5, L, 96]
        z = z.unsqueeze(-2)
        z = z.expand(*z.shape[:-2], 5, z.shape[-1])
        # [N_seed, L*5, L, 5, 96]
        z = z.reshape(*z.shape[:-3], -1, z.shape[-1])
        # [N_seed, L*5, L*5, 96]

    # 8. Get_ulr_mask
    ulr_mask = ulr_mask.unsqueeze(-1).expand(*ulr_mask.shape, 5)  # [N_seed, L, 5]
    ulr_mask = ulr_mask.reshape(*ulr_mask.shape[:-2], -1)  # [N_seed, L*5]
    if not idx_sel:
        sidechain_frames_5_rigid = Rigid.from_tensor_4x4(sidechain_frames_5)
        return s, z, rel_pos, sidechain_frames_5_rigid, fg_exist_mask_5, ulr_mask
    else:
        # Get selected features
        # 1. frames
        fg_exist_mask_frame = fg_exist_mask[..., None, None]  # [N_seed, L_sel, 1, 1]
        fg_exist_mask_frame = fg_exist_mask_frame.expand(
            *fg_exist_mask_frame.shape[:-2], 4, 4
        )  # [N_seed, L_sel, 4, 4]
        sidechain_frames = torch.gather(
            sidechain_frames, dim=-3, index=fg_exist_mask_frame
        )  # [N_seed, L_sel, 4, 4]
        inp_frames = Rigid.from_tensor_4x4(sidechain_frames)  # [N_seed, L_sel]
        inp_frames = inp_frames.scale_translation(float(1.0 / trans_scale_factor))
        # 2. single feature
        # s [N_seed, L*5, 128]
        if dim_change:
            fg_exist_mask_s = fg_exist_mask[..., None]  # [N_seed, L_sel, 1]
            fg_exist_mask_s = fg_exist_mask_s.expand(*fg_exist_mask_s.shape[:-1], 128)
            s_out = torch.gather(
                s, dim=-2, index=fg_exist_mask_s
            )  # [N_seed, L_sel, 128]
        else:
            s_out = s
        # 3. relative position
        # rel_pos [N_seed, L*5, L*5, 3]
        fg_exist_mask_row = fg_exist_mask[..., None, None]  # [N_seed, L_sel, 1, 1]
        fg_exist_mask_row = fg_exist_mask_row.expand(
            *fg_exist_mask_row.shape[:-2], *rel_pos.shape[-2:]
        )  # [N_seed, L_sel, L*5, 3]

        rel_pos_row = torch.gather(rel_pos, dim=-3, index=fg_exist_mask_row)
        # [N_seed, L_sel, L*5, 3]
        fg_exist_mask_column = fg_exist_mask[
            ..., None, None
        ]  # [*, N_seed, L_sel, 1, 1]
        fg_exist_mask_column = permute_final_dims(
            fg_exist_mask_column, (1, 0, 2)
        )  # [*, N_seed, 1, L_sel, 1]
        fg_exist_mask_column = fg_exist_mask_column.expand(
            *fg_exist_mask_column.shape[:-3], fg_exist_mask_column.shape[-2], -1, 3
        )
        # [N_seed, L_sel, L_sel, 3]
        rel_pos_final = torch.gather(
            rel_pos_row, dim=-2, index=fg_exist_mask_column
        )  # [N_seed, L_sel, L_sel, 3]
        # 4. pairwise feature
        # z [N_seed, L*5, L*5, 96]
        if dim_change:
            fg_exist_mask_row = fg_exist_mask[..., None, None]  # [N_seed, L_sel, 1, 1]
            fg_exist_mask_row = fg_exist_mask_row.expand(
                *fg_exist_mask_row.shape[:-2], *z.shape[-2:]
            )  # [N_seed, L_sel, L*5, 3]

            z_row = torch.gather(z, dim=-3, index=fg_exist_mask_row)
            # [N_seed, L_sel, L*5, 3]
            fg_exist_mask_column = fg_exist_mask[
                ..., None, None
            ]  # [N_seed, L_sel, 1, 1]
            fg_exist_mask_column = permute_final_dims(
                fg_exist_mask_column, (1, 0, 2)
            )  # [N_seed, 1, L_sel, 1]
            fg_exist_mask_column = fg_exist_mask_column.expand(
                *fg_exist_mask_column.shape[:-3], fg_exist_mask_column.shape[-2], -1, 96
            )
            # [N_seed, L_sel, L_sel, 96]
            z_final = torch.gather(
                z, dim=-2, index=fg_exist_mask_column
            )  # [N_seed, L_sel, L_sel, 96]
        else:
            z_final = z
        # 5. Get ulr_mask
        ulr_mask = torch.gather(
            ulr_mask, dim=-1, index=fg_exist_mask
        )  # [N_seed, L_sel]

        return s_out, z_final, rel_pos_final, inp_frames, fg_exist_mask, ulr_mask
        # s_out [*, N_seed, L_sel, 128]
        # rel_pos [*, N_seed, L_sel, L_sel, 3]
        # inp_frames [*, N_seed, L_sel]
        # fg_exist_mask [*, N_seed, L_sel]


class IPA_single(nn.Module):
    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_qk_points: int,
        no_v_points: int,
        no_transition_layers: int,
        bb_update_fn,
        build_str_fn,
        build_str_all_fn=None,
        inf: float = 1e5,
        eps: float = 1e-8,
        ######
        dropout_rate: float = 1e-1,
        update_rigids: bool = True,
        build_str_stat: bool = False,
        ###
        ipa_type="frame",
        merge_fg_n_point=16,
        fg_frame_trans_scale_factor=1.0,
        use_non_ulr=False,
        build_str_type="torsion",
        stop_rot_gradient=False,
        use_update_z_ipa=None,
    ):
        super(IPA_single, self).__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.no_transition_layers = no_transition_layers
        self.inf = inf
        self.epsilon = eps
        self.dropout_rate = dropout_rate
        self.build_str_stat = False
        self.build_str = build_str_fn
        self.ipa_type = ipa_type
        self.trans_scale_factor = fg_frame_trans_scale_factor
        self.use_non_ulr = use_non_ulr
        self.build_str_type = build_str_type
        self.stop_rot_gradient = stop_rot_gradient
        if (use_update_z_ipa == None) or (use_update_z_ipa==False):
            self.use_update_z_ipa = False
        else:
            self.use_update_z_ipa  = use_update_z_ipa[0]
            self.update_z_ipa_mode = use_update_z_ipa[1]
            self.return_update_z   = use_update_z_ipa[2]
        #
        if "default" in ipa_type.split("_"):
            self.ipa = InvariantPointAttention(
                self.c_s,
                self.c_z,
                self.c_hidden,
                self.no_heads,
                self.no_qk_points,
                self.no_v_points,
                inf=self.inf,
                eps=self.epsilon,
            )
        if "frame" in ipa_type.split("_") and not "all_frame" in ipa_type:
            self.ipa_frames = InvariantPointAttention_frames(
                self.c_s,
                self.c_z,
                self.c_hidden,
                self.no_heads,
                self.no_qk_points,
                self.no_v_points,
                inf=self.inf,
                eps=self.epsilon,
            )
            # self.frame_dropout=nn.Dropout(self.dropout_rate)
            self.merge_fg_bb = FgToBb(
                self.c_s,
                self.c_hidden,
                16,
                mode="both",
            )
            self.ipa_all_frames = InvariantPointAttention_all_frames_neigh(
                self.c_s,
                self.c_z,
                self.c_hidden,
                self.no_heads,
                self.no_qk_points,
                self.no_v_points,
                inf=self.inf,
                eps=self.epsilon,
            )
            self.ipa = InvariantPointAttention(
                self.c_s,
                self.c_z,
                self.c_hidden,
                self.no_heads,
                self.no_qk_points,
                self.no_v_points,
                inf=self.inf,
                eps=self.epsilon,
            )
        if ipa_type == "all_frame":
            self.ipa = InvariantPointAttention(
                self.c_s,
                self.c_z,
                self.c_hidden,
                self.no_heads,
                self.no_qk_points,
                self.no_v_points,
                inf=self.inf,
                eps=self.epsilon,
            )
            self.merge_all_frame = ScToBb(self.c_s, self.c_hidden, 16, mode="both")
            if build_str_type == "frame":
                self.build_str_all = build_str_all_fn
        if ipa_type == "implicit_all_frame":
            self.ipa_all_frames = InvariantPointAttention_all_frames(
                self.c_s,
                self.c_z,
                self.c_hidden,
                self.no_heads,
                self.no_qk_points,
                self.no_v_points,
                inf=self.inf,
                eps=self.epsilon,
            )

        # common
        self.ipa_dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm_ipa = nn.LayerNorm(self.c_s)
        self.layer_norm_adapt = nn.LayerNorm(self.c_s)
        self.transition = StructureModuleTransition(
            self.c_s,
            self.no_transition_layers,
            self.dropout_rate,
        )
        if self.ipa_type == "all_frame":
            self.layer_expand_sidechain = nn.Linear(self.c_s, self.c_s * 5)
        ##
        
        if self.use_update_z_ipa:
            self.update_z_ipa_fn = Hu_tmp_z(
                    self.c_s,
                    self.c_z,
                    mode= self.update_z_ipa_mode
                    )
        ##
        if not bb_update_fn == None:
            self.bb_update = bb_update_fn
        else:
            self.bb_update = Linear(c_s, 6, init="final")


        self.update_rigids = update_rigids

    def forward(self, inp):
        s = inp[0]
        s_initial = inp[1]
        z = inp[2]
        rigids = inp[3]
        mask = inp[4]
        ulr_mask = inp[5]
        rel_pos = inp[6]
        batch = inp[7]
        out_str_s = inp[8]
        ##
        if (
            self.update_rigids
        ):  # for IPA_enc module no structure building for current stage
            if self.ipa_type != "all_frame":
                str_out = self.build_str(s, rigids, s_initial, batch["aatype"])
                out_str_s.append(str_out)
            else:
                if self.build_str_type == "torsion":
                    str_out = self.build_str(s, rigids, s_initial, batch["aatype"])
                else:  # build by frame
                    str_out = self.build_str_all(
                        s, rigids, s_initial, batch["aatype"], all_frame=True
                    )
                out_str_s.append(str_out)
        ###
        out_z=z # is it okay?
        ###
        if self.use_update_z_ipa:
            z=self.update_z_ipa_fn(s,z,rigids,ulr_mask,rel_pos,mask)
        if self.ipa_type == "default":
            s = s + self.ipa(s, z, rigids, mask)
        elif self.ipa_type == "implicit_all_frame":
            (_, _, _, bb_fg_frames, fg_exist_mask, _) = prep_sidechain_frames(
                batch,
                str_out["sidechain_frames"],
                s,
                self.trans_scale_factor,
                z,
                ulr_mask,
                dim_change=False,
                idx_sel=False,
            )
            s = s + self.ipa_all_frames(s, z, bb_fg_frames, fg_exist_mask)
        elif self.ipa_type == "all_frame":
            s_length = s.shape[-2]
            if s_length > batch["aatype"].shape[-1]:  # if already have all frames
                s_frames = s
                (
                    _,
                    _,
                    rel_pos,
                    bb_fg_frames,
                    fg_exist_mask,
                    _,
                ) = prep_sidechain_frames(
                    batch,
                    str_out["sidehcain_frames"],
                    s,
                    self.trans_scale_factor,
                    z,
                    ulr_mask,
                    dim_change=False,
                    idx_sel=True,
                )
            else:
                s_expand = self.layer_expand_sidechain(s)  # need layernorm?
                (
                    s_frames,
                    z_final,
                    rel_pos,
                    bb_fg_frames,
                    fg_exist_mask,
                    _,
                ) = prep_sidechain_frames(
                    batch,
                    str_out["sidechain_frames"],
                    s_expand,
                    self.trans_scale_factor,
                    z,
                    ulr_mask,
                    dim_change=True,
                    idx_sel=True,
                )
            # s_out [N_seed, L_sel, 128]
            # rel_pos [N_seed, L_sel, L_sel, 3]
            # bb_fg_frames [N_seed, L_sel]
            # fg_exist_mask [N_seed, L_sel]
            # z  [N_seed, L, L, 96]

            s_frames_ipa = self.ipa(
                s_frames,
                z_final,
                bb_fg_frames,
                fg_exist_mask,  # rel_pos
            )

            if self.build_str_type == "torsion":
                s_frames_ipa = self.merge_all_frame(  # structure_module/ScToBb
                    s_frames_ipa,
                    bb_fg_frames,
                    fg_exist_mask,
                    length=batch["aatype"].shape[-1],
                )
                s = s + s_frames_ipa

            elif self.build_str_type == "frame":
                s = s_frames + s_frames_ipa
                rigids = bb_fg_frames

        s = self.ipa_dropout(s)
        s = self.layer_norm_ipa(s)
        s = self.transition(s)
        if self.ipa_type != "all_frame":
            if self.update_rigids:
                update_tr = self.bb_update(s)
                if not self.use_non_ulr:
                    update_tr = torch.masked_fill(
                        update_tr, ~ulr_mask.bool()[..., None], 0
                    )  #
                rigids = rigids.compose_q_update_vec(update_tr)
                if self.stop_rot_gradient:
                    rigids = rigids.stop_rot_gradient()
        #
        else:  # if all_frame is used, update side chain frames
            if self.update_rigids:
                update_tr = self.bb_update(s)
                if not self.use_non_ulr:
                    update_tr = torch.masked_fill(
                        update_tr, ~ulr_mask.bool()[..., None], 0
                    )
                rigids = rigids.compose_q_update_vec(update_tr)
                if self.stop_rot_gradient:
                    rigids = rigids.stop_rot_gradient()
        if self.use_update_z_ipa:
            if self.return_update_z:
                out_z=z
        return (s, s_initial, out_z, rigids, mask, ulr_mask,rel_pos, batch, out_str_s)


class IPA_block(nn.Module):
    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_qk_points: int,
        no_v_points: int,
        no_transition_layers: int,
        bb_update_fn,
        inf: float = 1e5,
        eps: float = 1e-8,
        ######
        dropout_rate: float = 1e-1,
        no_blocks: int = 8,
        ###
        use_update_z: bool = True,
        build_str_interval=1,
        update_z_fn=None,
        use_cross_over: bool = True,
        cross_over_interval=0,
        cross_over_fn=None,
        build_str_fn=None,
        build_str_fn_all=None,
        update_rigids=True,
        ######
        no_ipa_s: int = 1,
        ipa_type_s: list = None,
        fg_frame_trans_scale_factor=1.0,
        use_non_ulr=False,
        build_str_type="torsion",
        stop_rot_gradient=False,
        use_update_z_ipa=False,
    ):
        super(IPA_block, self).__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.epsilon = eps
        self.dropout_rate = dropout_rate
        self.no_blocks = no_blocks
        self.cross_over_interval = cross_over_interval
        self.no_transition_layers = no_transition_layers
        self.use_update_z = use_update_z
        self.build_str_interval = build_str_interval
        self.build_str_type = build_str_type
        self.build_str = build_str_fn
        self.stop_rot_gradient = stop_rot_gradient
        #
        self.ipa_module_s = []
        #
        different_build_str_stat=False
        if build_str_fn == None:
            different_build_str_stat=True
        #
        for idx in range(no_ipa_s):
            stop_rot_gradient_state = False
            ##
            if self.stop_rot_gradient and (not idx == no_ipa_s - 1):
                stop_rot_gradient_state = True
            ##
            if different_build_str_stat:
                print ("New!!!")
                print ("New!!!")
                print ("New!!!")
                print ("New!!!")
                print ("New!!!")
                print ("New!!!")
                print ("New!!!")
                print ("New!!!")
                print ("New!!!")
                print ("New!!!")
                print ("New!!!")
                print ("New!!!")
                print ("New!!!")
                print ("New!!!")
                print ("New!!!")
                print ("New!!!")
                print ("New!!!")
                print ("New!!!")
                hard_coding_config={
                    "trans_scale_factor": 10,
                    "angle_resnet": {
                        "c_in": 128,
                        "c_hidden": 128,
                        "no_blocks": 2,
                        "no_angles": 7,
                        "epsilon": 1e-8,
                    }
                }
                build_str_fn=Build_str(hard_coding_config)
                #build_str_fn_all=Build_str(config["Build_str_all"])
            ##

            if build_str_type == "frame" and ipa_type_s[idx] == "all_frame":
                self.ipa_module_s.append(
                    IPA_single(
                        self.c_s,
                        self.c_z,
                        self.c_hidden,
                        self.no_heads,
                        self.no_qk_points,
                        self.no_v_points,
                        self.no_transition_layers,
                        bb_update_fn=None,
                        inf=self.inf,
                        eps=self.epsilon,
                        update_rigids=update_rigids,
                        build_str_fn=build_str_fn,
                        build_str_all_fn=build_str_fn_all,
                        ipa_type=ipa_type_s[idx],
                        fg_frame_trans_scale_factor=fg_frame_trans_scale_factor,
                        use_non_ulr=use_non_ulr,
                        build_str_type=build_str_type,
                        stop_rot_gradient=stop_rot_gradient_state,
                        use_update_z_ipa=use_update_z_ipa,
                    )
                )
            else:
                self.ipa_module_s.append(
                    IPA_single(
                        self.c_s,
                        self.c_z,
                        self.c_hidden,
                        self.no_heads,
                        self.no_qk_points,
                        self.no_v_points,
                        self.no_transition_layers,
                        bb_update_fn=None,
                        inf=self.inf,
                        eps=self.epsilon,
                        update_rigids=update_rigids,
                        build_str_fn=build_str_fn,
                        ipa_type=ipa_type_s[idx],
                        fg_frame_trans_scale_factor=fg_frame_trans_scale_factor,
                        use_non_ulr=use_non_ulr,
                        build_str_type=build_str_type,
                        stop_rot_gradient=stop_rot_gradient_state,
                        use_update_z_ipa=use_update_z_ipa,
                    )
                )
        if different_build_str_stat:
            hard_coding_config={
                "trans_scale_factor": 10,
                "angle_resnet": {
                    "c_in": 128,
                    "c_hidden": 128,
                    "no_blocks": 2,
                    "no_angles": 7,
                    "epsilon": 1e-8,
                }
            }

            self.build_str=Build_str(hard_coding_config)
        self.ipa_module_s = Sequential(*self.ipa_module_s)
        ##
        self.layer_norm_s = nn.LayerNorm(self.c_s)
        self.layer_norm_z = nn.LayerNorm(self.c_z)
        ## Cross over
        self.use_cross_over = use_cross_over
        if self.use_cross_over:
            self.cross_over = cross_over_fn
        if self.use_update_z:
            self.update_z_module = update_z_fn

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        rigids: Rigid,
        mask: torch.Tensor,
        ulr_mask: torch.Tensor,
        aatype: torch.Tensor,
        rel_pos: torch.Tensor,
        batch: dict,
        return_output: bool,
        is_enc=False,
    ) -> torch.Tensor:
        if mask == None:
            mask = torch.zeros_like(ulr_mask).fill_(1).float()
        ##
        output_s = []
        out_str_s = []
        cross_over_prev = []
        cross_over_after = []
        ##
        s_initial = s
        s = self.layer_norm_s(s)
        attn = []
        for i in range(self.no_blocks):
            if self.use_cross_over:
                # cross-over by attention
                cross_over_prev.append(get_bb_pos(rigids, 10))
                s, z, rigids, mask, ulr_mask, attn = self.cross_over((s, z, rigids, mask, ulr_mask, attn))
                cross_over_after.append(get_bb_pos(rigids, 10))
            if self.use_update_z:
                z = self.update_z_module(s, s, z, rigids, ulr_mask, rel_pos)
            (
                s,
                s_initial,
                z,
                rigids,
                mask,
                ulr_mask,
                rel_pos,
                batch,
                out_str_s,
            ) = self.ipa_module_s(
                (s, s_initial, z, rigids, mask, ulr_mask,rel_pos, batch, out_str_s)
            )
            if (i == (self.no_blocks - 1)):
                if self.build_str_type == "torsion":
                    out_str_s.append(self.build_str(s, rigids, s_initial, aatype))
                else:
                    out_str_s.append(self.build_str_all(s, rigids, s_initial, aatype))
            # if i < (self.no_blocks - 1) and self.stop_grad_rot:
            #    rigids = rigids.stop_rot_gradient()
        if is_enc:
            return s
        out_dic = {}
        out_dic['cross_over_prev'] = torch.stack(cross_over_prev, dim=0) #[n_block, B, S, L, 3]
        out_dic['cross_over_after'] = torch.stack(cross_over_after, dim=0)
        out_dic['attention_value'] = torch.stack(attn, dim=0)
        #print(out_str_s[0]['positions'].shape)
        out_str_s = out_str_s[1:]  # because 0-index is initial perturbed structure
        out_dic["sm"] = dict_multimap(torch.stack, out_str_s)
        
        if return_output:
            out_dic = {}
            out_dic["sm"] = dict_multimap(torch.stack, out_str_s)
            # out_dic["sm"]["frames"] -> [3, 1, 32, 100, 7]
            out_dic["sm"]["single"] = s
            out_dic["pair"] = z
            out_dic["cross_over_prev"] = torch.stack(cross_over_prev, dim=0)
            out_dic["cross_over_after"] = torch.stack(cross_over_after, dim=0)
            out_dic["attention_value"] = torch.stack(attn, dim=0)
        
        return s, z, rigids, out_dic

class IPA_block_w_no_recycle(nn.Module):
    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_qk_points: int,
        no_v_points: int,
        no_transition_layers: int,
        bb_update_fn,
        inf: float = 1e5,
        eps: float = 1e-8,
        ######
        dropout_rate: float = 1e-1,
        no_blocks: int = 8,
        ###
        use_update_z: bool = True,
        build_str_interval=1,
        update_z_fn=None,
        use_cross_over: bool = True,
        cross_over_interval=0,
        cross_over_fn=None,
        build_str_fn=None,
        build_str_fn_all=None,
        update_rigids=True,
        ######
        no_ipa_s: int = 1,
        ipa_type_s: list = None,
        fg_frame_trans_scale_factor=1.0,
        use_non_ulr=False,
        build_str_type="torsion",
        stop_rot_gradient=False,
        use_update_z_ipa=False,
    ):
        super(IPA_block_w_no_recycle, self).__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.epsilon = eps
        self.dropout_rate = dropout_rate
        self.no_blocks = no_blocks
        self.cross_over_interval = cross_over_interval
        self.no_transition_layers = no_transition_layers
        self.use_update_z = use_update_z
        self.build_str_interval = build_str_interval
        self.build_str_type = build_str_type
        self.build_str = build_str_fn
        self.stop_rot_gradient = stop_rot_gradient
        #

        self.ipa_module_s = []
        #
        for idx in range(4):
            stop_rot_gradient_state = False
            ##
            if self.stop_rot_gradient and (not idx == 4 - 1):
                stop_rot_gradient_state = True
            if build_str_type == "frame" and ipa_type_s[idx] == "all_frame":
                self.ipa_module_s.append(
                    IPA_single(
                        self.c_s,
                        self.c_z,
                        self.c_hidden,
                        self.no_heads,
                        self.no_qk_points,
                        self.no_v_points,
                        self.no_transition_layers,
                        bb_update_fn=None,
                        inf=self.inf,
                        eps=self.epsilon,
                        update_rigids=update_rigids,
                        build_str_fn=build_str_fn,
                        build_str_all_fn=build_str_fn_all,
                        ipa_type=ipa_type_s[idx],
                        fg_frame_trans_scale_factor=fg_frame_trans_scale_factor,
                        use_non_ulr=use_non_ulr,
                        build_str_type=build_str_type,
                        stop_rot_gradient=stop_rot_gradient_state,
                        use_update_z_ipa=use_update_z_ipa,
                    )
                )
            else:
                self.ipa_module_s.append(
                    IPA_single(
                        self.c_s,
                        self.c_z,
                        self.c_hidden,
                        self.no_heads,
                        self.no_qk_points,
                        self.no_v_points,
                        self.no_transition_layers,
                        bb_update_fn=None,
                        inf=self.inf,
                        eps=self.epsilon,
                        update_rigids=update_rigids,
                        build_str_fn=build_str_fn,
                        ipa_type=ipa_type_s[idx],
                        fg_frame_trans_scale_factor=fg_frame_trans_scale_factor,
                        use_non_ulr=use_non_ulr,
                        build_str_type=build_str_type,
                        stop_rot_gradient=stop_rot_gradient_state,
                        use_update_z_ipa=use_update_z_ipa,
                    )
                )
        self.ipa_module_s_1 = Sequential(*self.ipa_module_s)
        self.ipa_module_s_2 = Sequential(*self.ipa_module_s)
        self.ipa_module_s_3 = Sequential(*self.ipa_module_s)

        ##
        self.layer_norm_s = nn.LayerNorm(self.c_s)
        self.layer_norm_z = nn.LayerNorm(self.c_z)
        ## Cross over
        self.use_cross_over = use_cross_over
        if self.use_cross_over:
            self.cross_over = cross_over_fn
        if self.use_update_z:
            self.update_z_module = update_z_fn

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        rigids: Rigid,
        mask: torch.Tensor,
        ulr_mask: torch.Tensor,
        aatype: torch.Tensor,
        rel_pos: torch.Tensor,
        batch: dict,
        return_output: bool,
        is_enc=False,
    ) -> torch.Tensor:
        if mask == None:
            mask = torch.zeros_like(ulr_mask).fill_(1).float()
        ##
        output_s = []
        out_str_s = []
        ##
        s_initial = s
        s = self.layer_norm_s(s)
        # cross-over by attention
        s, z, rigids = self.cross_over(s, z, rigids, mask, ulr_mask)
        z = self.update_z_module(s, s, z, rigids, ulr_mask, rel_pos)
        (
            s,
            s_initial,
            z,
            rigids,
            mask,
            ulr_mask,
            rel_pos,
            batch,
            out_str_s,
        ) = self.ipa_module_s_1((s, s_initial, z, rigids, mask, ulr_mask,rel_pos, batch, out_str_s))
        s, z, rigids = self.cross_over(s, z, rigids, mask, ulr_mask)
        (
            s,
            s_initial,
            z,
            rigids,
            mask,
            ulr_mask,
            rel_pos,
            batch,
            out_str_s,
        ) = self.ipa_module_s_2((s, s_initial, z, rigids, mask, ulr_mask,rel_pos, batch, out_str_s))
        s, z, rigids = self.cross_over(s, z, rigids, mask, ulr_mask)
        (
            s,
            s_initial,
            z,
            rigids,
            mask,
            ulr_mask,
            rel_pos,
            batch,
            out_str_s,
        ) = self.ipa_module_s_3((s, s_initial, z, rigids, mask, ulr_mask,rel_pos, batch, out_str_s))
        if return_output:
            if self.build_str_type == "torsion":
                out_str_s.append(self.build_str(s, rigids, s_initial, aatype))
            else:
                out_str_s.append(self.build_str_all(s, rigids, s_initial, aatype))
        # if i < (self.no_blocks - 1) and self.stop_grad_rot:
        #    rigids = rigids.stop_rot_gradient()
        if is_enc:
            return s
        out_dic = None
        out_str_s = out_str_s[1:]  # because 0-index is initial perturbed structure

        if return_output:
            out_dic = {}
            out_dic["sm"] = dict_multimap(torch.stack, out_str_s)
            out_dic["sm"]["single"] = s
            out_dic["pair"] = z

        return s, z, rigids, out_dic
