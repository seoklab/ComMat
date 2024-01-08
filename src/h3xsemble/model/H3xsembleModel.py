import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from copy import deepcopy
from h3xsemble.model.primitives import Linear, ipa_point_weights_init_


import h3xsemble.utils.feats
from h3xsemble.utils.rigid_utils import Rotation, Rigid
from h3xsemble.utils.tensor_utils import (
    dict_multimap,
    permute_final_dims,
    flatten_final_dims,
)

from h3xsemble.utils.feats import atom14_to_atom37
from h3xsemble.model.triangle_attn_relpos import Sequential
from h3xsemble.model.embedder import InputSeqFeatEmbedder, RecycleEmbedder
from h3xsemble.model.triangle_attn_relpos import WorkingZ
from h3xsemble.model.heads_new import AuxiliaryHeads

from h3xsemble.model.cross_over import (
    CrossOver,
)
from h3xsemble.model.structure_module import (
    StructureModuleTransition,
)

from h3xsemble.model.structure_building import (
    Build_str,
    Build_str_all_frame,
)

from h3xsemble.model.IPAModule import IPA_block, IPA_block_w_no_recycle
from h3xsemble.model.utils import prep_rel_pos, InitialPertTrs, clone_rigid, get_bb_pos

class CrossOverModule(nn.Module):
    """
    Implements Algorithm 22.
    """

    def __init__(
        self,
        c_s,
        c_z,
        c_hidden,
        no_heads,
        no_qk_points,
        no_v_points,
        no_transition_layers,
        dropout_rate,
        use_gloabal_feature=False,
        use_point_attention=False,
        use_triangle_attention=False,
        use_distance_bias=False,
        point_attention_weight=None,
        use_non_ulr=True,
        update_rigids=False,
        tri_attn_config = {},
        # config, # config.model.crossover / sub : co_mode, transition
    ):
        super(CrossOverModule, self).__init__()
        ##
        self.update_rigids = update_rigids
        if self.update_rigids:
            self.bb_update = Linear(c_s, 6, init="final")
        self.co = CrossOver(
            c_s,
            c_z,
            c_hidden,
            no_heads,
            no_qk_points=no_qk_points,
            no_v_points=no_v_points,
            use_gloabal_feature=use_gloabal_feature,
            use_point_attention=use_point_attention,
            use_triangle_attention=use_triangle_attention,
            use_distance_bias=use_distance_bias,
            point_attention_weight=point_attention_weight,
            use_non_ulr=use_non_ulr,
            tri_attn_config=tri_attn_config
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(c_s)
        self.transition = StructureModuleTransition(
            c_s, no_transition_layers, dropout_rate
        )

    def forward(
        self, inp
    ) -> torch.Tensor:
        
        s = inp[0]
        z = inp[1]
        r = inp[2]
        mask = inp[3]
        ulr_mask = inp[4]
        attn = inp[5]
        ##
        # ulr_mask=torch.zeros_like(ulr_mask).fill_(True)
        # s_full=s
        # s=s[:,ulr_mask[0]]
        # mask=mask[:,ulr_mask[0]]
        # inp_trans=permute_final_dims(r._trans[:,ulr_mask[0]],(1,0,2))
        # inp_quats=permute_final_dims(r._rots.get_quats()[:,ulr_mask[0]],(1,0,2))
        # inp_rigid=Rigid(Rotation(quats=inp_quats),inp_trans)
        inp_trans = permute_final_dims(r._trans, (1, 0, 2))
        inp_quats = permute_final_dims(r._rots.get_quats(), (1, 0, 2))
        inp_rigid = Rigid(Rotation(quats=inp_quats), inp_trans)
        s = permute_final_dims(s, (1, 0, 2))  # B,S,L,C -> B,L,S,C
        mask_ori = mask
        mask = permute_final_dims(mask, (1, 0))
        s_new, attn = self.co(s, z, inp_rigid, mask, attn)
        s = s+s_new
        s = permute_final_dims(s, (1, 0, 2))  # B,L,S,C -> B,S,L,C
        mask = permute_final_dims(mask, (1, 0))
        s = self.dropout(s)
        s = self.layer_norm(s)
        s = self.transition(s)
        if self.update_rigids:
            update_tr = self.bb_update(s)
            #update_tr = torch.masked_fill(update_tr, ~ulr_mask.bool()[..., None], 0)  #
            r = r.compose_q_update_vec(update_tr)
        return (s, z, r, mask_ori, ulr_mask, attn)


# MUST : check again mask is proper.
class H3xsembleModule(nn.Module):
    def __init__(
        self,
        config,
        use_torsion=True,
        use_lang_model=False,
        is_train=True,
        build_str_interval=1,
        # stop_rot_gradient=True,
    ):
        super().__init__()
        self.use_no_recycle = config.use_no_recycle
        self.is_train = is_train
        self.use_torsion = use_torsion
        self.use_lang_model = use_lang_model
        self.build_str_interval = build_str_interval
        # self.stop_rot_gradient = stop_rot_gradient
        self.no_recycle = config.no_recycle
        self.run_mode = None
        ##
        self.emb_z_initial = Linear(config.raw_2d_in, config.c_z, init="relu")
        self.linear_rel_pos = Linear(32 * 2 + 1 + 1, config.c_z, init="relu")
        self.norm_z_initial = nn.LayerNorm(config.c_z)
        self.act_fn = nn.ReLU()
        ###
        self.seq_embedder = InputSeqFeatEmbedder(
            **config["seq_feat_embedder"]
        )  # add backbone torsion? How to handle ulr region? zero?
        self.recycle_embedder = RecycleEmbedder(**config["recycle_embedder"])
        ###
        self.initial_bank_generator = InitialPertTrs(**config.InitialPertTrs)
        ###
        if (
            "all_frame" in config["IPA_block"]["ipa_type_s"]
            and config["IPA_block"]["build_str_type"] == "frame"
        ):
            self.build_str_all_fn = Build_str_all_frame(config["Build_str_all"])
        self.build_str_fn = Build_str(config["Build_str"])
        if "uniform_build_str" in config["Build_str"].keys():
            if not config["Build_str"]["uniform_build_str"]:
                self.build_str_fn=None

        ###
        self.bb_update = Linear(config["bb_update"]["c_s"], 6, init="final")
        if not self.run_mode == "withouth_IPA_enc":
            self.IPA_enc = IPA_block(
                **config["IPA_enc"],
                bb_update_fn=self.bb_update,
                build_str_fn=self.build_str_fn,
            )

        if config.use_cross_over:
            if "cross_over_no_block" not in config or config.cross_over_no_block == 1:
                self.cross_over_fn = CrossOverModule(**config["Cross_over_module"])
            else:
                self.cross_over_fn = []
                for i in range(config.cross_over_no_block):
                    self.cross_over_fn.append(CrossOverModule(**config["Cross_over_module"]))
                self.cross_over_fn = Sequential(*self.cross_over_fn)
        else:
            self.cross_over_fn = None
        # if config.use_update_z:
        self.workingz = WorkingZ(**config["WorkingZ"])

        if (
            "all_frame" in config["IPA_block"]["ipa_type_s"]
            and config["IPA_block"]["build_str_type"] == "frame"
        ):
            self.IPA_block = IPA_block(
                **config["IPA_block"],
                update_z_fn=self.workingz,
                bb_update_fn=self.bb_update,
                build_str_fn=self.build_str_fn,
                cross_over_fn=self.cross_over_fn,
                build_str_all_fn=self.build_str_all_fn,
            )
        # else:
        #    self.workingz=None
        self.IPA_block = IPA_block(
            **config["IPA_block"],
            update_z_fn=self.workingz,
            bb_update_fn=self.bb_update,
            build_str_fn=self.build_str_fn,
            cross_over_fn=self.cross_over_fn,
        )

        if self.use_no_recycle:
            self.IPA_block_w_no_recycle = IPA_block_w_no_recycle(
            **config["IPA_block"],
            update_z_fn = self.workingz,
            bb_update_fn=self.bb_update,
            build_str_fn=self.build_str_fn,
            cross_over_fn=self.cross_over_fn,
            )
        ###
        self.aux_heads = AuxiliaryHeads(config["heads"])

    def forward(
        self,
        input_dic,
        sampled_n_recycle=None,
        tmp_stat=False,
        reset_structure=False,
        force_freeze_generator=False,
    ):
        if sampled_n_recycle == None:
            sampled_n_recycle = self.no_recycle
        s = input_dic["aatype"].clone().detach()  # [S, L]
        aatype = input_dic["aatype"]  # [S, L]
        z = None
        r = input_dic["inp_gt"]  # (S, L(R, T))
        # [L] : h3xsemble.utils.rigid_utils.Rigid
        # (_rot (h3xsemble.utils.Rigid, _rot_mats[3, 3]), _trans [3])
        ulr_mask = input_dic["ulr_mask"]  # where to predict [S, L]
        str_mask = input_dic["miss_mask"]  # [S, L]
        train_mode = input_dic["train_mode"]  #
        inference_mode = input_dic["inference_mode"]

        with torch.no_grad():
            torsion_angle = input_dic["torsion_angles_sin_cos"]  # [S, L, 14, 3]
            tmp = torsion_angle.clone()
            tmp.masked_fill_(ulr_mask[..., None, None], 0.0)
            torsion_angle = tmp
        ##
        if not "lang_out" in input_dic.keys():
            lang_out = None
        else:
            lang_out = input_dic["lang_out"]  # [S, L, 1280] #esm embedding?
        ##
        if not self.use_no_recycle:
            output_s = self.run_with_recycle_ver2(
            s,
            z,
            r,
            ulr_mask,
            str_mask,
            torsion_angle,
            aatype,
            sampled_n_recycle,
            input_dic,
            lang_out=lang_out,
            train_mode=train_mode,
            inference_mode=inference_mode, 
            tmp_stat=tmp_stat,
            reset_structure=reset_structure,
            force_freeze_generator=force_freeze_generator,
            )
        else:
            output_s = self.run_wo_recycle(
            s,
            z,
            r,
            ulr_mask,
            str_mask,
            torsion_angle,
            aatype,
            sampled_n_recycle,
            input_dic,
            lang_out=lang_out,
            train_mode=train_mode,
            inference_mode=inference_mode, 
            tmp_stat=tmp_stat,
            reset_structure=reset_structure,
            force_freeze_generator=force_freeze_generator,
            )
        output_s["final_atom_positions"] = atom14_to_atom37(
            output_s["sm"]["positions"][-1], input_dic
        )
        n_recycle = output_s["positions_all_recycle"].shape[0]
        positions_all_recycle = []
        for i in range(n_recycle):
            positions_all_recycle.append(atom14_to_atom37(
                output_s["positions_all_recycle"][i][-1], input_dic))
        output_s["positions_all_recycle"] = torch.stack(positions_all_recycle, dim=0)
        output_s["final_atom_mask"] = input_dic["atom37_atom_exists"] #[8, 1, 32, 100, 14, 3]
        return output_s

    def run_wo_recycle(
        self,
        s: torch.Tensor,  # [S, L]
        z: torch.Tensor,
        r: torch.Tensor,
        ulr_mask: torch.Tensor,
        str_mask: torch.Tensor,
        torsion_angle: torch.Tensor,
        aatype: torch.Tensor,
        sampled_n_recycle,
        batch,
        lang_out=None,
        build_str_interval=1,
        stop_gradient=True,
        train_mode=True,
        inference_mode=False,
        tmp_stat=False,
        reset_structure=False,
        force_freeze_generator=False,
    ):
        is_final_iter = True
        if self.is_train:
            str_mask = None

        if train_mode:
            is_grad_enabled = True
        else:
            is_grad_enabled = False
        if force_freeze_generator:
            is_grad_enabled = False

        rel_pos = torch.nn.functional.one_hot(
            prep_rel_pos(batch["hu_residue_index"], batch["chain_id"]), num_classes=66
        ).float()  # [S, L, L, 66]

        input_r = self.initial_bank_generator(
            s, z, r, ulr_mask, str_mask, train_mode, inference_mode
        )  # Rigid [S, L]
        ##
        input_s = s.clone().detach()
        ##
        #! No_internal Z_update
        new_r = clone_rigid(input_r)
        
        with torch.set_grad_enabled(is_grad_enabled):
            ##
            s = self.seq_embedder(input_s.long(), lang_out, ulr_mask, torsion_angle)
            # [S, L, 128]
            z = self.workingz(s, s, None, new_r, ulr_mask, rel_pos, naive=True)
            # print (z.shape)
            # [S, L, L, 329]
            z = self.act_fn(self.emb_z_initial(z))
            # [S, L, L, 96]
            z = z + self.act_fn(self.linear_rel_pos(rel_pos))
            z = self.norm_z_initial(z)
            # [S, L, L, 96]
            if reset_structure:
                new_r = clone_rigid(input_r)
            if not self.run_mode == "withouth_IPA_enc":
                s = self.IPA_enc(
                    s,
                    z,
                    new_r,
                    str_mask,
                    ulr_mask,
                    aatype,
                    rel_pos,
                    batch,
                    return_output=False,
                    is_enc=True,
                )
            new_s, new_z, new_r, output_s = self.IPA_block_w_no_recycle(
                s,
                z,
                new_r,
                str_mask,
                ulr_mask,
                aatype,
                rel_pos,
                batch,
                return_output=is_final_iter,
             )
            ##
        output_s.update(self.aux_heads(output_s,train_mode))
        return output_s

    def run_with_recycle_ver2(
        self,
        s: torch.Tensor,  # [S, L]
        z: torch.Tensor,
        r: torch.Tensor,
        ulr_mask: torch.Tensor,
        str_mask: torch.Tensor,
        torsion_angle: torch.Tensor,
        aatype: torch.Tensor,
        sampled_n_recycle,
        batch,
        lang_out=None,
        build_str_interval=1,
        stop_gradient=True,
        train_mode=True,
        inference_mode=False,
        tmp_stat=False,
        reset_structure=False,
        force_freeze_generator=False,
    ):
        if self.is_train:
            str_mask = None

        if train_mode:
            is_grad_enabled = True
        else:
            is_grad_enabled = False
        if force_freeze_generator:
            is_grad_enabled = False

        rel_pos = torch.nn.functional.one_hot(
            prep_rel_pos(batch["hu_residue_index"], batch["chain_id"]), num_classes=66
        ).float()  # [S, L, L, 66]

        input_r = self.initial_bank_generator(
            s, z, r, ulr_mask, str_mask, train_mode, inference_mode
        )  # Rigid [S, L]
        ##
        input_s = s.clone().detach()
        ##
        #! No_internal Z_update
        new_r = clone_rigid(input_r)


        initial_14 = torch.zeros(4,*input_r.shape, 14, 3).to(input_r.device) #[4, 1, 32, 100, 3, 3]
        initial_bb = get_bb_pos(input_r, 10).to(input_r.device) #[1, 1, 32, 100, 3, 3]
        initial_bb = initial_bb.unsqueeze(0).expand(4, *initial_bb.shape)
        initial_14[..., :3, :] = initial_bb 

        cross_over_prev = []
        cross_over_after = []
        attention_value = []
        positions_all_recycle = []
        positions_all_recycle.append(initial_14)

        for cycle_idx in range(sampled_n_recycle):
            is_final_iter = cycle_idx + 1 == sampled_n_recycle
            with torch.set_grad_enabled(
                is_grad_enabled and is_final_iter
            ):  # grad enable only in last recycle
                ##
                s = self.seq_embedder(input_s.long(), lang_out, ulr_mask, torsion_angle)
                # [S, L, 128]
                z = self.workingz(s, s, None, new_r, ulr_mask, rel_pos, naive=True)
                # print (z.shape)
                # [S, L, L, 329]
                z = self.act_fn(self.emb_z_initial(z))
                # [S, L, L, 96]
                z = z + self.act_fn(self.linear_rel_pos(rel_pos))
                z = self.norm_z_initial(z)
                # [S, L, L, 96]
                if tmp_stat:
                    if cycle_idx == 0:
                        new_s = torch.zeros_like(s)
                        new_z = torch.zeros_like(z)
                        s, z = self.recycle_embedder(s, z, new_s, new_z)
                if not cycle_idx == 0:  # using recycling
                    s, z = self.recycle_embedder(s, z, new_s, new_z)
                if reset_structure:
                    new_r = clone_rigid(input_r)
                if not self.run_mode == "withouth_IPA_enc":
                    s = self.IPA_enc(
                        s,
                        z,
                        new_r,
                        str_mask,
                        ulr_mask,
                        aatype,
                        rel_pos,
                        batch,
                        return_output=False,
                        is_enc=True,
                    )
                new_s, new_z, new_r, output_s = self.IPA_block(
                    s,
                    z,
                    new_r,
                    str_mask,
                    ulr_mask,
                    aatype,
                    rel_pos,
                    batch,
                    return_output=is_final_iter,
                )
                cross_over_prev.append(output_s['cross_over_prev'])
                cross_over_after.append(output_s['cross_over_after'])
                attention_value.append(output_s['attention_value'])
                positions_all_recycle.append(output_s['sm']['positions'])
                ##
        output_s.update(self.aux_heads(output_s,train_mode))
        output_s['cross_over_prev'] = torch.stack(cross_over_prev, dim=0)
        #[n_recycle, n_block, B, S, L, 3, 3] 
        output_s['cross_over_after'] = torch.stack(cross_over_after, dim=0)
        output_s['attention_value'] = torch.stack(attention_value, dim=0)
        # [n_recycle, n_block_co, B, L, H, S, S]
        output_s['positions_all_recycle'] = torch.stack(positions_all_recycle, dim=0)
        # [n_recycle, n_block_ipa, B, S, L, 14, 3]
        return output_s


#
