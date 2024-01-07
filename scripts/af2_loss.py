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

from functools import partial
import logging, sys
import ml_collections
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from typing import Dict, Optional, Tuple

from openfold.np import residue_constants
from openfold.utils import feats
from openfold.utils.rigid_utils import Rotation, Rigid
from openfold.utils.tensor_utils import (
    tree_map,
    tensor_tree_map,
    masked_mean,
    permute_final_dims,
    batched_gather,
)
from openfold.np import residue_constants as rc


def select_loss_from_decoy(sel_loss):
    # sel_loss :B,S
    min_idx = torch.argmin(sel_loss, dim=-1)
    min_idx = (torch.arange(min_idx.shape[0]).to(device=min_idx.device), min_idx)
    return min_idx


def softmax_cross_entropy(logits, labels):
    loss = -1 * torch.sum(
        labels * torch.nn.functional.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss


def sigmoid_cross_entropy(logits, labels):
    logits_dtype = logits.dtype
    logits = logits.double()
    labels = labels.double()
    log_p = torch.nn.functional.logsigmoid(logits)
    # log_p = torch.log(torch.sigmoid(logits))
    log_not_p = torch.nn.functional.logsigmoid(-1 * logits)
    # log_not_p = torch.log(torch.sigmoid(-logits))
    loss = (-1.0 * labels) * log_p - (1.0 - labels) * log_not_p
    loss = loss.to(dtype=logits_dtype)
    return loss


def get_local_sidechain_mask(
    target_positions,
    residx_rigidgroup_base_atom14_idx,
    frames_mask,
    positions_mask,
    local_criteria,
):
    batch_size = target_positions.shape[-4]
    target_positions_zero = target_positions[..., 0, :, :]  # [B,1,L*14, 3]
    target_positions_reshape = target_positions_zero.view(
        batch_size, 1, -1, 14, 3
    )  # [B,1,L, 14, 3]
    target_positions_reshape = target_positions_reshape.expand(
        batch_size, 8, *target_positions_reshape.shape[-3:]
    )
    # [B, 8, L, 14, 3]
    assert (
        residx_rigidgroup_base_atom14_idx.shape[-2]
        == target_positions_reshape.shape[-3]
    )
    index_tensor = (
        residx_rigidgroup_base_atom14_idx[:, 0]
        .unsqueeze(-1)
        .expand(*residx_rigidgroup_base_atom14_idx[:, 0].shape, 3)
    )  # [B,8, L, 3, 3]
    target_positions_select = torch.gather(
        target_positions_reshape, -2, index_tensor
    )  # [B, 8, L, 3, 3]
    target_positions_select = target_positions_select.view(
        *target_positions_select.shape[:-3], -1, 3
    )
    # [B, 8, L*3, 3]
    dist_matrix = (
        target_positions_select[..., :, None, :]
        - target_positions_zero[..., None, None, :, :]
    )
    dist_matrix = torch.norm(dist_matrix, dim=-1)  # [B, 8, L*3, L*14]
    aa_length = int(dist_matrix.shape[-1] / 14)
    dist_matrix = dist_matrix.view(
        *dist_matrix.shape[:-3], 8, aa_length, 3, -1
    )  # [*, 8, L, 3, 14L]
    dist_matrix = dist_matrix.transpose(-3, -4)  # [L, 8, 3, 14L]
    dist_matrix_reshape = dist_matrix.reshape(
        *dist_matrix.shape[:-4], aa_length * 8, 3, -1
    )  # [B, L*8, 3, 14L]
    dist_matrix_frame_mask = (
        1 - frames_mask[:, 0][..., None, None]
    ) * 100 + dist_matrix_reshape * frames_mask[:, 0][
        ..., None, None
    ]  # [B, L*8, 3, 14L]

    dist_matrix_frame_mask = dist_matrix_frame_mask.view(
        *dist_matrix_frame_mask.shape[:-3], -1, dist_matrix_frame_mask.shape[-1]
    )  # [B, L*8*3, L*14]

    dist_matrix_final = (
        1 - positions_mask[:, 0][..., None, :]
    ) * 100 + dist_matrix_frame_mask * positions_mask[:, 0][..., None, :]
    # [B, L*8*3, L*14]

    dist_matrix_final = dist_matrix_final.view(
        *dist_matrix_final.shape[:-2], -1, 24, dist_matrix_final.shape[-1]
    )
    # [B, L, 8*3, L*14]

    dist_matrix_final = dist_matrix_final.view(
        *dist_matrix_final.shape[:-3], -1, 8, 3, dist_matrix_final.shape[-1]
    )
    # [B, L, 8, 3, L*14]
    assert dist_matrix_final.shape[-1] % 14 == 0
    min_values, _ = torch.min(
        dist_matrix_final,
        dim=-2,
    )
    # [B, L, 8, 3, L*14] -> [B, L, 8, L*14]

    min_values = min_values.view(
        *min_values.shape[:-3], -1, min_values.shape[-1]
    )  # [B,L*8, L*14]

    dist_matrix_mask = min_values < local_criteria  # [B, L*8, L*14]
    dist_matrix_mask = dist_matrix_mask[:, None, ...].expand(
        -1, frames_mask.shape[1], -1, -1
    )
    # [B, 32, L*8,L*14]
    return dist_matrix_mask


def compute_fape(
    pred_frames: Rigid,
    target_frames: Rigid,
    frames_mask: torch.Tensor,
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
    length_scale: float,
    ulr_mask: Optional[torch.Tensor] = None,
    length_scale_intra: Optional[float] = None,
    length_scale_inter: Optional[float] = None,
    l1_clamp_distance: Optional[float] = None,
    pair_mask: Optional[torch.Tensor] = None,
    run_mode: Optional[str] = None,
    eps=1e-8,
    mask_bb_criteria=3.0,
    mask_local_bb_criteria=1.0,
    use_bb_mask: bool = False,
    use_local_FAPE: bool = False,
    local_bb_criteria: Optional[float] = 12.0,
    local_sc_criteria: Optional[float] = 5.0,
    residx_rigidgroup_base_atom14_idx: Optional[bool] = None,
    use_non_ulr=False,
    weight=0.0,
    weight_supervised_chi=0.0,
    backbone_torsion_chi_mask: torch.Tensor = None,
    sidechain_frame_mask: torch.Tensor = None,
    using_distance_weight=False,
) -> torch.Tensor:
    """
    Computes FAPE loss.

    Args:
        pred_frames:
            [*, N_frames] Rigid object of predicted frames
        target_frames:
            [*, N_frames] Rigid object of ground truth frames
        frames_mask:
            [*, N_frames] binary mask for the frames
        pred_positions:
            [*, N_pts, 3] predicted atom positions
        target_positions:
            [*, N_pts, 3] ground truth positions
        positions_mask:
            [*, N_pts] positions mask
        length_scale:
            Length scale by which the loss is divided
        l1_clamp_distance:
            Cutoff above which distance errors are disregarded
        eps:
            Small value used to regularize denominators
    Returns:
        [*] loss tensor
    """
    # [*, N_frames, N_pts, 3]
    # If bb mode
    # pred_frames [*, N_seed, L]
    # target_frames [*, N_seed, L]
    # frame_mask    [*, N_seed, L]
    # pred_positions [*, N_seed, L, 3]
    # target_positions [*, N_seed, L, 3]
    # positions_mask [*, N_seed, L]
    #

    # if sc mode
    # pred_frames, # [N_seed, L*8]
    # target_frames, # [N_seed, L*8]
    # frames_mask, # [N_seed, L*8]
    # pred_positions, # [N_seed, L*14, 3]
    # target_positions, # [N_seed, L*14, 3]
    # positions_mask, # [N_seed, L*14]

    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :],
    )

    # if bb -> local_pred_pos & local_target_pos [*, N_seed, L, L,  3]
    # if sc -> local_pred_pos & local_target_pos [N_seed, L*8, L*14, 3]

    error_dist = torch.sqrt(
        torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
    )
    # if bb -> error_dist [*, N_seed, L, L]
    # if sc -> error_dist [N_seed, 8*L, 14*L]

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)
    # [*,N_frames, N_pts] : error_dist

    # 3 cases
    # 1. run_mode == sc : just use raw frames_mask
    # 2. run_mode == bb & ulr_mask: use frames_mask * ulr_mask
    # 3. run_mode == bb & non_ulr_mask: use frames_mask * (~ulr_mask), use_frames_mask * ulr_mask
    if run_mode == "bb":
        frames_mask_ori = frames_mask
        assert ulr_mask is not None
        if use_non_ulr:
            frames_mask_ulr = frames_mask * ulr_mask[None, ...]
            frames_mask_non_ulr = frames_mask * (~ulr_mask[None, ...])
        else:
            # has problem if calculate side chain torsion mask
            frames_mask = frames_mask * ulr_mask[None, ...]
        # positions_mask=positions_mask * ulr_mask[None,...]

    if run_mode == "sc":
        if use_bb_mask:  # sc loss + bb_mask
            # backbone_torsion_chi_mask [1, N_seed, L]
            # dist_matrix_mask [N_seed, L*8, L*14]
            backbone_torsion_sc_mask = (~backbone_torsion_chi_mask.bool())[
                0
            ]  # [N_seed, L]
            backbone_torsion_frames_mask = backbone_torsion_sc_mask.unsqueeze(
                -1
            ).expand(
                *backbone_torsion_sc_mask.shape, 8
            )  # [N_seed, L, 8]
            backbone_torsion_frames_mask = backbone_torsion_frames_mask.clone()
            backbone_torsion_frames_mask[..., 0] = 1  # backbone frame = 1
            backbone_torsion_frames_mask = backbone_torsion_frames_mask.reshape(
                *backbone_torsion_sc_mask.shape[:-1], -1
            )
            # [N_seed, L * 8]

            backbone_torsion_positions_mask = backbone_torsion_sc_mask.unsqueeze(
                -1
            ).expand(
                *backbone_torsion_sc_mask.shape, 14
            )  # [N_seed, L, 14]
            backbone_torsion_positions_mask = backbone_torsion_positions_mask.clone()
            backbone_torsion_positions_mask[..., :3] = 1  # [N, CA, C] = 1
            backbone_torsion_positions_mask = backbone_torsion_positions_mask.reshape(
                *backbone_torsion_sc_mask.shape[:-1], -1
            )
            # [N_seed, L * 14]
            frames_mask = frames_mask * backbone_torsion_frames_mask
            positions_mask = positions_mask * backbone_torsion_positions_mask

    if use_local_FAPE:  # target_positions [*, N_seed, L, 3]
        if run_mode == "bb":
            dist_matrix_target = (
                target_positions[..., None, :, :] - target_positions[..., :, None, :]
            )
            dist_matrix_pred = (
                pred_positions[..., None, :, :] - pred_positions[..., :, None, :]
            )
            dist_matrix_target = torch.norm(
                dist_matrix_target, dim=-1
            )  # [*, N_seed, L, L]
            dist_matrix_pred = torch.norm(dist_matrix_pred, dim=-1)  # [*, N_seed, L, L]
            dist_matrix_target_mask = dist_matrix_target < local_bb_criteria
            dist_matrix_pred_mask = dist_matrix_pred < local_bb_criteria
            dist_matrix_mask = torch.logical_or(
                dist_matrix_target_mask, dist_matrix_pred_mask
            )
            normed_error = error_dist * dist_matrix_mask
            normed_error = normed_error / length_scale
            if using_distance_weight:
                with torch.no_grad():
                    tmp = dist_matrix_target.expand(
                        dist_matrix_pred.shape[0], *dist_matrix_target.shape[1:]
                    ).clone()
                    tmp_distance_weight = torch.stack([dist_matrix_pred, tmp], dim=-1)
                    tmp_distance_weight = torch.min(tmp_distance_weight, dim=-1)[0]
                    tmp_distance_weight = 1 + tmp_distance_weight
                    ###
                    filtering_mask = dist_matrix_mask * positions_mask[..., None, :]
                    filtering_val = tmp_distance_weight * filtering_mask
                    filtering_val = filtering_val.sum(dim=-1) / (
                        eps + positions_mask[..., None, :].sum(dim=-1)
                    )
                    ###
                    del (filtering_val, tmp)
                    tmp_distance_weight = filtering_val[..., None] * (
                        (tmp_distance_weight) ** (-1)
                    )
                normed_error = normed_error * tmp_distance_weight
        elif run_mode == "sc":  # target_positions,[N_seed, L*14, 3]
            # residx_rigidgroup_base_atom14_idx, [8, 128, 3]
            # target_positions = [*, N_seed, 14 * L, 3]
            # dist_matrixt_mask [N_seed, L*8,L*14]
            dist_matrix_mask_target = get_local_sidechain_mask(
                target_positions,
                residx_rigidgroup_base_atom14_idx,
                frames_mask,
                positions_mask,
                local_sc_criteria,
            )
            dist_matrix_mask_pred = get_local_sidechain_mask(
                pred_positions,
                residx_rigidgroup_base_atom14_idx,
                frames_mask,
                positions_mask,
                local_sc_criteria,
            )
            dist_matrix_mask = torch.logical_or(
                dist_matrix_mask_target, dist_matrix_mask_pred
            )

            normed_error = error_dist * dist_matrix_mask
            normed_error = normed_error / length_scale
            if using_distance_weight:
                with torch.no_grad():
                    tmp_distance_weight = torch.stack(
                        [dist_matrix_pred, dist_matrix_target], dim=-1
                    )
                    tmp_distance_weight = torch.min(tmp_distance_weight, dim=-1)[0]
                    tmp_distance_weight = 1 + tmp_distance_weight
                    filtering_mask = dist_matrix_mask * positions_mask[..., None, :]
                    filtering_val = tmp_distance_weight * filtering_mask
                    filtering_val = filtering_val.sum(dim=-1) / (
                        eps + positions_mask[..., None, :].sum(dim=-1)
                    )
                    ###
                    tmp_distance_weight = filtering_val[..., None] * (
                        (tmp_distance_weight) ** (-1)
                    )
                normed_error = normed_error * tmp_distance_weight

    else:
        normed_error = error_dist / length_scale

    # calculate backbone mask for torsion calculation (total bb mode only, not local error)
    if run_mode == "bb":
        if not use_local_FAPE:
            normed_error_for_mask = error_dist * frames_mask_ori[..., None]
            normed_error_for_mask = normed_error_for_mask * positions_mask[..., None, :]

            normed_error_for_mask = torch.sum(normed_error_for_mask, dim=-1)
            normed_error_for_mask = normed_error_for_mask / (
                eps + torch.sum(positions_mask[..., None, :], dim=-1)
            )  # [*, N_seed, L]
            backbone_mask = normed_error_for_mask < mask_bb_criteria
            backbone_mask = backbone_mask.float()
            backbone_mask *= frames_mask_ori  # [1, 32, L]
            # If use this mask, then upweight the backbone loss in case of incorrect prediction

        else:
            normed_error_for_mask = (
                error_dist * frames_mask_ori[..., None] * dist_matrix_mask
            )
            normed_error_for_mask = normed_error_for_mask * positions_mask[..., None, :]
            normed_error_for_mask = torch.sum(
                normed_error_for_mask, dim=-1
            )  # [*, N_seed, L]
            positions_mask_bb = (
                positions_mask[..., None, :] * dist_matrix_mask
            )  # [*, N_seed, L, L]
            normed_error_for_mask = normed_error_for_mask / (
                eps + torch.sum(positions_mask_bb, dim=-1)
            )  # [*, N_seed, L]
            backbone_mask = normed_error_for_mask < mask_local_bb_criteria
            backbone_mask = backbone_mask.float()
            backbone_mask *= frames_mask_ori

        if use_bb_mask:
            # which number to upweight? right now simply multiply 3
            chi_loss_max = 2 ** (0.5)
            weight_ratio = weight_supervised_chi / (weight + eps)
            penalizing_weight = mask_bb_criteria + weight_ratio * chi_loss_max
            penalizing_tensor = torch.full(backbone_mask.size(), penalizing_weight).to(
                device=backbone_mask.device
            )
            assert not sidechain_frame_mask is None
            sidechain_frame_mask_count = torch.sum(sidechain_frame_mask, dim=-1)
            sidechain_frame_mask_count = sidechain_frame_mask_count - 1
            penalizing_tensor += (
                sidechain_frame_mask_count.unsqueeze(0)
                * l1_clamp_distance
                / length_scale
            )
            penalizing_tensor /= mask_bb_criteria
            normed_error = (
                normed_error
                * (1 - backbone_mask[..., None])
                * penalizing_tensor[..., None]
                + normed_error * backbone_mask[..., None]
            )  # [*, N_seed, L, L]

    if run_mode == "bb" and use_non_ulr:
        normed_error_ulr = normed_error * frames_mask_ulr[..., None]
        normed_error_ulr = normed_error_ulr * positions_mask[..., None, :]
        normed_error_non_ulr = normed_error * frames_mask_non_ulr[..., None]
        normed_error_non_ulr = normed_error_non_ulr * positions_mask[..., None, :]
    else:
        normed_error = normed_error * frames_mask[..., None]
        normed_error = normed_error * positions_mask[..., None, :]

    # bb -> normed_error [*, N_seed, L, L] * [*, N_seed, L, 1] * [*, N_seed, 1, L]
    # sc -> normed_error [N_seed, 8*L, 14*L] * [N_seed, 8*L, 1] * [N_seed, 1, 14*L]

    # if pair_mask == None:
    #    ####
    if use_non_ulr and run_mode == "bb":
        if use_local_FAPE:
            # ulr region
            normed_error_ulr = torch.sum(normed_error_ulr, dim=-1)  # *,N_seed, L
            positions_mask_ulr = positions_mask[..., None, :] * dist_matrix_mask
            normed_error_ulr = normed_error_ulr / (
                eps + torch.sum(positions_mask_ulr, dim=-1)
            )
            frames_mask_ulr = frames_mask_ulr[..., None] * dist_matrix_mask
            frames_mask_any_ulr = torch.any(frames_mask_ulr != 0, dim=-1).bool()
            normed_error_ulr = torch.sum(normed_error_ulr, dim=-1)  # *, N_seed
            normed_error_ulr = normed_error_ulr / (
                eps + torch.sum(frames_mask_any_ulr, dim=-1)
            )
            # non-ulr region
            normed_error_non_ulr = torch.sum(
                normed_error_non_ulr, dim=-1
            )  # *,N_seed, L
            positions_mask_non_ulr = positions_mask[..., None, :] * dist_matrix_mask
            normed_error_non_ulr = normed_error_non_ulr / (
                eps + torch.sum(positions_mask_non_ulr, dim=-1)
            )
            frames_mask_non_ulr = frames_mask_non_ulr[..., None] * dist_matrix_mask
            frames_mask_any_non_ulr = torch.any(frames_mask_non_ulr != 0, dim=-1).bool()
            normed_error_non_ulr = torch.sum(normed_error_non_ulr, dim=-1)  # *, N_seed
            normed_error_non_ulr = normed_error_non_ulr / (
                eps + torch.sum(frames_mask_any_non_ulr, dim=-1)
            )

        else:
            # ulr region
            normed_error_ulr = torch.sum(normed_error_ulr, dim=-1)  # *,N_seed, L
            normed_error_ulr = (
                normed_error_ulr / (eps + torch.sum(frames_mask_ulr, dim=-1))[..., None]
            )
            normed_error_ulr = torch.sum(normed_error_ulr, dim=-1)  # *, N_seed
            normed_error_ulr = normed_error_ulr / (
                eps + torch.sum(positions_mask, dim=-1)
            )
            # non ulr region
            normed_error_non_ulr = torch.sum(
                normed_error_non_ulr, dim=-1
            )  # *,N_seed, L
            normed_error_non_ulr = (
                normed_error_non_ulr
                / (eps + torch.sum(frames_mask_non_ulr, dim=-1))[..., None]
            )
            normed_error_non_ulr = torch.sum(normed_error_non_ulr, dim=-1)  # *, N_seed
            normed_error_non_ulr = normed_error_non_ulr / (
                eps + torch.sum(positions_mask, dim=-1)
            )

        normed_error = normed_error_ulr + normed_error_non_ulr
    else:
        if use_local_FAPE:
            normed_error = torch.sum(normed_error, dim=-1)
            positions_mask = positions_mask[..., None, :] * dist_matrix_mask
            normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))

            frames_mask = frames_mask[..., None] * dist_matrix_mask
            frames_mask_any = torch.any(frames_mask != 0, dim=-1).bool()
            normed_error = torch.sum(normed_error, dim=-1)
            normed_error = normed_error / (eps + torch.sum(frames_mask_any, dim=-1))
        else:
            normed_error = torch.sum(normed_error, dim=-1)  # *,N_seed, L
            # frames_mask = [*, N_seed, L] -> sum [*, N_seed]
            # positions_mask = [*, N_seed, L] -> sum[*, N_seed]
            normed_error = (
                normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
            )
            normed_error = torch.sum(normed_error, dim=-1)  # *, N_seed
            normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))
    if run_mode == "bb":
        return normed_error, backbone_mask
    else:
        return normed_error


def backbone_loss(
    backbone_rigid_tensor: torch.Tensor,
    backbone_rigid_mask: torch.Tensor,
    traj: torch.Tensor,
    rigidgroups_gt_exists: torch.Tensor,
    use_clamped_fape: Optional[torch.Tensor] = None,
    clamp_distance: float = 10.0,
    loss_unit_distance: float = 10.0,
    eps: float = 1e-4,
    ulr_mask: Optional[torch.Tensor] = None,
    use_non_ulr=False,
    use_local_FAPE=False,
    use_bb_mask=False,
    weight_supervised_chi=0.0,
    weight=0.0,
    mask_bb_criteria=0.0,
    using_distance_weight=False,
    **kwargs,
) -> torch.Tensor:
    ##

    pred_aff = Rigid.from_tensor_7(traj)
    pred_aff = Rigid(
        Rotation(rot_mats=pred_aff.get_rots().get_rot_mats(), quats=None),
        pred_aff.get_trans(),
    )
    # DISCREPANCY: DeepMind somehow gets a hold of a tensor_7 version of
    # backbone tensor, normalizes it, and then turns it back to a rotation
    # matrix. To avoid a potentially numerically unstable rotation matrix
    # to quaternion conversion, we just use the original rotation matrix
    # outright. This one hasn't been composed a bunch of times, though, so
    # it might be fine.
    gt_aff = Rigid.from_tensor_4x4(backbone_rigid_tensor)

    fape_loss, backbone_mask = compute_fape(
        pred_aff,  # [*, N_seed, L]
        gt_aff[None],  # [*, N_seed, L]
        backbone_rigid_mask[None],  # [*, N_seed, L]
        pred_aff.get_trans(),
        gt_aff[None].get_trans(),
        backbone_rigid_mask[None],
        l1_clamp_distance=clamp_distance,
        length_scale=loss_unit_distance,
        eps=eps,
        ulr_mask=ulr_mask,  # [N_seed, L]
        run_mode="bb",
        use_non_ulr=use_non_ulr,
        use_local_FAPE=use_local_FAPE,
        use_bb_mask=use_bb_mask,
        weight_supervised_chi=weight_supervised_chi,
        weight=weight,
        mask_bb_criteria=mask_bb_criteria,
        sidechain_frame_mask=rigidgroups_gt_exists,
        using_distance_weight=using_distance_weight,
    )
    if use_clamped_fape is not None:
        unclamped_fape_loss, backbone_mask = compute_fape(
            pred_aff,
            gt_aff[None],
            backbone_rigid_mask[None],
            pred_aff.get_trans(),
            gt_aff[None].get_trans(),
            backbone_rigid_mask[None],
            l1_clamp_distance=None,
            length_scale=loss_unit_distance,
            eps=eps,
            ulr_mask=ulr_mask,
            run_mode="bb",
            use_non_ulr=use_non_ulr,
            use_local_FAPE=use_local_FAPE,
            use_bb_mask=use_bb_mask,
            sidechain_frame_mask=rigidgroups_gt_exists,
            using_distance_weight=using_distance_weight,
        )

        fape_loss = fape_loss * use_clamped_fape + unclamped_fape_loss * (
            1 - use_clamped_fape
        )
    # Average over the batch dimension
    # fape_loss = torch.mean(fape_loss)
    return fape_loss, backbone_mask


def sidechain_loss(
    sidechain_frames: torch.Tensor,  # [*, N_seed, L, 8, 4, 4]
    sidechain_atom_pos: torch.Tensor,  # [*, N_seed, L, 14, 3]
    rigidgroups_gt_frames: torch.Tensor,  # [N_seed, L, 8, 4, 4]
    rigidgroups_alt_gt_frames: torch.Tensor,  # [N_seed, L, 8, 4, 4]
    rigidgroups_gt_exists: torch.Tensor,  # [N_seed, L, 8]
    renamed_atom14_gt_positions: torch.Tensor,  # [N_seed, L, 14, 3]
    renamed_atom14_gt_exists: torch.Tensor,  # [N_seed, L, 14]
    alt_naming_is_better: torch.Tensor,  # [N_seed, L]
    ulr_mask: Optional[torch.Tensor] = None,  # [N_seed, L]
    clamp_distance: float = 10.0,
    length_scale: float = 10.0,
    eps: float = 1e-4,
    use_mask: bool = False,
    all_atom_mask: Optional[torch.Tensor] = None,
    aatype: Optional[torch.Tensor] = None,
    use_local_FAPE: bool = False,
    use_non_ulr: bool = False,
    backbone_torsion_chi_mask: Optional[torch.Tensor] = None,
    use_bb_mask: bool = False,
    using_distance_weight=False,
    **kwargs,
) -> torch.Tensor:
    renamed_gt_frames = (
        1.0 - alt_naming_is_better[..., None, None, None]
    ) * rigidgroups_gt_frames + alt_naming_is_better[
        ..., None, None, None
    ] * rigidgroups_alt_gt_frames
    ####
    sidechain_frames = sidechain_frames[-1]  # [N_seed, L, 8, 4, 4]
    batch_dims = sidechain_frames.shape[:-4]  # N_seed
    sidechain_frames = sidechain_frames.view(
        *batch_dims, -1, 4, 4
    )  # [N_seed, L*8, 4, 4]
    sidechain_frames = Rigid.from_tensor_4x4(sidechain_frames)  # [N_seed, L*8]
    renamed_gt_frames = renamed_gt_frames.view(
        *batch_dims, -1, 4, 4
    )  # [N_seed, L*8, 4, 4]
    renamed_gt_frames = Rigid.from_tensor_4x4(renamed_gt_frames)  # [N_seed, L*8]
    if (not ulr_mask == None) and use_mask:
        rigidgroups_gt_exists = rigidgroups_gt_exists * ulr_mask[..., None]
        # [N_seed, L, 8] * [N_seed, L, 1] -> [N_seed, L, 8]
        # renamed_atom14_gt_exists = renamed_atom14_gt_exists * ulr_mask[...,None]
    rigidgroups_gt_exists = rigidgroups_gt_exists.reshape(
        *batch_dims, -1
    )  # [N_seed, L*8]
    sidechain_atom_pos = sidechain_atom_pos[-1]  # [N_seed, L, 14, 3]
    sidechain_atom_pos = sidechain_atom_pos.view(
        *batch_dims, -1, 3
    )  # [N_seed, L*14, 3]
    renamed_atom14_gt_positions = renamed_atom14_gt_positions.view(
        *batch_dims, -1, 3
    )  # [N_seed, L*14, 3]
    renamed_atom14_gt_exists = renamed_atom14_gt_exists.view(
        *batch_dims, -1
    )  # [N_seed, L*14]

    restype_rigidgroup_base_atom_names = np.full([21, 8, 3], "", dtype=object)
    restype_rigidgroup_base_atom_names[:, 0, :] = ["C", "CA", "N"]
    restype_rigidgroup_base_atom_names[:, 3, :] = ["CA", "C", "O"]

    for restype, restype_letter in enumerate(rc.restypes):
        resname = rc.restype_1to3[restype_letter]
        for chi_idx in range(4):
            if rc.chi_angles_mask[restype][chi_idx]:
                names = rc.chi_angles_atoms[resname][chi_idx]
                restype_rigidgroup_base_atom_names[restype, chi_idx + 4, :] = names[1:]

    restype_rigidgroup_mask = all_atom_mask.new_zeros(21, 8)
    restype_rigidgroup_mask[..., 0] = 1
    restype_rigidgroup_mask[..., 3] = 1
    restype_rigidgroup_mask[..., :20, 4:] = all_atom_mask.new_tensor(rc.chi_angles_mask)

    # lookuptable = rc.atom_order.copy()  # dictionary {atom:atom_idx}
    # lookuptable[""] = 0
    # lookup = np.vectorize(lambda x: lookuptable[x])
    from collections import defaultdict

    res_atom14_dict = defaultdict(dict)
    for i, (k, v) in enumerate(rc.restype_name_to_atom14_names.items()):
        for j, atmname in enumerate(v):
            res_atom14_dict[i][atmname] = j

    restype_rigidgroup_base_atom14_idx = np.full([21, 8, 3], -1, dtype=int)
    for i, v in enumerate(restype_rigidgroup_base_atom_names):
        lookuptable = res_atom14_dict[i]
        if i == 20:
            lookuptable = res_atom14_dict[0]
        lookuptable[""] = 0
        lookup = np.vectorize(lambda x: lookuptable[x])
        restype_rigidgroup_base_atom14_idx[i] = lookup(v)

    # restype_rigidgroup_base_atom37_idx = lookup(
    #    restype_rigidgroup_base_atom_names,
    # )  # [21,8,3]
    restype_rigidgroup_base_atom14_idx = aatype.new_tensor(
        restype_rigidgroup_base_atom14_idx,
    )  # [21, 8, 3]
    # restype_rigidgroup_base_atom37_idx = restype_rigidgroup_base_atom37_idx.view(
    #      *restype_rigidgroup_base_atom37_idx.shape
    # )  # [21, 8, 3]
    residx_rigidgroup_base_atom14_idx = batched_gather(
        restype_rigidgroup_base_atom14_idx,
        aatype,
        dim=-3,
    )  # [B, N_seed, L, 8, 3]
    group_exists = batched_gather(
        restype_rigidgroup_mask,
        aatype,
        dim=-2,
    )  # [N_seed, L, 8]

    group_exists = group_exists.bool()

    residx_rigidgroup_base_atom14_idx = (
        residx_rigidgroup_base_atom14_idx * group_exists.unsqueeze(-1)
    )

    residx_rigidgroup_base_atom14_idx = residx_rigidgroup_base_atom14_idx.transpose(
        -2, -3
    )

    fape = compute_fape(
        sidechain_frames,  # [N_seed, L*8]
        renamed_gt_frames,  # [N_seed, L*8]
        rigidgroups_gt_exists,  # [N_seed, L*8]
        sidechain_atom_pos,  # [N_seed, L*14, 3]
        renamed_atom14_gt_positions,  # [N_seed, L*14, 3]
        renamed_atom14_gt_exists,  # [N_seed, L*14]
        l1_clamp_distance=clamp_distance,
        length_scale=length_scale,
        residx_rigidgroup_base_atom14_idx=residx_rigidgroup_base_atom14_idx,
        eps=eps,
        ulr_mask=None,
        run_mode="sc",
        use_local_FAPE=use_local_FAPE,
        use_non_ulr=use_non_ulr,
        sidechain_frame_mask=None,  # not used in sidechain frames
        backbone_torsion_chi_mask=backbone_torsion_chi_mask,
        use_bb_mask=use_bb_mask,
        using_distance_weight=using_distance_weight,
    )

    return fape


def fape_bb_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    config: ml_collections.ConfigDict,
) -> torch.Tensor:
    bb_loss, bb_mask = backbone_loss(
        traj=out["sm"]["frames"],
        **{**batch, **config.backbone},
    )
    if not config.backbone.use_intermediate_str:
        bb_loss = bb_loss[-1:]
    batch["backbone_torsion_chi_mask"] = bb_mask[-1]
    # add mask to calculate torsion angle loss
    return bb_loss


def fape_sc_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    config: ml_collections.ConfigDict,
) -> torch.Tensor:
    sc_loss = sidechain_loss(
        out["sm"]["sidechain_frames"],
        out["sm"]["positions"],
        **{**batch, **config.sidechain},
    )
    return sc_loss


def supervised_chi_loss(
    angles_sin_cos: torch.Tensor,
    unnormalized_angles_sin_cos: torch.Tensor,
    aatype: torch.Tensor,
    # seq_mask: torch.Tensor,
    chi_mask: torch.Tensor,
    chi_angles_sin_cos: torch.Tensor,
    chi_weight: float,
    angle_norm_weight: float,
    backbone_torsion_chi_mask: torch.Tensor = None,
    eps=1e-6,
    use_bb_mask: bool = False,
    use_cumulative_loss: bool = False,
    **kwargs,
) -> torch.Tensor:
    """
    Implements Algorithm 27 (torsionAngleLoss)

    Args:
        angles_sin_cos:
            [*, N, 7, 2] predicted angles
        unnormalized_angles_sin_cos:
            The same angles, but unnormalized
        aatype:
            [*, N] residue indices
        seq_mask:
            [*, N] sequence mask
        chi_mask:
            [*, N, 7] angle mask
        chi_angles_sin_cos:
            [*, N, 7, 2] ground truth angles #[32, 100, 4, 2]
        chi_weight:
            Weight for the angle component of the loss
        angle_norm_weight:
            Weight for the normalization component of the loss
        backbone_torsion_chi_mask:
            [*, N] mask for backbone FAPE
    Returns:
        [*] loss tensor
    """
    seq_mask = torch.zeros_like(aatype).fill_(1).bool()
    pred_angles = angles_sin_cos[..., 3:, :]  # [*, N, 4, 2]
    residue_type_one_hot = torch.nn.functional.one_hot(
        aatype,
        residue_constants.restype_num + 1,
    )
    chi_pi_periodic = torch.einsum(
        "...ij,jk->ik",
        residue_type_one_hot.type(angles_sin_cos.dtype),
        angles_sin_cos.new_tensor(residue_constants.chi_pi_periodic),
    )
    # chi_mask [32, 100, 4]
    if use_bb_mask:
        assert not backbone_torsion_chi_mask is None
        backbone_torsion_chi_mask = backbone_torsion_chi_mask.view(
            *chi_mask.shape[:-1]
        )  # [1,B, S, 100] -> [B,S, 100]
        chi_mask = (
            chi_mask * backbone_torsion_chi_mask[..., None]
        )  # [B,N_seed, ,100, 4]
    true_chi = chi_angles_sin_cos[None]
    shifted_mask = (1 - 2 * chi_pi_periodic).unsqueeze(-1)
    true_chi_shifted = shifted_mask * true_chi

    if use_cumulative_loss:
        sq_chi_error_norm = torch.sqrt(
            torch.norm(true_chi - pred_angles + eps, dim=-1) + eps
        )  # [*, N, 4]
        sq_chi_error_norm_shifted = torch.sqrt(
            torch.norm(true_chi_shifted - pred_angles + eps, dim=-1) + eps
        )
        # [*, N, 4]
        sq_chi_error_norm = torch.minimum(sq_chi_error_norm, sq_chi_error_norm_shifted)
        # [*, N, 4]

        chi_1_error = sq_chi_error_norm[..., 0]
        chi_2_error = (
            chi_1_error
            + sq_chi_error_norm[..., 1]
            - (chi_1_error * sq_chi_error_norm[..., 1] / np.sqrt(2))
        )
        chi_3_error = (
            chi_2_error
            + sq_chi_error_norm[..., 2]
            - (chi_2_error * sq_chi_error_norm[..., 2] / np.sqrt(2))
        )
        chi_4_error = (
            chi_3_error
            + sq_chi_error_norm[..., 3]
            - (chi_3_error * sq_chi_error_norm[..., 3] / np.sqrt(2))
        )

        sq_chi_error = torch.cat(
            [chi_1_error, chi_2_error, chi_3_error, chi_4_error], dim=-1
        )

        sq_chi_error = sq_chi_error.view(*chi_1_error.shape, -1)
        # [*, N, 4]
        sq_chi_error = sq_chi_error.permute(
            *range(len(sq_chi_error.shape))[1:-2], 0, -2, -1
        )
        # sq_chi_error [32, 1, 100, 4]
        sq_chi_loss = masked_mean(
            chi_mask[..., None, :, :], sq_chi_error, dim=(-1, -2, -3)
        )
        # [N_seed]
        loss = chi_weight * sq_chi_loss

    else:
        sq_chi_error = torch.sum((true_chi - pred_angles) ** 2, dim=-1)
        sq_chi_error_shifted = torch.sum((true_chi_shifted - pred_angles) ** 2, dim=-1)
        sq_chi_error = torch.minimum(sq_chi_error, sq_chi_error_shifted)

        # The ol' switcheroo
        # sq_chi_error [1, 32, 100, 4]
        sq_chi_error = sq_chi_error.permute(
            *range(len(sq_chi_error.shape))[1:-2], 0, -2, -1
        )
        # chi_mask [32, 100, 4]
        # sq_chi_error [32, 1, 100, 4]
        sq_chi_loss = masked_mean(
            chi_mask[..., None, :, :], sq_chi_error, dim=(-1, -2, -3)
        )

        loss = chi_weight * sq_chi_loss

    angle_norm = torch.sqrt(torch.sum(unnormalized_angles_sin_cos**2, dim=-1) + eps)
    norm_error = torch.abs(angle_norm - 1.0)
    norm_error = norm_error.permute(*range(len(norm_error.shape))[1:-2], 0, -2, -1)
    angle_norm_loss = masked_mean(
        seq_mask[..., None, :, None], norm_error, dim=(-1, -2, -3)
    )
    loss = loss + angle_norm_weight * angle_norm_loss
    # Average over the batch dimension
    # loss = torch.mean(loss)
    return loss


def compute_plddt(logits: torch.Tensor) -> torch.Tensor:
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bounds = torch.arange(
        start=0.5 * bin_width, end=1.0, step=bin_width, device=logits.device
    )
    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_lddt_ca = torch.sum(
        probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape),
        dim=-1,
    )
    return pred_lddt_ca * 100


def lddt(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> torch.Tensor:
    n = all_atom_mask.shape[-2]
    dmat_true = torch.sqrt(
        eps
        + torch.sum(
            (all_atom_positions[..., None, :] - all_atom_positions[..., None, :, :])
            ** 2,
            dim=-1,
        )
    )

    dmat_pred = torch.sqrt(
        eps
        + torch.sum(
            (all_atom_pred_pos[..., None, :] - all_atom_pred_pos[..., None, :, :]) ** 2,
            dim=-1,
        )
    )
    dists_to_score = (
        (dmat_true < cutoff)
        * all_atom_mask
        * permute_final_dims(all_atom_mask, (1, 0))
        * (1.0 - torch.eye(n, device=all_atom_mask.device))
    )

    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25

    dims = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    score = norm * (eps + torch.sum(dists_to_score * score, dim=dims))

    return score


def lddt_ca(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> torch.Tensor:
    ca_pos = residue_constants.atom_order["CA"]
    all_atom_pred_pos = all_atom_pred_pos[..., ca_pos, :]
    all_atom_positions = all_atom_positions[..., ca_pos, :]
    all_atom_mask = all_atom_mask[..., ca_pos : (ca_pos + 1)]  # keep dim

    return lddt(
        all_atom_pred_pos,
        all_atom_positions,
        all_atom_mask,
        cutoff=cutoff,
        eps=eps,
        per_residue=per_residue,
    )


def lddt_loss(
    logits: torch.Tensor,
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    ulr_mask: torch.Tensor,
    # resolution: torch.Tensor,
    cutoff: float = 15.0,
    no_bins: int = 50,
    # min_resolution: float = 0.1,
    # max_resolution: float = 3.0,
    eps: float = 1e-10,
    # ignore_resolution =True,
    use_non_ulr=False,
    **kwargs,
) -> torch.Tensor:
    ##
    n = all_atom_mask.shape[-2]

    ca_pos = residue_constants.atom_order["CA"]
    all_atom_pred_pos = all_atom_pred_pos[..., ca_pos, :]
    all_atom_positions = all_atom_positions[..., ca_pos, :]
    all_atom_mask = all_atom_mask[..., ca_pos : (ca_pos + 1)]  # keep dim
    score = lddt(
        all_atom_pred_pos,
        all_atom_positions,
        all_atom_mask,
        cutoff=cutoff,
        eps=eps,
    )  # [N_seed, L_ulr]
    score = score.detach()
    bin_index = torch.floor(score * no_bins).long()
    bin_index = torch.clamp(bin_index, max=(no_bins - 1))
    lddt_ca_one_hot = torch.nn.functional.one_hot(bin_index, num_classes=no_bins)
    errors = softmax_cross_entropy(logits, lddt_ca_one_hot)
    all_atom_mask = all_atom_mask.squeeze(-1)

    mask = ulr_mask * all_atom_mask
    loss = torch.sum(errors * mask, dim=-1) / (eps + torch.sum(mask, dim=-1))
    if use_non_ulr:
        mask = (~ulr_mask) * all_atom_mask
        loss = loss + torch.sum(errors * mask, dim=-1) / (eps + torch.sum(mask, dim=-1))
    loss = torch.mean(loss, dim=-1)
    return loss


def rmsd_loss(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    ulr_mask: torch.Tensor,
    eps: float = 1e-10,
    hu_mode="bb",
    clamp_distance=10.0,
    use_non_ulr=False,
    **kwargs,
) -> torch.Tensor:
    # N ,CA, C
    ##Lousy but efficient calculation
    if hu_mode == "bb":
        all_atom_pred_pos = all_atom_pred_pos[..., 0:3, :]
        all_atom_positions = all_atom_positions[..., 0:3, :]
        all_atom_mask = all_atom_mask[..., 0:3]
    # ulr_mask [B,N_seed, L], torch.bool
    # all_atom_mask [B,N_seed, L, 3], torch.bool
    if not ulr_mask == None:
        all_atom_mask_ulr = all_atom_mask * ulr_mask[..., None]
        all_atom_mask_non_ulr = all_atom_mask * (~ulr_mask)[..., None]
        all_atom_mask_ulr = all_atom_mask_ulr.bool()
        all_atom_mask_non_ulr = all_atom_mask_non_ulr.bool()
    diff = all_atom_pred_pos - all_atom_positions
    diff = (diff**2).sum(dim=-1)
    diff = torch.clamp(diff, max=clamp_distance**2)  # B,S,L
    diff_ulr = diff * all_atom_mask_ulr.long()  # B,S,L,N
    diff_ulr = diff_ulr.sum(dim=-1).sum(dim=-1)  # B,S
    div_ulr = all_atom_mask_ulr.long().sum(dim=-1).sum(dim=-1)  # B,S
    diff_ulr = diff_ulr / (div_ulr + eps)
    diff_ulr = torch.sqrt(diff_ulr + eps)  # B,S
    loss = diff_ulr
    if use_non_ulr:
        diff_non_ulr = diff * all_atom_mask_non_ulr.long()  # B,S,L,N
        diff_non_ulr = diff_non_ulr.sum(dim=-1).sum(dim=-1)  # B,S
        div_non_ulr = all_atom_mask_non_ulr.long().sum(dim=-1).sum(dim=-1)  # B,S
        diff_non_ulr = diff_non_ulr / (div_non_ulr + eps)
        diff_non_ulr = torch.sqrt(diff_non_ulr + eps)  # B,S
        loss = loss + diff_non_ulr
    return loss

def total_rmsd_loss(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    ulr_mask: torch.Tensor,
    eps: float = 1e-10,
    hu_mode="bb",
    use_non_ulr=False,
    **kwargs,
):
    with torch.no_grad():
        true_rmsd = rmsd_loss(
            all_atom_pred_pos,
            all_atom_positions,
            all_atom_mask,
            ulr_mask,
            use_non_ulr=use_non_ulr,
            **kwargs,
        )  # B,S
    return true_rmsd

def best_rmsd_loss(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    ulr_mask: torch.Tensor,
    eps: float = 1e-10,
    hu_mode="bb",
    use_non_ulr=False,
    **kwargs,
):
    with torch.no_grad():
        true_rmsd = rmsd_loss(
            all_atom_pred_pos,
            all_atom_positions,
            all_atom_mask,
            ulr_mask,
            use_non_ulr=use_non_ulr,
            **kwargs,
        )  # B,S
        sel_idx = torch.argmin(true_rmsd, dim=-1)  # B
        sel_idx = (torch.arange(sel_idx.shape[0]).to(device=sel_idx.device), sel_idx)
    return true_rmsd[sel_idx]

def top1_rmsd_loss(
    lddt_logits: torch.Tensor,
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    ulr_mask: torch.Tensor,
    eps: float = 1e-10,
    hu_mode="bb",
    use_non_ulr=False,
    **kwargs,
):
    with torch.no_grad():
        true_rmsd = rmsd_loss(
            all_atom_pred_pos,
            all_atom_positions,
            all_atom_mask,
            ulr_mask,
            use_non_ulr=use_non_ulr,
            **kwargs,
        )  # B,S
        lddt_logits = torch.nn.functional.softmax(lddt_logits, dim=-1)  # B,S,L,50
        lddt_logits = lddt_logits * torch.arange(50)[None, None, :].to(
            device=lddt_logits.device
        )
        lddt_logits = lddt_logits.sum(dim=-1)  # B,S,L
        lddt_logits = lddt_logits * ulr_mask  # B,S,L * B,S,L -> B,S,L
        lddt_logits = lddt_logits.sum(dim=-1)  # B,S
        sel_idx = torch.argmax(lddt_logits, dim=-1)  # B
        sel_idx = (torch.arange(sel_idx.shape[0]).to(device=sel_idx.device), sel_idx)
    return true_rmsd[sel_idx]


def distogram_loss(
    logits,
    pseudo_beta,
    pseudo_beta_mask,
    ulr_mask=None,
    min_bin=2.3125,
    max_bin=21.6875,
    no_bins=64,
    eps=1e-6,
    use_non_ulr=False,
    **kwargs,
):
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        no_bins - 1,
        device=logits.device,
    )
    boundaries = boundaries**2

    dists = torch.sum(
        (pseudo_beta[..., None, :] - pseudo_beta[..., None, :, :]) ** 2,
        dim=-1,
        keepdims=True,
    )

    true_bins = torch.sum(dists > boundaries, dim=-1)

    errors = softmax_cross_entropy(
        logits,
        torch.nn.functional.one_hot(true_bins, no_bins),
    )  # [N_seed, L, L]
    # ulr_mask [N_seed, L]
    # N_seed, 1, L, + N_seed, L, 1 -> N_seed, L, L?
    ulr_involve_mask = ulr_mask[..., None, :] + ulr_mask[..., None]
    if use_non_ulr:
        assert ulr_involve_mask.dtype == torch.bool
        ulr_not_involve_mask = ~ulr_involve_mask
        ulr_not_involve_mask = ulr_not_involve_mask.long()

    ulr_involve_mask = ulr_involve_mask.long()
    square_mask = pseudo_beta_mask[..., None] * pseudo_beta_mask[..., None, :]

    if use_non_ulr:
        square_mask_involved = square_mask * ulr_involve_mask  # [N_seed, L, L]
        square_mask_not_involved = square_mask * ulr_not_involve_mask  # [N_seed, L, L]
        # square_mask = 2 * square_mask_involved + square_mask_not_involved
        # 2 is hyper-parameter
        # FP16-friendly sum. Equivalent to:
        # mean = (torch.sum(errors * square_mask, dim=(-1, -2)) /
        #         (eps + torch.sum(square_mask, dim=(-1, -2))))
        denom_involved = eps + torch.sum(square_mask_involved, dim=(-1, -2))
        denome_not_involved = eps + torch.sum(square_mask_not_involved, dim=(-1, -2))
        mean_involved = errors * square_mask_involved  # [N_seed, L, L]
        mean_not_involved = errors * square_mask_not_involved
        mean_involved = torch.sum(mean_involved, -1)  # [N_seed, L]
        mean_not_involved = torch.sum(mean_not_involved, -1)
        mean_involved = mean_involved / denom_involved[..., None]  # [N_seed, L]
        mean_not_involved = mean_not_involved / denome_not_involved[..., None]
        mean_involved = torch.sum(mean_involved, dim=-1)  # [N_seed]
        mean_not_involved = torch.sum(mean_not_involved, dim=-1)
        mean = mean_involved + mean_not_involved

    else:
        square_mask = square_mask * ulr_involve_mask
        denom = eps + torch.sum(square_mask, dim=(-1, -2))
        mean = errors * square_mask
        mean = torch.sum(mean, dim=-1)
        mean = mean / denom[..., None]
        mean = torch.sum(mean, dim=-1)
    # Average over the batch dimensions
    # mean = torch.mean(mean)

    return mean


def _calculate_bin_centers(boundaries: torch.Tensor):
    step = boundaries[1] - boundaries[0]
    bin_centers = boundaries + step / 2
    bin_centers = torch.cat(
        [bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], dim=0
    )
    return bin_centers


def _calculate_expected_aligned_error(
    alignment_confidence_breaks: torch.Tensor,
    aligned_distance_error_probs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bin_centers = _calculate_bin_centers(alignment_confidence_breaks)
    return (
        torch.sum(aligned_distance_error_probs * bin_centers, dim=-1),
        bin_centers[-1],
    )


def compute_predicted_aligned_error(
    logits: torch.Tensor,
    max_bin: int = 31,
    no_bins: int = 64,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Computes aligned confidence metrics from logits.

    Args:
      logits: [*, num_res, num_res, num_bins] the logits output from
        PredictedAlignedErrorHead.
      max_bin: Maximum bin value
      no_bins: Number of bins
    Returns:
      aligned_confidence_probs: [*, num_res, num_res, num_bins] the predicted
        aligned error probabilities over bins for each residue pair.
      predicted_aligned_error: [*, num_res, num_res] the expected aligned distance
        error for each pair of residues.
      max_predicted_aligned_error: [*] the maximum predicted error possible.
    """
    boundaries = torch.linspace(0, max_bin, steps=(no_bins - 1), device=logits.device)

    aligned_confidence_probs = torch.nn.functional.softmax(logits, dim=-1)
    (
        predicted_aligned_error,
        max_predicted_aligned_error,
    ) = _calculate_expected_aligned_error(
        alignment_confidence_breaks=boundaries,
        aligned_distance_error_probs=aligned_confidence_probs,
    )

    return {
        "aligned_confidence_probs": aligned_confidence_probs,
        "predicted_aligned_error": predicted_aligned_error,
        "max_predicted_aligned_error": max_predicted_aligned_error,
    }


def compute_tm(
    logits: torch.Tensor,
    residue_weights: Optional[torch.Tensor] = None,
    max_bin: int = 31,
    no_bins: int = 64,
    eps: float = 1e-8,
    **kwargs,
) -> torch.Tensor:
    if residue_weights is None:
        residue_weights = logits.new_ones(logits.shape[-2])

    boundaries = torch.linspace(0, max_bin, steps=(no_bins - 1), device=logits.device)

    bin_centers = _calculate_bin_centers(boundaries)
    clipped_n = max(torch.sum(residue_weights), 19)

    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3) - 1.8

    probs = torch.nn.functional.softmax(logits, dim=-1)

    tm_per_bin = 1.0 / (1 + (bin_centers**2) / (d0**2))
    predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)

    normed_residue_mask = residue_weights / (eps + residue_weights.sum())
    per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)

    weighted = per_alignment * residue_weights

    argmax = (weighted == torch.max(weighted)).nonzero()[0]
    return per_alignment[tuple(argmax)]


def tm_loss(
    logits,
    final_affine_tensor,
    backbone_rigid_tensor,
    backbone_rigid_mask,
    resolution,
    max_bin=31,
    no_bins=64,
    min_resolution: float = 0.1,
    max_resolution: float = 3.0,
    eps=1e-8,
    **kwargs,
):
    pred_affine = Rigid.from_tensor_7(final_affine_tensor)
    backbone_rigid = Rigid.from_tensor_4x4(backbone_rigid_tensor)

    def _points(affine):
        pts = affine.get_trans()[..., None, :, :]
        return affine.invert()[..., None].apply(pts)

    sq_diff = torch.sum((_points(pred_affine) - _points(backbone_rigid)) ** 2, dim=-1)

    sq_diff = sq_diff.detach()

    boundaries = torch.linspace(0, max_bin, steps=(no_bins - 1), device=logits.device)
    boundaries = boundaries**2
    true_bins = torch.sum(sq_diff[..., None] > boundaries, dim=-1)

    errors = softmax_cross_entropy(
        logits, torch.nn.functional.one_hot(true_bins, no_bins)
    )

    square_mask = backbone_rigid_mask[..., None] * backbone_rigid_mask[..., None, :]

    loss = torch.sum(errors * square_mask, dim=-1)
    scale = 0.5  # hack to help FP16 training along
    denom = eps + torch.sum(scale * square_mask, dim=(-1, -2))
    loss = loss / denom[..., None]
    loss = torch.sum(loss, dim=-1)
    loss = loss * scale

    loss = loss * ((resolution >= min_resolution) & (resolution <= max_resolution))

    # Average over the loss dimension
    loss = torch.mean(loss)

    return loss


def between_residue_bond_loss(
    pred_atom_positions: torch.Tensor,  # (*, N, 37/14, 3)
    pred_atom_mask: torch.Tensor,  # (*, N, 37/14)
    residue_index: torch.Tensor,  # (*, N)
    aatype: torch.Tensor,  # (*, N)
    tolerance_factor_soft=12.0,
    tolerance_factor_hard=12.0,
    eps=1e-6,
) -> Dict[str, torch.Tensor]:
    """Flat-bottom loss to penalize structural violations between residues.

    This is a loss penalizing any violation of the geometry around the peptide
    bond between consecutive amino acids. This loss corresponds to
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 44, 45.

    Args:
      pred_atom_positions: Atom positions in atom37/14 representation
      pred_atom_mask: Atom mask in atom37/14 representation
      residue_index: Residue index for given amino acid, this is assumed to be
        monotonically increasing.
      aatype: Amino acid type of given residue
      tolerance_factor_soft: soft tolerance factor measured in standard deviations
        of pdb distributions
      tolerance_factor_hard: hard tolerance factor measured in standard deviations
        of pdb distributions

    Returns:
      Dict containing:
        * 'c_n_loss_mean': Loss for peptide bond length violations
        * 'ca_c_n_loss_mean': Loss for violations of bond angle around C spanned
            by CA, C, N
        * 'c_n_ca_loss_mean': Loss for violations of bond angle around N spanned
            by C, N, CA
        * 'per_residue_loss_sum': sum of all losses for each residue
        * 'per_residue_violation_mask': mask denoting all residues with violation
            present.
    """
    # Get the positions of the relevant backbone atoms.
    this_ca_pos = pred_atom_positions[..., :-1, 1, :]
    this_ca_mask = pred_atom_mask[..., :-1, 1]
    this_c_pos = pred_atom_positions[..., :-1, 2, :]
    this_c_mask = pred_atom_mask[..., :-1, 2]
    next_n_pos = pred_atom_positions[..., 1:, 0, :]
    next_n_mask = pred_atom_mask[..., 1:, 0]
    next_ca_pos = pred_atom_positions[..., 1:, 1, :]
    next_ca_mask = pred_atom_mask[..., 1:, 1]
    has_no_gap_mask = (residue_index[..., 1:] - residue_index[..., :-1]) == 1.0

    # Compute loss for the C--N bond.
    c_n_bond_length = torch.sqrt(
        eps + torch.sum((this_c_pos - next_n_pos) ** 2, dim=-1)
    )

    # The C-N bond to proline has slightly different length because of the ring.
    next_is_proline = aatype[..., 1:] == residue_constants.resname_to_idx["PRO"]
    gt_length = (~next_is_proline) * residue_constants.between_res_bond_length_c_n[
        0
    ] + next_is_proline * residue_constants.between_res_bond_length_c_n[1]
    gt_stddev = (
        ~next_is_proline
    ) * residue_constants.between_res_bond_length_stddev_c_n[
        0
    ] + next_is_proline * residue_constants.between_res_bond_length_stddev_c_n[
        1
    ]
    c_n_bond_length_error = torch.sqrt(eps + (c_n_bond_length - gt_length) ** 2)
    c_n_loss_per_residue = torch.nn.functional.relu(
        c_n_bond_length_error - tolerance_factor_soft * gt_stddev
    )
    mask = this_c_mask * next_n_mask * has_no_gap_mask
    c_n_loss = torch.sum(mask * c_n_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    c_n_violation_mask = mask * (
        c_n_bond_length_error > (tolerance_factor_hard * gt_stddev)
    )

    # Compute loss for the angles.
    ca_c_bond_length = torch.sqrt(
        eps + torch.sum((this_ca_pos - this_c_pos) ** 2, dim=-1)
    )
    n_ca_bond_length = torch.sqrt(
        eps + torch.sum((next_n_pos - next_ca_pos) ** 2, dim=-1)
    )

    c_ca_unit_vec = (this_ca_pos - this_c_pos) / ca_c_bond_length[..., None]
    c_n_unit_vec = (next_n_pos - this_c_pos) / c_n_bond_length[..., None]
    n_ca_unit_vec = (next_ca_pos - next_n_pos) / n_ca_bond_length[..., None]

    ca_c_n_cos_angle = torch.sum(c_ca_unit_vec * c_n_unit_vec, dim=-1)
    gt_angle = residue_constants.between_res_cos_angles_ca_c_n[0]
    gt_stddev = residue_constants.between_res_bond_length_stddev_c_n[0]
    ca_c_n_cos_angle_error = torch.sqrt(eps + (ca_c_n_cos_angle - gt_angle) ** 2)
    ca_c_n_loss_per_residue = torch.nn.functional.relu(
        ca_c_n_cos_angle_error - tolerance_factor_soft * gt_stddev
    )
    mask = this_ca_mask * this_c_mask * next_n_mask * has_no_gap_mask
    ca_c_n_loss = torch.sum(mask * ca_c_n_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    ca_c_n_violation_mask = mask * (
        ca_c_n_cos_angle_error > (tolerance_factor_hard * gt_stddev)
    )

    c_n_ca_cos_angle = torch.sum((-c_n_unit_vec) * n_ca_unit_vec, dim=-1)
    gt_angle = residue_constants.between_res_cos_angles_c_n_ca[0]
    gt_stddev = residue_constants.between_res_cos_angles_c_n_ca[1]
    c_n_ca_cos_angle_error = torch.sqrt(eps + torch.square(c_n_ca_cos_angle - gt_angle))
    c_n_ca_loss_per_residue = torch.nn.functional.relu(
        c_n_ca_cos_angle_error - tolerance_factor_soft * gt_stddev
    )
    mask = this_c_mask * next_n_mask * next_ca_mask * has_no_gap_mask
    c_n_ca_loss = torch.sum(mask * c_n_ca_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    c_n_ca_violation_mask = mask * (
        c_n_ca_cos_angle_error > (tolerance_factor_hard * gt_stddev)
    )

    # Compute a per residue loss (equally distribute the loss to both
    # neighbouring residues).
    per_residue_loss_sum = (
        c_n_loss_per_residue + ca_c_n_loss_per_residue + c_n_ca_loss_per_residue
    )
    per_residue_loss_sum = 0.5 * (
        torch.nn.functional.pad(per_residue_loss_sum, (0, 1))
        + torch.nn.functional.pad(per_residue_loss_sum, (1, 0))
    )

    # Compute hard violations.
    violation_mask = torch.max(
        torch.stack(
            [c_n_violation_mask, ca_c_n_violation_mask, c_n_ca_violation_mask],
            dim=-2,
        ),
        dim=-2,
    )[0]
    violation_mask = torch.maximum(
        torch.nn.functional.pad(violation_mask, (0, 1)),
        torch.nn.functional.pad(violation_mask, (1, 0)),
    )

    return {
        "c_n_loss_mean": c_n_loss,
        "ca_c_n_loss_mean": ca_c_n_loss,
        "c_n_ca_loss_mean": c_n_ca_loss,
        "per_residue_loss_sum": per_residue_loss_sum,
        "per_residue_violation_mask": violation_mask,
    }


def between_residue_clash_loss(
    atom14_pred_positions: torch.Tensor,
    atom14_atom_exists: torch.Tensor,
    atom14_atom_radius: torch.Tensor,
    residue_index: torch.Tensor,
    overlap_tolerance_soft=1.5,
    overlap_tolerance_hard=1.5,
    eps=1e-10,
) -> Dict[str, torch.Tensor]:
    """Loss to penalize steric clashes between residues.

    This is a loss penalizing any steric clashes due to non bonded atoms in
    different peptides coming too close. This loss corresponds to the part with
    different residues of
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.

    Args:
      atom14_pred_positions: Predicted positions of atoms in
        global prediction frame
      atom14_atom_exists: Mask denoting whether atom at positions exists for given
        amino acid type
      atom14_atom_radius: Van der Waals radius for each atom.
      residue_index: Residue index for given amino acid.
      overlap_tolerance_soft: Soft tolerance factor.
      overlap_tolerance_hard: Hard tolerance factor.

    Returns:
      Dict containing:
        * 'mean_loss': average clash loss
        * 'per_atom_loss_sum': sum of all clash losses per atom, shape (N, 14)
        * 'per_atom_clash_mask': mask whether atom clashes with any other atom
            shape (N, 14)
    """
    fp_type = atom14_pred_positions.dtype

    # Create the distance matrix.
    # (N, N, 14, 14)
    dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_pred_positions[..., :, None, :, None, :]
                - atom14_pred_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    # Create the mask for valid distances.
    # shape (N, N, 14, 14)
    dists_mask = (
        atom14_atom_exists[..., :, None, :, None]
        * atom14_atom_exists[..., None, :, None, :]
    ).type(fp_type)

    # Mask out all the duplicate entries in the lower triangular matrix.
    # Also mask out the diagonal (atom-pairs from the same residue) -- these atoms
    # are handled separately.
    dists_mask = dists_mask * (
        residue_index[..., :, None, None, None]
        < residue_index[..., None, :, None, None]
    )

    # Backbone C--N bond between subsequent residues is no clash.
    residue_index = residue_index.long()
    c_one_hot = torch.nn.functional.one_hot(residue_index.new_tensor(2), num_classes=14)
    c_one_hot = c_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])), *c_one_hot.shape
    )
    c_one_hot = c_one_hot.type(fp_type)
    n_one_hot = torch.nn.functional.one_hot(residue_index.new_tensor(0), num_classes=14)
    n_one_hot = n_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])), *n_one_hot.shape
    )
    n_one_hot = n_one_hot.type(fp_type)

    neighbour_mask = (residue_index[..., :, None, None, None] + 1) == residue_index[
        ..., None, :, None, None
    ]
    c_n_bonds = (
        neighbour_mask
        * c_one_hot[..., None, None, :, None]
        * n_one_hot[..., None, None, None, :]
    )
    dists_mask = dists_mask * (1.0 - c_n_bonds)

    # Disulfide bridge between two cysteines is no clash.
    cys = residue_constants.restype_name_to_atom14_names["CYS"]
    cys_sg_idx = cys.index("SG")
    cys_sg_idx = residue_index.new_tensor(cys_sg_idx)
    cys_sg_idx = cys_sg_idx.reshape(*((1,) * len(residue_index.shape[:-1])), 1).squeeze(
        -1
    )
    cys_sg_one_hot = torch.nn.functional.one_hot(cys_sg_idx, num_classes=14)
    disulfide_bonds = (
        cys_sg_one_hot[..., None, None, :, None]
        * cys_sg_one_hot[..., None, None, None, :]
    )
    dists_mask = dists_mask * (1.0 - disulfide_bonds)

    # Compute the lower bound for the allowed distances.
    # shape (N, N, 14, 14)
    dists_lower_bound = dists_mask * (
        atom14_atom_radius[..., :, None, :, None]
        + atom14_atom_radius[..., None, :, None, :]
    )

    # Compute the error.
    # shape (N, N, 14, 14)
    dists_to_low_error = dists_mask * torch.nn.functional.relu(
        dists_lower_bound - overlap_tolerance_soft - dists
    )

    # Compute the mean loss.
    # shape ()
    mean_loss = torch.sum(dists_to_low_error) / (1e-6 + torch.sum(dists_mask))

    # Compute the per atom loss sum.
    # shape (N, 14)
    per_atom_loss_sum = torch.sum(dists_to_low_error, dim=(-4, -2)) + torch.sum(
        dists_to_low_error, axis=(-3, -1)
    )

    # Compute the hard clash mask.
    # shape (N, N, 14, 14)
    clash_mask = dists_mask * (dists < (dists_lower_bound - overlap_tolerance_hard))

    # Compute the per atom clash.
    # shape (N, 14)
    per_atom_clash_mask = torch.maximum(
        torch.amax(clash_mask, axis=(-4, -2)),
        torch.amax(clash_mask, axis=(-3, -1)),
    )

    return {
        "mean_loss": mean_loss,  # shape ()
        "per_atom_loss_sum": per_atom_loss_sum,  # shape (N, 14)
        "per_atom_clash_mask": per_atom_clash_mask,  # shape (N, 14)
    }


def within_residue_violations(
    atom14_pred_positions: torch.Tensor,
    atom14_atom_exists: torch.Tensor,
    atom14_dists_lower_bound: torch.Tensor,
    atom14_dists_upper_bound: torch.Tensor,
    tighten_bounds_for_loss=0.0,
    eps=1e-10,
) -> Dict[str, torch.Tensor]:
    """Loss to penalize steric clashes within residues.

    This is a loss penalizing any steric violations or clashes of non-bonded atoms
    in a given peptide. This loss corresponds to the part with
    the same residues of
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.

    Args:
        atom14_pred_positions ([*, N, 14, 3]):
            Predicted positions of atoms in global prediction frame.
        atom14_atom_exists ([*, N, 14]):
            Mask denoting whether atom at positions exists for given
            amino acid type
        atom14_dists_lower_bound ([*, N, 14]):
            Lower bound on allowed distances.
        atom14_dists_upper_bound ([*, N, 14]):
            Upper bound on allowed distances
        tighten_bounds_for_loss ([*, N]):
            Extra factor to tighten loss

    Returns:
      Dict containing:
        * 'per_atom_loss_sum' ([*, N, 14]):
              sum of all clash losses per atom, shape
        * 'per_atom_clash_mask' ([*, N, 14]):
              mask whether atom clashes with any other atom shape
    """
    # Compute the mask for each residue.
    dists_masks = 1.0 - torch.eye(14, device=atom14_atom_exists.device)[None]
    dists_masks = dists_masks.reshape(
        *((1,) * len(atom14_atom_exists.shape[:-2])), *dists_masks.shape
    )
    dists_masks = (
        atom14_atom_exists[..., :, :, None]
        * atom14_atom_exists[..., :, None, :]
        * dists_masks
    )

    # Distance matrix
    dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_pred_positions[..., :, :, None, :]
                - atom14_pred_positions[..., :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    # Compute the loss.
    dists_to_low_error = torch.nn.functional.relu(
        atom14_dists_lower_bound + tighten_bounds_for_loss - dists
    )
    dists_to_high_error = torch.nn.functional.relu(
        dists - (atom14_dists_upper_bound - tighten_bounds_for_loss)
    )
    loss = dists_masks * (dists_to_low_error + dists_to_high_error)

    # Compute the per atom loss sum.
    per_atom_loss_sum = torch.sum(loss, dim=-2) + torch.sum(loss, dim=-1)

    # Compute the violations mask.
    violations = dists_masks * (
        (dists < atom14_dists_lower_bound) | (dists > atom14_dists_upper_bound)
    )

    # Compute the per atom violations.
    per_atom_violations = torch.maximum(
        torch.max(violations, dim=-2)[0], torch.max(violations, axis=-1)[0]
    )

    return {
        "per_atom_loss_sum": per_atom_loss_sum,
        "per_atom_violations": per_atom_violations,
    }


def find_structural_violations(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,
    violation_tolerance_factor: float,
    clash_overlap_tolerance: float,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Computes several checks for structural violations."""

    # Compute between residue backbone violations of bonds and angles.
    connection_violations = between_residue_bond_loss(
        pred_atom_positions=atom14_pred_positions,
        pred_atom_mask=batch["atom14_atom_exists"],
        residue_index=batch["residue_index"],
        aatype=batch["aatype"],
        tolerance_factor_soft=violation_tolerance_factor,
        tolerance_factor_hard=violation_tolerance_factor,
    )

    # Compute the Van der Waals radius for every atom
    # (the first letter of the atom name is the element type).
    # Shape: (N, 14).
    atomtype_radius = [
        residue_constants.van_der_waals_radius[name[0]]
        for name in residue_constants.atom_types
    ]
    atomtype_radius = atom14_pred_positions.new_tensor(atomtype_radius)
    atom14_atom_radius = (
        batch["atom14_atom_exists"] * atomtype_radius[batch["residx_atom14_to_atom37"]]
    )

    # Compute the between residue clash loss.
    between_residue_clashes = between_residue_clash_loss(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=batch["atom14_atom_exists"],
        atom14_atom_radius=atom14_atom_radius,
        residue_index=batch["residue_index"],
        overlap_tolerance_soft=clash_overlap_tolerance,
        overlap_tolerance_hard=clash_overlap_tolerance,
    )

    # Compute all within-residue violations (clashes,
    # bond length and angle violations).
    restype_atom14_bounds = residue_constants.make_atom14_dists_bounds(
        overlap_tolerance=clash_overlap_tolerance,
        bond_length_tolerance_factor=violation_tolerance_factor,
    )
    atom14_atom_exists = batch["atom14_atom_exists"]
    batch["aatype"] = batch["aatype"].long()
    atom14_dists_lower_bound = atom14_pred_positions.new_tensor(
        restype_atom14_bounds["lower_bound"]
    )[batch["aatype"]]
    atom14_dists_upper_bound = atom14_pred_positions.new_tensor(
        restype_atom14_bounds["upper_bound"]
    )[batch["aatype"]]
    residue_violations = within_residue_violations(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=batch["atom14_atom_exists"],
        atom14_dists_lower_bound=atom14_dists_lower_bound,
        atom14_dists_upper_bound=atom14_dists_upper_bound,
        tighten_bounds_for_loss=0.0,
    )

    # Combine them to a single per-residue violation mask (used later for LDDT).
    per_residue_violations_mask = torch.max(
        torch.stack(
            [
                connection_violations["per_residue_violation_mask"],
                torch.max(between_residue_clashes["per_atom_clash_mask"], dim=-1)[0],
                torch.max(residue_violations["per_atom_violations"], dim=-1)[0],
            ],
            dim=-1,
        ),
        dim=-1,
    )[0]

    return {
        "between_residues": {
            "bonds_c_n_loss_mean": connection_violations["c_n_loss_mean"],  # ()
            "angles_ca_c_n_loss_mean": connection_violations["ca_c_n_loss_mean"],  # ()
            "angles_c_n_ca_loss_mean": connection_violations["c_n_ca_loss_mean"],  # ()
            "connections_per_residue_loss_sum": connection_violations[
                "per_residue_loss_sum"
            ],  # (N)
            "connections_per_residue_violation_mask": connection_violations[
                "per_residue_violation_mask"
            ],  # (N)
            "clashes_mean_loss": between_residue_clashes["mean_loss"],  # ()
            "clashes_per_atom_loss_sum": between_residue_clashes[
                "per_atom_loss_sum"
            ],  # (N, 14)
            "clashes_per_atom_clash_mask": between_residue_clashes[
                "per_atom_clash_mask"
            ],  # (N, 14)
        },
        "within_residues": {
            "per_atom_loss_sum": residue_violations["per_atom_loss_sum"],  # (N, 14)
            "per_atom_violations": residue_violations[
                "per_atom_violations"
            ],  # (N, 14),
        },
        "total_per_residue_violations_mask": per_residue_violations_mask,  # (N)
    }


def find_structural_violations_np(
    batch: Dict[str, np.ndarray],
    atom14_pred_positions: np.ndarray,
    config: ml_collections.ConfigDict,
) -> Dict[str, np.ndarray]:
    to_tensor = lambda x: torch.tensor(x)
    batch = tree_map(to_tensor, batch, np.ndarray)
    atom14_pred_positions = to_tensor(atom14_pred_positions)

    out = find_structural_violations(batch, atom14_pred_positions, **config)

    to_np = lambda x: np.array(x)
    np_out = tensor_tree_map(to_np, out)

    return np_out


def extreme_ca_ca_distance_violations(
    pred_atom_positions: torch.Tensor,  # (N, 37(14), 3)
    pred_atom_mask: torch.Tensor,  # (N, 37(14))
    residue_index: torch.Tensor,  # (N)
    max_angstrom_tolerance=1.5,
    eps=1e-6,
) -> torch.Tensor:
    """Counts residues whose Ca is a large distance from its neighbour.

    Measures the fraction of CA-CA pairs between consecutive amino acids that are
    more than 'max_angstrom_tolerance' apart.

    Args:
      pred_atom_positions: Atom positions in atom37/14 representation
      pred_atom_mask: Atom mask in atom37/14 representation
      residue_index: Residue index for given amino acid, this is assumed to be
        monotonically increasing.
      max_angstrom_tolerance: Maximum distance allowed to not count as violation.
    Returns:
      Fraction of consecutive CA-CA pairs with violation.
    """
    this_ca_pos = pred_atom_positions[..., :-1, 1, :]
    this_ca_mask = pred_atom_mask[..., :-1, 1]
    next_ca_pos = pred_atom_positions[..., 1:, 1, :]
    next_ca_mask = pred_atom_mask[..., 1:, 1]
    has_no_gap_mask = (residue_index[..., 1:] - residue_index[..., :-1]) == 1.0
    ca_ca_distance = torch.sqrt(
        eps + torch.sum((this_ca_pos - next_ca_pos) ** 2, dim=-1)
    )
    violations = (ca_ca_distance - residue_constants.ca_ca) > max_angstrom_tolerance
    mask = this_ca_mask * next_ca_mask * has_no_gap_mask
    mean = masked_mean(mask, violations, -1)
    return mean


def compute_violation_metrics(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,  # (N, 14, 3)
    violations: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Compute several metrics to assess the structural violations."""
    ret = {}
    extreme_ca_ca_violations = extreme_ca_ca_distance_violations(
        pred_atom_positions=atom14_pred_positions,
        pred_atom_mask=batch["atom14_atom_exists"],
        residue_index=batch["residue_index"],
    )
    ret["violations_extreme_ca_ca_distance"] = extreme_ca_ca_violations
    ret["violations_between_residue_bond"] = masked_mean(
        batch["seq_mask"],
        violations["between_residues"]["connections_per_residue_violation_mask"],
        dim=-1,
    )
    ret["violations_between_residue_clash"] = masked_mean(
        mask=batch["seq_mask"],
        value=torch.max(
            violations["between_residues"]["clashes_per_atom_clash_mask"],
            dim=-1,
        )[0],
        dim=-1,
    )
    ret["violations_within_residue"] = masked_mean(
        mask=batch["seq_mask"],
        value=torch.max(violations["within_residues"]["per_atom_violations"], dim=-1)[
            0
        ],
        dim=-1,
    )
    ret["violations_per_residue"] = masked_mean(
        mask=batch["seq_mask"],
        value=violations["total_per_residue_violations_mask"],
        dim=-1,
    )
    return ret


def compute_violation_metrics_np(
    batch: Dict[str, np.ndarray],
    atom14_pred_positions: np.ndarray,
    violations: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    to_tensor = lambda x: torch.tensor(x)
    batch = tree_map(to_tensor, batch, np.ndarray)
    atom14_pred_positions = to_tensor(atom14_pred_positions)
    violations = tree_map(to_tensor, violations, np.ndarray)

    out = compute_violation_metrics(batch, atom14_pred_positions, violations)

    to_np = lambda x: np.array(x)
    return tree_map(to_np, out, torch.Tensor)


def violation_loss(
    violations: Dict[str, torch.Tensor],
    atom14_atom_exists: torch.Tensor,
    eps=1e-6,
    **kwargs,
) -> torch.Tensor:
    num_atoms = torch.sum(atom14_atom_exists)
    l_clash = torch.sum(
        violations["between_residues"]["clashes_per_atom_loss_sum"]
        + violations["within_residues"]["per_atom_loss_sum"]
    )
    l_clash = l_clash / (eps + num_atoms)
    loss = (
        violations["between_residues"]["bonds_c_n_loss_mean"]
        + violations["between_residues"]["angles_ca_c_n_loss_mean"]
        + violations["between_residues"]["angles_c_n_ca_loss_mean"]
        + l_clash
    )
    ## HU Changed ##
    loss = loss.mean(dim=-1)  # B
    return loss


def compute_renamed_ground_truth(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,
    eps=1e-10,
) -> Dict[str, torch.Tensor]:
    """
    Find optimal renaming of ground truth based on the predicted positions.

    Alg. 26 "renameSymmetricGroundTruthAtoms"

    This renamed ground truth is then used for all losses,
    such that each loss moves the atoms in the same direction.

    Args:
      batch: Dictionary containing:
        * atom14_gt_positions: Ground truth positions.
        * atom14_alt_gt_positions: Ground truth positions with renaming swaps.
        * atom14_atom_is_ambiguous: 1.0 for atoms that are affected by
            renaming swaps.
        * atom14_gt_exists: Mask for which atoms exist in ground truth.
        * atom14_alt_gt_exists: Mask for which atoms exist in ground truth
            after renaming.
        * atom14_atom_exists: Mask for whether each atom is part of the given
            amino acid type.
      atom14_pred_positions: Array of atom positions in global frame with shape
    Returns:
      Dictionary containing:
        alt_naming_is_better: Array with 1.0 where alternative swap is better.
        renamed_atom14_gt_positions: Array of optimal ground truth positions
          after renaming swaps are performed.
        renamed_atom14_gt_exists: Mask after renaming swap is performed.
    """

    pred_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_pred_positions[..., None, :, None, :]
                - atom14_pred_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    atom14_gt_positions = batch["atom14_gt_positions"]
    gt_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_gt_positions[..., None, :, None, :]
                - atom14_gt_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    atom14_alt_gt_positions = batch["atom14_alt_gt_positions"]
    alt_gt_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_alt_gt_positions[..., None, :, None, :]
                - atom14_alt_gt_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    lddt = torch.sqrt(eps + (pred_dists - gt_dists) ** 2)
    alt_lddt = torch.sqrt(eps + (pred_dists - alt_gt_dists) ** 2)

    atom14_gt_exists = batch["atom14_gt_exists"]
    atom14_atom_is_ambiguous = batch["atom14_atom_is_ambiguous"]
    mask = (
        atom14_gt_exists[..., None, :, None]
        * atom14_atom_is_ambiguous[..., None, :, None]
        * atom14_gt_exists[..., None, :, None, :]
        * (1.0 - atom14_atom_is_ambiguous[..., None, :, None, :])
    )

    per_res_lddt = torch.sum(mask * lddt, dim=(-1, -2, -3))
    alt_per_res_lddt = torch.sum(mask * alt_lddt, dim=(-1, -2, -3))

    fp_type = atom14_pred_positions.dtype
    alt_naming_is_better = (alt_per_res_lddt < per_res_lddt).type(fp_type)

    renamed_atom14_gt_positions = (
        1.0 - alt_naming_is_better[..., None, None]
    ) * atom14_gt_positions + alt_naming_is_better[
        ..., None, None
    ] * atom14_alt_gt_positions

    renamed_atom14_gt_mask = (
        1.0 - alt_naming_is_better[..., None]
    ) * atom14_gt_exists + alt_naming_is_better[..., None] * batch[
        "atom14_alt_gt_exists"
    ]

    return {
        "alt_naming_is_better": alt_naming_is_better,
        "renamed_atom14_gt_positions": renamed_atom14_gt_positions,
        "renamed_atom14_gt_exists": renamed_atom14_gt_mask,
    }

def crossover_loss(
    cross_over_prev: torch.Tensor,  #[n_recycle, n_block, B, S, L, 3, 3]
    cross_over_after: torch.Tensor, #[n_recycle, n_block, B, S, L, 3, 3]
    all_atom_mask: torch.Tensor, #[B, S, L, 37]
    all_atom_positions: torch.Tensor,
    ulr_mask: torch.Tensor = None,
    eps: float = 1e-10,
    clamp_distance=10.0,
    use_non_ulr=False,
    **kwargs,
) -> torch.Tensor:

    all_atom_positions = all_atom_positions[..., 0:3, :] #[B, S, L, 3, 3]
    all_atom_mask = all_atom_mask[..., 0:3] #[B, S, L, 3]

    if not ulr_mask == None:
        all_atom_mask_ulr = all_atom_mask * ulr_mask[..., None]
        all_atom_mask_ulr = all_atom_mask_ulr.bool()
    
    div_ulr = all_atom_mask_ulr.long().sum(dim=-1).sum(dim=-1)
    # use last recycle, last block
    cross_over_prev = cross_over_prev[-1, -1, ...] #[B, S, L, 3, 3]
    cross_over_after = cross_over_after[-1, -1, ...] #[B, S, L, 3, 3]
    prev_rmsd = torch.sum((cross_over_prev - all_atom_positions)**2, dim=-1)
    prev_rmsd = torch.clamp(prev_rmsd, max=clamp_distance**2)
    prev_rmsd_ulr = prev_rmsd * all_atom_mask_ulr.long()
    prev_rmsd_ulr = prev_rmsd_ulr.sum(dim=-1).sum(dim=-1)
    prev_rmsd_ulr = prev_rmsd_ulr/(div_ulr + eps)
    prev_rmsd_ulr = torch.sqrt(prev_rmsd_ulr + eps)

    after_rmsd = torch.sum((cross_over_after - all_atom_positions)**2, dim=-1)
    after_rmsd = torch.clamp(after_rmsd, max=clamp_distance**2)
    after_rmsd_ulr = after_rmsd * all_atom_mask_ulr.long()
    after_rmsd_ulr = after_rmsd_ulr.sum(dim=-1).sum(dim=-1)
    after_rmsd_ulr = after_rmsd_ulr/(div_ulr + eps)
    after_rmsd_ulr = torch.sqrt(after_rmsd_ulr + eps)
    prev_rmsd_ulr_min = torch.min(prev_rmsd_ulr, dim=-1)[0]
    after_rmsd_ulr_min = torch.min(after_rmsd_ulr, dim=-1)[0]

    rmsd_diff = after_rmsd_ulr_min - prev_rmsd_ulr_min
    return rmsd_diff


def prmsd_loss(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    ulr_mask: torch.Tensor = None,
    eps: float = 1e-10,
    hu_mode="bb",
    clamp_distance=10.0,
    **kwargs,
) -> torch.Tensor:
    # N ,CA, C
    ##Lousy but efficient calculation
    if hu_mode == "bb":
        all_atom_pred_pos = all_atom_pred_pos[..., 0:3, :] #[B, S, L, 3, 3]
        all_atom_positions = all_atom_positions[..., 0:3, :]
    if not ulr_mask == None: #[B, S, L]
        ulr_mask_ori = ulr_mask
        ulr_mask = ulr_mask[..., None].expand(*ulr_mask.shape, 3) #[B, S, L, 3]
    else:
        ulr_mask_ori = torch.ones(all_atom_mask.shape[:-1])
        ulr_mask = all_atom_mask.fill_(1)
    
    tmp = (all_atom_pred_pos * ulr_mask[..., None]).float() 
    dist = tmp[..., None, :, :, :] - tmp[..., None, :, :, :, :] #[B, S, S, L, 3, 3]
    dist = (dist**2).sum(dim=-1) #[B, S, S, L, 3]
    dist = dist * ulr_mask[:, None, ...] * ulr_mask[:, :, None, ...] #[B, S, S, L, 3]
    dist = dist.mean(dim=-1) # mean between N, CA, C
    dist = dist.sum(dim=-1) # sum between L #[B, S, S]
    ulr_mask_sum = ulr_mask_ori.sum(dim = -1)[..., None] #[B, S, 1]
    ulr_mask_sum = ulr_mask_sum.expand(*ulr_mask_sum.shape[:-1], ulr_mask_sum.shape[-2])
    dist = dist / ulr_mask_sum 
    dist = torch.sqrt(dist + eps)
    n_seed = dist.shape[-1]
    n_div = (n_seed ** 2) - n_seed
    rmsd_mean = torch.sum(dist, dim=(1, 2)) / n_div
    return rmsd_mean

class AlphaFoldLoss(nn.Module):
    """Aggregation of the various losses described in the supplement"""

    def __init__(self, config):
        super(AlphaFoldLoss, self).__init__()
        self.config = config
        self.sel_mode = config.sel_mode

    def forward(self, tag, out, batch, _return_breakdown=True):
        if "violation" not in out.keys():
            out["violation"] = find_structural_violations(
                batch,
                out["sm"]["positions"][-1],
                **self.config.violation,
            )

        if "renamed_atom14_gt_positions" not in out.keys():
            batch.update(
                compute_renamed_ground_truth(
                    batch,
                    out["sm"]["positions"][-1],
                )
            )

        loss_fns = {
            "distogram": lambda: distogram_loss(
                logits=out["distogram_logits"],
                **{**batch, **self.config.distogram},
            ),
            "rmsd_loss": lambda: rmsd_loss(
                all_atom_pred_pos=out["final_atom_positions"],
                all_atom_positions=batch["all_atom_positions"],
                all_atom_mask=batch["all_atom_mask"],
                ulr_mask=batch["ulr_mask"],
                **self.config.rmsd_loss,
            ),
            "fape_bb": lambda: fape_bb_loss(
                out,
                batch,
                self.config.fape_bb,
            ),
            "fape_sc": lambda: fape_sc_loss(
                out,
                batch,
                self.config.fape_sc,
            ),
            "plddt_loss": lambda: lddt_loss(
                logits=out["lddt_logits"],
                all_atom_pred_pos=out["final_atom_positions"],
                **{**batch, **self.config.plddt_loss},
            ),
            "supervised_chi": lambda: supervised_chi_loss(
                out["sm"]["angles"],
                out["sm"]["unnormalized_angles"],
                **{**batch, **self.config.supervised_chi},
            ),
            "violation": lambda: violation_loss(
                out["violation"],
                **batch,
            ),
            "best_H3_rmsd": lambda: best_rmsd_loss(
                all_atom_pred_pos=out["final_atom_positions"],
                all_atom_positions=batch["all_atom_positions"],
                all_atom_mask=batch["all_atom_mask"],
                ulr_mask=batch["raw_ulr_mask"],
                **self.config.best_H3_rmsd,
            ),
            "top1_H3_rmsd": lambda: top1_rmsd_loss(
                lddt_logits=out["lddt_logits"],
                all_atom_pred_pos=out["final_atom_positions"],
                all_atom_positions=batch["all_atom_positions"],
                all_atom_mask=batch["all_atom_mask"],
                ulr_mask=batch["raw_ulr_mask"],
                **self.config.top1_H3_rmsd,
            ),
            "best_ulr_rmsd": lambda: best_rmsd_loss(
                all_atom_pred_pos=out["final_atom_positions"],
                all_atom_positions=batch["all_atom_positions"],
                all_atom_mask=batch["all_atom_mask"],
                ulr_mask=batch["ulr_mask"],
                **self.config.best_ulr_rmsd,
            ),
            "top1_ulr_rmsd": lambda: top1_rmsd_loss(
                lddt_logits=out["lddt_logits"],
                all_atom_pred_pos=out["final_atom_positions"],
                all_atom_positions=batch["all_atom_positions"],
                all_atom_mask=batch["all_atom_mask"],
                ulr_mask=batch["ulr_mask"],
                **self.config.top1_ulr_rmsd,
            ),
            "obs_H3_rmsd": lambda: rmsd_loss(
                all_atom_pred_pos=out["final_atom_positions"],
                all_atom_positions=batch["all_atom_positions"],
                all_atom_mask=batch["all_atom_mask"],
                ulr_mask=batch["raw_ulr_mask"],
                **self.config.obs_H3_rmsd,
            ),
            "prmsd_loss":lambda: prmsd_loss(
               all_atom_pred_pos=out["final_atom_positions"],
               all_atom_positions=batch["all_atom_positions"],
               all_atom_mask=batch["all_atom_mask"],
               ulr_mask=batch["ulr_mask"],
               **self.config.prmsd_loss,
            ),
            "crossover_loss": lambda: crossover_loss(
              cross_over_prev = out['cross_over_prev'],
              cross_over_after = out['cross_over_after'],
              all_atom_mask=batch['all_atom_mask'],
              all_atom_positions=batch["all_atom_positions"],
              ulr_mask=batch['ulr_mask'],
              **self.config.crossover_loss)
#            "total_rmsd":lambda: total_rmsd_loss(
#                all_atom_pred_pos=out["final_atom_positions"],
#                all_atom_positions=batch["all_atom_positions"],
#                all_atom_mask=batch["all_atom_mask"],
#                ulr_mask=batch["ulr_mask"],
#                **self.config.total_rmsd,
#            ),

        }
        batch_size = batch["ulr_mask"].shape[0]
        sel_loss = torch.zeros(batch_size, self.config["seed_size"]).to(
            device=out["sm"]["positions"].device
        )
        per_decoy_loss = torch.zeros(batch_size, self.config["seed_size"]).to(
            device=out["sm"]["positions"].device
        )
        cum_loss = 0.0
        losses = {}
        for loss_name, loss_fn in loss_fns.items():
            weight = self.config[loss_name].weight
            loss = loss_fn()
            if torch.isnan(loss.sum()) or torch.isinf(loss.sum()):
                logging.warning(f"{loss_name} loss is NaN. Skipping...")
                new_value = np.zeros(loss.shape)
                loss = loss.new_tensor(new_value, requires_grad=True)
            ###
            if loss_name in self.config.sel_name_s:
                if loss_name in ["fape_bb"]:
                    sel_loss = (
                        sel_loss + weight * loss[-1].detach().clone()
                    )  # jus use for selecton nogradients
                else:
                    sel_loss = (
                        sel_loss + weight * loss.detach().clone()
                    )  # jus use for selecton nogradients
            if loss_name in self.config.per_name_s:
                if loss_name in ["fape_bb"]:
                    loss = torch.mean(loss, dim=0)
                per_decoy_loss = per_decoy_loss + weight * loss
            else:
                if not loss_name in ["obs_H3_rmsd"]:
                    cum_loss = cum_loss + weight * loss
            losses[loss_name] = loss.detach().clone()
        ##
        if self.sel_mode == "min":
            sel_idx = select_loss_from_decoy(sel_loss)
            cum_loss = cum_loss + per_decoy_loss[sel_idx]
            for key in losses.keys():
                if not key in self.config.per_name_s:
                    continue
                losses[key] = losses[key][sel_idx]
        ##
        elif self.sel_mode == 'mean':
            cum_loss = cum_loss + torch.mean(per_decoy_loss, dim=-1)
            for key in losses.keys():
                if not key in self.config.per_name_s:
                    continue
                losses[key] = torch.mean(losses[key], dim = -1)
        losses["unscaled_loss"] = cum_loss.detach().clone()
        crop_len = batch["aatype"].shape[-1]
        cum_loss = cum_loss * (crop_len ** (0.5))
        losses["loss"] = cum_loss.detach().clone()
        cum_loss = torch.mean(cum_loss)
        if not _return_breakdown:
            return cum_loss

        return cum_loss, losses
