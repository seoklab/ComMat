#!/usr/bin/env/python3

from data_utils import _read_pickle_select, initialize_ulr_rigid
import torch
import numpy as np
from h3xsemble.utils.rigid_utils import Rigid, Rotation


def data_preprocess(pdbfile):
    return


def zeropadding_for_collate(fn_dat, n_crop):
    curr_len = fn_dat.shape[0]
    if curr_len == n_crop:
        return fn_dat
    out = fn_dat.new_zeros(n_crop, *fn_dat.shape[1:]).clone()
    out[:curr_len] = fn_dat
    return out


def chain_tagging(
    tag,  # 6fe4_F_#_A
    dat,  # dict with various keys (chain_id, ...)
    mode="ab",
    ag_remove_prob=0.25,
    ulr_type=["H_3"],
    inference_n_residue_extension=0,
    build_all_cdr=False,
):
    n_residue_extension = inference_n_residue_extension
    chain_idx = dat["chain_id"]
    # chain_tag // 0: heavy chain, 1:light chain, 2:antigen
    assert mode == "ab"
    tmp = tag.split("_")[1:]  # F, #, A
    memo = []
    dat["raw_ulr_mask"] = dat["tot_ulr"]["H_3"].clone().detach()
    for idx, x in enumerate(tmp):
        if x == "#":
            continue
        for y in x:
            idx = min(2, idx)
            memo.append(idx)
    if not len(memo) == (torch.max(chain_idx).item() + 1):
        print("some chain tagging failed")
        sys.exit()
    memo = torch.tensor(memo)
    dat["chain_tag"] = memo[chain_idx]  # chain_tag same as chain_idx...?
    ###
    tmp = dat["all_atom_mask"]
    active_cdr_H = []
    active_cdr_L = []
    active_H3 = False
    all_cdr_mask = []
    ##
    for cdr_type in ["H_1", "H_2", "H_3", "L_1", "L_2", "L_3"]:
        if not cdr_type in dat["tot_ulr"]:
            continue
        ulr_mask_cdr = dat["tot_ulr"][cdr_type]
        ulr_range = dat["tot_ulr_range"][cdr_type]
        from_index = ulr_range[0] - n_residue_extension
        to_index = ulr_range[-1] + n_residue_extension
        ulr_mask_cdr[from_index : from_index + n_residue_extension] = True
        ulr_mask_cdr[to_index - n_residue_extension + 1 : to_index + 1] = True
        all_cdr_mask.append(ulr_mask_cdr)

        anchor_mask_cdr = torch.zeros(dat["tot_ulr"][cdr_type].shape)
        if (to_index + 1 >= len(anchor_mask_cdr)) or (from_index + 1 < 0):
            continue
        anchor_mask_cdr[from_index - 1] = 1
        anchor_mask_cdr[to_index + 1] = 1
        anchor_mask_cdr = anchor_mask_cdr.bool()
        ulr_all_missing = dat["all_atom_mask"][ulr_mask_cdr.bool(), 1].sum() == 0
        anchor_missing = dat["miss_mask"][anchor_mask_cdr.bool()].sum() != 2
        if (not ulr_all_missing) and (not anchor_missing):
            if cdr_type == "H_3":
                active_H3 = True
            elif cdr_type in ["H_1", "H_2"]:
                active_cdr_H.append(cdr_type)
            else:
                active_cdr_L.append(cdr_type)
    active_set = ["H_3"] + active_cdr_H + active_cdr_L
    ulr_type = list(set(ulr_type) & set(active_set))
    all_cdr_mask = torch.stack(all_cdr_mask, dim=-2)  # *,6 or3,L
    all_cdr_mask = all_cdr_mask.sum(dim=-2)
    dat["all_cdr_mask"] = all_cdr_mask

    # remove antigen chain for abag complex
    ag_stat = (dat["chain_tag"] == 2).sum()
    if np.random.random() < ag_remove_prob and ag_stat:
        out_dic = {}
        mask = dat["chain_tag"] != 2
        for key in dat.keys():
            if key in ["tot_ulr_range", "ulr_range"]:
                out_dic[key] = dat[key]
                continue
            if key in ["center"]:
                out_dic[key] = dat[key]
                continue
            if key in ["tot_ulr"]:
                out_dic[key] = dat[key]
                for _ulr_type in out_dic[key].keys():
                    out_dic[key][_ulr_type] = out_dic[key][_ulr_type][mask]
                continue
            out_dic[key] = dat[key][mask]
        dat = out_dic
        ag_stat = False

    assert active_H3
    if build_all_cdr:
        print("build_all_cdr!!!!!!!")
        ulr_type = ["H_3"] + active_cdr_H + active_cdr_L
        print(ulr_type)
    else:
        ulr_type = ulr_type

    ##
    (
        dat["inp_gt"],
        dat["center"],
        skip_ulr_type,
        dat["ulr_mask"],
    ) = initialize_ulr_rigid(
        dat,
        ulr_type_s=ulr_type,
        mode="linspace",
        n_residue_extension=n_residue_extension,
    )

    return dat


def preprocess_input(
    dat, mode, seed_size, trans_scale, n_crop, build_from_scratch=False
):
    dat["ulr_mask"] = dat["ulr_mask"].bool()
    dat["kabsch_mask"] = gen_kabsch_mask(dat, mode)
    if build_from_scratch:  # set all residue region to be ulr
        dat["ulr_mask"] = dat["ulr_mask"].fill_(1).bool()
    dat = crop_with_center(dat, mode, n_crop=n_crop)
    ##
    for key in dat.keys():
        if isinstance(dat[key], Rigid):
            tmp_trs = zeropadding_for_collate(dat[key]._trans, n_crop)
            tmp_trs = (
                (tmp_trs / (trans_scale)).expand(seed_size, *tmp_trs.shape).clone()
            )
            if build_from_scratch:  # all translation to zero
                tmp_trs = torch.zeros_like(tmp_trs)
            tmp_rot = zeropadding_for_collate(dat[key]._rots._rot_mats, n_crop)
            tmp_rot = (tmp_rot).expand(seed_size, *tmp_rot.shape).clone()
            if build_from_scratch:  # all rotation to identity
                tmp_rot = torch.zeros_like(tmp_rot)
                tmp_rot[..., 0, 0] = 1
                tmp_rot[..., 1, 1] = 1
                tmp_rot[..., 2, 2] = 1
            dat[key] = Rigid(Rotation(rot_mats=tmp_rot), tmp_trs)
        else:
            dat[key] = zeropadding_for_collate(dat[key], n_crop)
            dat[key] = dat[key].expand(seed_size, *dat[key].shape).clone()
    return dat


def crop_with_center(dic, mode, n_crop=100):
    miss_mask = dic["miss_mask"]
    ulr_mask = dic["ulr_mask"]
    ca_coord_s = dic["inp_gt"]._trans.double()
    n_crop = min(ca_coord_s.shape[0], n_crop)
    ##
    center = dic["center"].double()
    ###
    dist = ca_coord_s - center[None, :]
    dist = torch.norm(dist, dim=-1)
    block_mask = (~miss_mask) & (~ulr_mask)  # missing이면서 ulr도 아닌영역
    ###
    mask = torch.zeros_like(dic["aatype"]).bool()  # whole sequence
    ag_mask = (dic["chain_tag"] == 2).bool()
    ag_stat = ag_mask.sum()
    if mode == "ab":
        # Now, all cdr region must be contained with ab_cropping
        all_cdr_mask = dic["all_cdr_mask"].bool() & (
            ~block_mask
        )  # cdr residues that are not missing
        min_no_ab = all_cdr_mask.sum().item()
        dist[all_cdr_mask] = 0
        dist[block_mask] = 9999
        if ag_stat:
            ### antigen cropping with n/2
            no_ag = (ag_mask).sum()
            no_ag = min(no_ag, int(n_crop / 2), n_crop - min_no_ab)
            tmp_dist = dist.clone().detach()
            tmp_dist[(~ag_mask)] = 9999
            crop_idx = torch.topk(tmp_dist, no_ag, largest=False, sorted=False)[1]
            mask[crop_idx] = True
            ### antibody cropping with n/2
            tmp_dist = dist.clone().detach()
            tmp_dist[(ag_mask)] = 9999
            crop_idx = torch.topk(
                tmp_dist, n_crop - no_ag, largest=False, sorted=False
            )[1]
            mask[crop_idx] = True
        else:
            tmp_dist = dist.clone().detach()
            crop_idx = torch.topk(tmp_dist, int(n_crop), largest=False, sorted=False)[1]
            mask[crop_idx] = True
    elif mode == "general":
        dist[block_mask] = 9999
        tmp_dist = dist.clone().detach()
        crop_idx = torch.topk(tmp_dist, int(n_crop), largest=False, sorted=False)[1]
        mask[crop_idx] = True
    else:
        sys.exit()

    ###
    out_dic = {}
    for key in dic.keys():
        if key in ["center", "tot_ulr", "tot_ulr_range", "ulr_range"]:
            continue
        out_dic[key] = dic[key][mask]
    ###
    return out_dic


def gen_kabsch_mask(dat, mode):
    kabsch_mask = (
        dat["miss_mask"] * dat["all_atom_mask"][..., 1]
    ).bool()  # missing in models & missing in crystal
    if mode == "general":
        return kabsch_mask
    ##
    memo = []
    for x in [-1, 0, 1, 2]:
        mask1 = dat["chain_tag"] == x
        if mask1.sum().item() == 0:
            continue
        tmp = torch.nonzero(mask1)
        mask1[tmp.min() : tmp.min() + 10] = False
        mask1[tmp.max() - 9 : tmp.max() + 1] = False
        memo.append(mask1)
    memo = torch.stack(memo, dim=0).sum(dim=0).bool()
    kabsch_mask = kabsch_mask & memo
    return kabsch_mask.bool()


def collate_batch(input_dic):
    output_dic = {}
    for k, v in input_dic.items():
        output_dic[k] = v.unsqueeze(0)
    return output_dic


def data_preprocess_temp(
    pdbname,
    output_folder,
    ag_remove,
    seed_size,
    trans_scale_factor,
    n_crop,
    build_from_scratch,
):
    fn = f"{output_folder}/{pdbname}.pkl"
    tag = pdbname
    selected_structure = "IgFold"
    select_index = 0
    dat = _read_pickle_select(
        fn, structure_mode=selected_structure, model_index=select_index
    )
    if not ag_remove:
        ag_remove_prob = 0
    else:
        ag_remove_prob = 1
    dat = chain_tagging(
        tag,
        dat,
        mode="ab",
        ag_remove_prob=ag_remove_prob,
        ulr_type=["H_3"],
        inference_n_residue_extension=0,
        build_all_cdr=False,
    )
    new_dat = {}
    for key in dat.keys():
        if key in ["tot_ulr", "tot_ulr_range", "ulr_range"]:
            continue
        new_dat[key] = dat[key]

    mode = "ab"
    with torch.no_grad():
        input_dic = preprocess_input(
            dat=new_dat,
            mode=mode,
            seed_size=seed_size,
            trans_scale=trans_scale_factor,
            n_crop=n_crop,
            build_from_scratch=build_from_scratch,
        )
    input_dic = collate_batch(input_dic)
    return input_dic, tag, mode
