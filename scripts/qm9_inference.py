import sys
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import random, sys, pickle
from openfold.utils.rigid_utils import Rigid, Rotation

from data_utils import *
from data_utils import _read_pickle_select
from data_module import DataModule

MAX_EPOCH = 500
RANDOM_SEED = 7


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
        print("GOD DAMN")
        sys.exit()
    ###
    out_dic = {}
    for key in dic.keys():
        if key in ["center", "tot_ulr", "tot_ulr_range", "ulr_range"]:
            continue
        out_dic[key] = dic[key][mask]
    ###
    return out_dic


def zeropadding_for_collate(fn_dat, n_crop):
    curr_len = fn_dat.shape[0]
    if curr_len == n_crop:
        return fn_dat
    out = fn_dat.new_zeros(n_crop, *fn_dat.shape[1:]).clone()
    out[:curr_len] = fn_dat
    return out

def gen_kabsch_mask(dat, mode):
    kabsch_mask = (
        dat["miss_mask"] * dat["all_atom_mask"][..., 1]
    ).bool()  # missing in models & missing in crystal
    if mode == "general":
        return kabsch_mask
    ##
    memo = []
    for x in [-1, 0, 1,2]:
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


def chain_tagging(
    tag,  # 6fe4_F_#_A
    dat,  # dict with various keys (chain_id, ...)
    mode="ab",
    ag_remove_prob=0.25,
    ulr_type=["H_3"],
    inference_n_residue_extension=0,
    build_all_cdr = False,
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
    active_set = ['H_3'] + active_cdr_H + active_cdr_L
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
        print('build_all_cdr!!!!!!!')
        ulr_type = ['H_3'] + active_cdr_H + active_cdr_L
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


class MyDataset(Dataset):
    def __init__(
        self,
        inp_dat,
        seed_size=32,
        trans_scale_factor=10.0,
        use_gp=False,  # using general protein data
        use_chain_tag=False,  # for developing
        ag_remove_prob=0.25,
        use_pert_crop_center=False,
        pert_crop_center=2.0,  # upper bound of uniform distribution [0,x)
        fix_tag_debug=None,
        fix_structure_mode=None,
        fix_model_index=None,
        build_from_scratch=False,
        ulr_type=["H_3"],
        build_all_cdr = False,
    ):
        self.inp_dat = inp_dat  # clstr_idx
        self.seed_size = seed_size
        self.trans_scale_factor = trans_scale_factor
        self.use_gp = use_gp
        self.ag_remove_prob = ag_remove_prob
        self.use_chain_tag = use_chain_tag
        self.use_pert_crop_center = use_pert_crop_center
        self.pert_crop_center = pert_crop_center  # 2.0
        self.fix_tag_debug = fix_tag_debug
        self.fix_structure_mode = fix_structure_mode
        self.fix_model_index = fix_model_index
        self.build_from_scratch = build_from_scratch
        self.ulr_type = ulr_type
        self.build_all_cdr = build_all_cdr

    def __len__(self):
        return len(self.inp_dat)

    def __getitem__(self, index):
        with torch.no_grad():
            tmp = self.inp_dat[index]
            tag = tmp[0]
            mode = tmp[1]  # ex) tag: 6fe4_F_#_A, mode: ab
            if not self.fix_tag_debug == None:
                tag = self.fix_tag_debug[0]
                mode = self.fix_tag_debug[1]
            fn = f"/store/hu_tmp/merged/{tag}_merged.dat"
            if mode == "ab":
                igFold_probability = 0.2
                pertMD_probability = 0.8
                if random.random() < igFold_probability:
                    selected_structure = "IgFold"
                    select_index = 0
                else:
                    selected_structure = "pertMD"
                    select_index = random.randint(0, 31)
                if not self.fix_structure_mode == None:
                    selected_structure = self.fix_structure_mode
                    if self.fix_structure_mode == "IgFold":
                        select_index = 0
                    elif self.fix_structure_mode == "PertMD":
                        select_index = random.randint(0, 31)
                self.fix_model_index == None
                if not self.fix_model_index == None:
                    select_index = self.fix_model_index
                dat = _read_pickle_select(
                    fn, structure_mode=selected_structure, model_index=select_index
                )
            # if self.use_chain_tag:
            dat = chain_tagging(
                tag,  # protein ID
                dat,  #
                mode=mode,  # ab/gp
                ag_remove_prob=self.ag_remove_prob,  # 0.25
                ulr_type=self.ulr_type,
                build_all_cdr = self.build_all_cdr,
            )

            new_dat = {}
            for key in dat.keys():
                if key in ["tot_ulr", "tot_ulr_range", "ulr_range"]:
                    continue
                new_dat[key] = dat[key]
            if self.use_pert_crop_center:
                vec = torch.rand(3)
                vec = vec / vec.norm(dim=-1)
                new_dat["center"] = new_dat["center"] + self.pert_crop_center * vec
            return new_dat, tag, mode


class TestDataModule(DataModule):
    def __init__(
        self,
        batch_size: int = 3,
        seed_size: int = 8,
        trans_scale_factor: float = 10.0,
        use_gp=False,
        use_chain_tag=False,
        ulr_type=["H_3"],
        ag_remove_prob=0.25,
        use_pert_crop_center=False,
        pert_crop_center=2.0,
        fix_tag_debug=None,
        fix_structure_mode=None,
        fix_model_index=None,
        build_from_scratch=False,
        n_crop: int = 100,
        build_all_cdr = False,
        **kwargs,
    ):
        batch_size = 1  # fixed for least loss/redundancy for benchmark set sample
        super().__init__(
            batch_size=batch_size, num_workers=12, collate_fn=self._collate
        )
        ##
        random.seed(RANDOM_SEED)
        self.seed_size = seed_size
        self.batch_size = batch_size
        self.trans_scale_factor = trans_scale_factor
        self.use_chain_tag = use_chain_tag
        self.ag_remove_prob = ag_remove_prob
        self.use_pert_crop_center = use_pert_crop_center
        self.pert_crop_center = pert_crop_center
        ##
        self.fix_tag_debug = fix_tag_debug
        self.build_from_scratch = build_from_scratch
        self.n_crop = n_crop
        ##
        bench_immune_set, bench_ig_set, bench_test_set = prep_set()
        bench_immune_set = [[x, "ab"] for x in bench_immune_set]
        bench_ig_set = [[x, "ab"] for x in bench_ig_set]
        bench_test_set = [[x, "ab"] for x in bench_test_set]
        ##
        self.use_gp = use_gp
        ##
        self.ds_bench_immune = MyDataset(
            bench_immune_set,
            seed_size=self.seed_size,
            trans_scale_factor=self.trans_scale_factor,
            use_gp=self.use_gp,
            use_chain_tag=True,
            ag_remove_prob=0.0,
            use_pert_crop_center=False,
            pert_crop_center=0.0,
            fix_structure_mode=fix_structure_mode,
            fix_model_index=fix_model_index,
            ulr_type=ulr_type,
            build_all_cdr = build_all_cdr
        )
        self.ds_bench_immune_apo = MyDataset(
            bench_immune_set,
            seed_size=self.seed_size,
            trans_scale_factor=self.trans_scale_factor,
            use_gp=self.use_gp,
            use_chain_tag=True,
            ag_remove_prob=1.0,
            use_pert_crop_center=False,
            pert_crop_center=0.0,
            fix_structure_mode=fix_structure_mode,
            fix_model_index=fix_model_index,
            ulr_type=ulr_type,
            build_all_cdr = build_all_cdr
        )
        self.ds_bench_ig = MyDataset(
            bench_ig_set,
            seed_size=self.seed_size,
            trans_scale_factor=self.trans_scale_factor,
            use_gp=self.use_gp,
            use_chain_tag=True,
            ag_remove_prob=0.0,
            use_pert_crop_center=False,
            pert_crop_center=0.0,
            fix_structure_mode=fix_structure_mode,
            fix_model_index=fix_model_index,
            ulr_type=ulr_type,
            build_all_cdr = build_all_cdr
        )
        self.ds_bench_ig_apo = MyDataset(
            bench_ig_set,
            seed_size=self.seed_size,
            trans_scale_factor=self.trans_scale_factor,
            use_gp=self.use_gp,
            use_chain_tag=True,
            ag_remove_prob=1.0,
            use_pert_crop_center=False,
            pert_crop_center=0.0,
            fix_structure_mode=fix_structure_mode,
            fix_model_index=fix_model_index,
            ulr_type=ulr_type,
            build_all_cdr = build_all_cdr
        )
        self.ds_test = MyDataset(
            bench_test_set,
            seed_size=self.seed_size,
            trans_scale_factor=self.trans_scale_factor,
            use_gp=self.use_gp,
            use_chain_tag=True,
            ag_remove_prob=0.0,
            use_pert_crop_center=False,
            pert_crop_center=0.0,
            fix_structure_mode=fix_structure_mode,
            fix_model_index=fix_model_index,
            ulr_type=ulr_type,
            build_all_cdr = build_all_cdr
        )
        self.ds_test_apo = MyDataset(
            bench_test_set,
            seed_size=self.seed_size,
            trans_scale_factor=self.trans_scale_factor,
            use_gp=self.use_gp,
            use_chain_tag=True,
            ag_remove_prob=1.0,
            use_pert_crop_center=False,
            pert_crop_center=0.0,
            fix_structure_mode=fix_structure_mode,
            fix_model_index=fix_model_index,
            ulr_type=ulr_type,
            build_all_cdr = build_all_cdr
        )
    def prepare_data(self):
        pass

    def _collate(self, samples):
        dic_s = []
        tag_s = []
        mode_s = []
        max_n = self.n_crop
        if self.build_from_scratch:
            tmp_n = []
            for sample in samples:
                dat = sample[0]
                tmp_n.append(dat["aatype"].shape[0])
            max_n = max(tmp_n)
        for sample in samples:
            dat = sample[0]
            tag = sample[1]
            target_mode = sample[2]
            tag_s.append(tag)
            mode_s.append(target_mode)
            with torch.no_grad():
                input_dic = preprocess_input(
                    dat,
                    target_mode,
                    self.seed_size,
                    self.trans_scale_factor,
                    n_crop=max_n,
                    build_from_scratch=self.build_from_scratch,
                )
            dic_s.append(input_dic)
        dat = collate_batch(dic_s)
        return dat, tag_s, mode_s


def gen_tag_list(
    clstr_dic,
    max_epoch=MAX_EPOCH,
    mode="tr",
    get_from_70=True,
):
    # clstr_dic: {key_70: value_70}
    # value_70: {key_100_s: pdb_s}
    # memo = [pdb1, pdb2, ..] #1507
    # note = [memo1, memo2, ..]
    note = []
    if not mode == "tr":
        max_epoch = 1
    for epoch_idx in range(max_epoch):
        memo = []
        if not get_from_70:
            for key_70, value_70 in clstr_dic.items():
                key_100_s = list(value_70.keys())
                random.shuffle(key_100_s)
                if mode == "tr":
                    sel_key_100_idx = epoch_idx % len(key_100_s)
                else:
                    sel_key_100_idx = 0

                sel_key_100 = key_100_s[sel_key_100_idx]
                pdb_s = clstr_dic[key_70][sel_key_100]
                random.shuffle(pdb_s)

                if mode == "tr":
                    sel_pdb_idx = epoch_idx % len(key_100_s)
                else:
                    sel_pdb_idx = 0
                sel_pdb_idx = epoch_idx % len(pdb_s)
                sel_pdb = pdb_s[sel_pdb_idx]
                memo.append(sel_pdb)
        else:
            clstr_70_list = []
            for key_70, value_70 in clstr_dic.items():
                for k, v in value_70.items():
                    clstr_70_list.append(v)
            random.shuffle(clstr_70_list)
            if not mode == "tr":
                for i in clstr_70_list:
                    memo.append(i[0])
            else:
                for clstr_70 in clstr_70_list:
                    random.shuffle(clstr_70)
                    sel_pdb_idx = epoch_idx % len(clstr_70)
                    sel_pdb = clstr_70[sel_pdb_idx]
                    memo.append(sel_pdb)

        note.append(memo)
    return note


def prep_set():
    bench_immune_set = (
        "/home/dngusdnr1/loop_db/immunebuilder_test_set/immunebuilder.test.set"
    )
    bench_ig_set = "/home/dngusdnr1/loop_db/igfold_test_set/igfold.test.set"
    bench_test_set = "/home/yubeen/h3_loop_modeling/DB/new_cluster/cdhit/rfab_valid.pkl"
    return (
        hu_clstr_inp(bench_immune_set, mode="test_immune")[0],
        hu_clstr_inp(bench_ig_set, mode="test_ig")[0],
        hu_clstr_inp(bench_test_set, mode="test")[0],
    )


def hu_clstr_inp(fn, mode="inf"):
    with open(fn, "rb") as fp:
        clstr_dic = pickle.load(fp)
    return gen_tag_list(clstr_dic, mode=mode)
