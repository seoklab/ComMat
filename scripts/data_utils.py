from h3xsemble.utils.rigid_utils import Rigid, Rotation
import torch, pickle,random
#with open('manual_cut_region_for_safe_kabsch
modi_dic={}
modi_dic['3upc_G_#_#']=[ ['backward',-13] ]
modi_dic['6pk8_A_a_#'] =[['case2']]
modi_dic['6pk8_C_c_#'] =[['case2']]
modi_dic['5j74_B_b_#'] =[['case2']]
modi_dic['5aam_B_b_J'] =[['case2']]
modi_dic['6g8r_A_a_B'] =[['case2']]
modi_dic['5b3n_A_a_#'] =[['case2']]
modi_dic['6k42_H_h_AB']=[['case2']]
modi_dic['6ehy_A_a_#'] =[['case2']]
modi_dic['6ehy_B_b_#'] =[['case2']]
modi_dic['6oan_B_b_A'] =[['case2']]
modi_dic['6oan_D_d_C'] =[['case2']]
modi_dic['6xbl_S_s_AB']=[['case2']]
modi_dic['6xbk_S_s_AB']=[['case2']]
modi_dic['6xbj_S_s_AB']=[['case2']]
modi_dic['6os9_D_d_AB']=[['case2']]
modi_dic['6wha_E_e_BC']=[['case2']]
modi_dic['6xbm_S_s_AB']=[['case2']]
modi_dic['6k41_H_h_AB']=[['case2']]
modi_dic['7e33_E_e_AB']=[['case2']]
modi_dic['7cmu_E_e_AB']=[['case2']]
modi_dic['7cmv_E_e_AB']=[['case2']]
modi_dic['7d77_S_s_AB']=[['case2']]
modi_dic['7deo_C_c_D'] =[['case2']]
modi_dic['5yd3_E_e_F'] =[['case2']]
modi_dic['5yd3_C_c_D'] =[['case2']]
modi_dic['5yd4_E_e_F'] =[['case2']]
modi_dic['5yd4_C_c_D'] =[['case2']]
modi_dic['5j75_A_a_#'] =[['case2']]
modi_dic['5j75_B_b_#'] =[['case2']]
modi_dic['5aaw_F_f_G'] =[['case2']]
modi_dic['7kbm_A_a_#'] =[['case1']]
modi_dic['7kbo_A_a_#'] =[['case1']]
modi_dic['7kbp_A_a_#'] =[['case1']]
modi_dic['7kbp_B_b_#'] =[['case1']]
modi_dic['7kbp_C_c_#'] =[['case1']]
modi_dic['7kbp_D_d_#'] =[['case1']]
modi_dic['7cu65_B_b_E']=[['case1']]
modi_dic['7cu65_A_a_Q']=[['case1']]
modi_dic['7ah1_A_a_#'] =[['case1']]
modi_dic['5fcs_H_h_#'] =[['case1']]
modi_dic['5fcs_L_l_#'] =[['case1']]
modi_dic['5fcs_L_l_#'] =[['case1']]
modi_dic['6kn9_E_e_B'] =[['case3']]

def truncate_manually_by_hu(dat,tag,mode):
    if not tag in modi_dic.keys():
        return dat
    for x in modi_dic[tag]:
        if x[0] =='point': 
            mask1=dat['chain_id'] == x[1][0]
            mask2=dat['hu_residue_index']>= x[1][1]
            mask3=dat['hu_residue_index']<=x[1][2]
            mask=mask1&mask2
            mask=mask&mask3
        elif x[0] =='case2':
            mask1=dat['chain_tag']==0
            mask2=dat['all_atom_mask'][...,1].bool() 
            mask=mask1&mask2
            last_true_index = torch.nonzero(mask).max()
            result=torch.zeros_like(mask).bool()
            result[last_true_index]=True
            mask=result
        elif x[0] == 'case1':
            mask1=dat['chain_tag']==1
            mask2=dat['all_atom_mask'][...,1].bool() 
            mask=mask1&mask2
        elif x[0]=='case3':
            mask=dat['all_atom_mask'][...,1].bool()
            mask=~mask
        for key in dat.keys():
            if not (('mask' in key) or ('exists' in key)):
                continue
            if x[0] =='backward':
                dat[key][x[1]:]=0
            elif x[0] =='front':
                dat[key][:x[1]]=0
            elif x[0] in ['point','case2','case1','case3']:
                dat[key][mask]=0
    return dat


def collate_batch(dic_s):
    tmp_dic = dic_s[0]
    out_dic = {}
    for key in tmp_dic.keys():
        out_dic[key] = []
        for dic in dic_s:
            out_dic[key].append(dic[key].unsqueeze(0))
        if isinstance(dic[key], torch.Tensor):
            out_dic[key] = torch.cat(out_dic[key], dim=0)
        elif isinstance(dic[key], Rigid):
            out_dic[key] = dic[key].cat(out_dic[key], dim=0)
        else:
            print("Hmm", type(dic[key]))
            sys.exit()
    return out_dic


def _read_pickle_select(
    fn,
    structure_mode,  # igfold, immunebuilder, etc...
    model_index,
    select_feature_s=["rigidgroups_gt_frames"],
    use_crystal=True
    # check_consistency=False,
    # select_feature_s=["rigidgroups_gt_frames", "torsion_angle_sin_cos"],
    # trial: train w/o torsion angle as input feature. It might be useful for adding gaussian noise to input structure.
):
    # for error handling
    with open(fn, "rb") as fp:
        dat = pickle.load(fp)
    # set crystal structure mode manually
    # structure_mode='crystal'
    if structure_mode =="IgFold":
        model_index=0
    if structure_mode == "crystal":
        black = []
        for key in dat.keys():
            if "pertMD" in key:
                black.append(key)
            elif "IgFold" in key:
                black.append(key)
        for key in black:
            dat.pop(key)
        dat["keep_miss_mask"] = dat["miss_mask"].clone().detach()
        dat["input_rigidgroups_gt_frames"] = dat["rigidgroups_gt_frames"]
    else:
        if structure_mode == "IgFold" and (
            not "IgFold_rigidgroups_gt_frames" in dat.keys()
        ):
            structure_mode = "pertMD"
            model_index = random.randint(0, 31)
        elif structure_mode == "pertMD" and (
            not "pertMD_rigidgroups_gt_frames" in dat.keys()
        ):
            structure_mode = "IgFold"
            model_index = 0
            # if not check_consistency:
        """for debugging"""
        """
        structure_mode='IgFold'
        model_index=0
        print (structure_mode)
        print(model_index)
        """
        dat = generate_chimeric_feature(
            dat, structure_mode, select_feature_s, model_index
        )
    return dat


def find_index_from_tag(
    whole_hu_residue_index,
    whole_chain_id,
    tar_hu_residue_index,
    tar_chain_id,
):
    mask = (whole_hu_residue_index == tar_hu_residue_index) & (
        whole_chain_id == tar_chain_id
    )
    return mask.nonzero().item()


def generate_mapper_model_key(model_dat, structure_mode):  # generate model tag
    chain_s = list(set(model_dat[f"{structure_mode}_chain_id"].tolist()))
    out = {}
    out["tag"] = {}
    out["mapper"] = {}
    whole_hu_residue_index = model_dat[f"{structure_mode}_hu_residue_index"]
    for chain in chain_s:
        out["tag"][chain] = {}
        mask = model_dat[f"{structure_mode}_chain_id"] == chain
        out["tag"][chain]["model_tag"] = [
            whole_hu_residue_index[mask][0].item(),
            whole_hu_residue_index[mask][-1].item(),
        ]
    for chain in chain_s:
        out["mapper"][chain] = {}
        out["mapper"][chain]["model_range"] = [
            find_index_from_tag(
                whole_hu_residue_index,
                model_dat[f"{structure_mode}_chain_id"],
                out["tag"][chain]["model_tag"][0],
                chain,
            ),
            find_index_from_tag(
                whole_hu_residue_index,
                model_dat[f"{structure_mode}_chain_id"],
                out["tag"][chain]["model_tag"][-1],
                chain,
            ),
        ]
    return out


def generate_mapper_crystal_key(out, crystal_dic, structure_mode):
    chain_s = list(set(crystal_dic[f"{structure_mode}_chain_id"].tolist()))
    for chain in chain_s:
        out["mapper"][chain]["crystal_range"] = [
            find_index_from_tag(
                crystal_dic["hu_residue_index"],
                crystal_dic["chain_id"],
                out["tag"][chain]["model_tag"][0],
                chain,
            ),
            find_index_from_tag(
                crystal_dic["hu_residue_index"],
                crystal_dic["chain_id"],
                out["tag"][chain]["model_tag"][-1],
                chain,
            ),
        ]
    return out["mapper"]


def change_exists_with_miss_mask(dic, miss_mask):
    """
    Not all residues in crystal structure are constructed in model structure.
    This may cause problem in calculating loss.
    이미 cropping 단계에서 'miss_mask'를 참조해서 missing인 영역은 cropping이 안되긴하지만,
    이중 안전 장치겸, 나중에 cropping 없이 full structure refinement task를 위해 추가함.
    주의: miss_mask는 missing 영역이 False
    """
    stat = False
    for key in dic.keys():
        if not (isinstance(dic[key], torch.Tensor) or isinstance(dic[key], Rigid)):
            continue
        if key in ["miss_mask", "ulr_mask"]:
            continue
        if ("exist" in key) or ("mask" in key):
            dic[key][~miss_mask] = False
    return dic


def generate_chimeric_feature(
    dic,
    structure_mode,
    selected_feature_s,
    model_index,
):
    """
    overwrite model feature to crystal feature
    """
    crystal_chain_id_tensor = dic["chain_id"]
    # selected_feature_s=[f'{structure_mode}_{x}'for x in selected_feature_s]
    mapper = generate_mapper_model_key(dic, structure_mode)
    mapper = generate_mapper_crystal_key(mapper, dic, structure_mode)
    selected_feature_s.append("miss_mask")
    selected_feature_s = list(set(selected_feature_s))
    # Directly change the crystal feature dictionary. Is it okay?
    tmp_dic = {}
    for key in dic.keys():
        if not key in selected_feature_s:
            continue
        model_feature_key = f"{structure_mode}_{key}"
        new_out_key = f"input_{key}"
        ###
        crystal_feature = dic[key]
        if not len(dic[model_feature_key].shape) == 1:
            model_feature = dic[model_feature_key][model_index]
        else:
            model_feature = dic[model_feature_key]
        tmp_dic[new_out_key] = generate_chimeric_feature_single(
            crystal_feature, model_feature, mapper, key, crystal_chain_id_tensor
        )
    dic.update(tmp_dic)
    del_list = []
    for key in dic.keys():
        if "pertMD" in key or "IgFold" in key:
            del_list.append(key)
    for key in del_list:
        dic.pop(key)
    # overwrtie miss_mask
    dic["keep_miss_mask"] = dic["miss_mask"].clone().detach()
    dic["miss_mask"] = dic["input_miss_mask"]
    dic = change_exists_with_miss_mask(dic, dic["miss_mask"])
    return dic


def generate_chimeric_feature_single(
    crystal_feature,
    model_feature,
    mapper,
    key,
    crystal_chain_id_tensor,
):
    if not (
        isinstance(crystal_feature, torch.Tensor) or isinstance(crystal_feature, Rigid)
    ):
        print(
            f"Wrong input feature type: neither torch.Tensor nor Rigid,{type(crystal_feature)} is given."
        )
        sys.exit()
    # dic['mapper'] = {0: {'crystal_range':[index1,index2], 'model_range':[index1, index2]}, 1:{}}
    new_feature = crystal_feature.clone().detach()
    for chain_id in mapper.keys():  # chain_id is not important
        crystal_from_index = mapper[chain_id]["crystal_range"][0]
        crystal_to_index = mapper[chain_id]["crystal_range"][1] + 1
        model_from_index = mapper[chain_id]["model_range"][0]
        model_to_index = mapper[chain_id]["model_range"][1] + 1
        if isinstance(
            model_feature, Rigid
        ):  # if input obejct is "Rigid" class object -> reconstruct//
            """
            for now, input rigid is tensorform
            """
            inp_gt = Rigid(rots=None, trans=torch.zeros(1, 3))
            inp_gt = inp_gt.from_tensor_4x4(new_feature)
            model_trans = model_feature._trans
            model_rots = model_feature._rots._rot_mats
            chimeric_trans = inp_gt._trans.clone()[
                ..., 0, :
            ]  # select backbone frame only
            chimeric_rots = inp_gt._rots._rot_mats.clone()[
                ..., 0, :, :
            ]  # select backbone frame onlyi
            chimeric_trans[crystal_from_index:crystal_to_index] = model_trans[
                model_from_index:model_to_index
            ]
            chimeric_rots[crystal_from_index:crystal_to_index] = model_rots[
                model_from_index:model_to_index
            ]
        else:  # normal tensor information
            if (
                key == "miss_mask"
            ):  # may not all crystal structure is modelled.ex.IgFold
                chain_mask = crystal_chain_id_tensor == chain_id
                new_feature[chain_mask] = False

            if chain_id == 0 and (not model_from_index == 0):
                print(
                    "WARNING : for the first chain , model_from_index is not 0. Are you sure? Is it okay?"
                )
                sys.exit()
            if not (crystal_to_index - crystal_from_index) == (
                model_to_index - model_from_index
            ):
                print("WARNING : mapping length is different. {mapper},{chain_id}")
                sys.exit()
            new_feature[crystal_from_index:crystal_to_index] = model_feature[
                model_from_index:model_to_index
            ]
    if isinstance(
        model_feature, Rigid
    ):  # if input obejct is "Rigid" class object -> reconstruct
        new_feature = Rigid(Rotation(rot_mats=chimeric_rots), chimeric_trans)
    return new_feature


def initialize_ulr_rigid(
    dic, ulr_type_s=["H_3"], mode="linspace", n_residue_extension=0
):
    inp_gt = Rigid(rots=None, trans=torch.zeros(1, 3))
    inp_gt = inp_gt.from_tensor_4x4(dic["input_rigidgroups_gt_frames"].clone().detach())
    # [L_tot, 8, 4, 4]
    new_trans = inp_gt._trans[..., 0, :]  # (C, CA, N frame)
    new_rots = inp_gt._rots._rot_mats[..., 0, :, :]
    ###########
    """ for debugging"""
    """
    inp_gt = inp_gt.from_tensor_4x4(dic["rigidgroups_gt_frames"].clone().detach())
    # [L_tot, 8, 4, 4]
    new_trans2 = inp_gt._trans[..., 0, :]  # (C, CA, N frame)
    print (new_trans[0])
    print (new_trans2[0])
    print (new_trans-new_trans2)
    sys.exit()
    """
    # how about just with backbone_rigid_tensor?
    ##
    skip_ulr_type = []
    trs_point_s = []
    whole_ulr_mask = []
    for ulr_type in ulr_type_s:
        if ulr_type == "general":
            tmp = dic["ulr_mask"].nonzero()
            ulr_range = [tmp[0].item(), tmp[-1].item()]
        else:
            ulr_range = dic["tot_ulr_range"][ulr_type]
        ##
        from_index = ulr_range[0] - n_residue_extension
        to_index = ulr_range[-1] + n_residue_extension
        ##
        # Define ulr mask
        ulr_mask = torch.zeros_like(dic["miss_mask"]).clone().bool()
        ulr_mask[from_index : to_index + 1] = True
        ulr_rigid = Rigid(rots=None, trans=torch.zeros(1, 3))
        ulr_rigid = ulr_rigid.identity(inp_gt[ulr_mask, 0].shape, fmt="rot_mat")
        ##
        miss_mask = dic["miss_mask"]
        if (not miss_mask[from_index - 1]) or (not miss_mask[to_index + 1]):
            print(f"WARNING :: Anchor reisude is missing, skip {ulr_type}")
            skip_ulr_type.append(ulr_type)
            continue
        ## cropping center (average of anchors on both sides)
        trs_point = (
            inp_gt[from_index - 1][0]._trans + inp_gt[to_index + 1][0]._trans
        ) / 2
        trs_point_s.append(trs_point)
        if mode == "center":
            ulr_rigid._trans = trs_point
        elif mode == "linspace":
            from_point = inp_gt[from_index - 1][0]._trans  # anchor
            to_point = inp_gt[to_index + 1][0]._trans  # anchor
            length = (to_index + 1) - (from_index - 1) + 1
            memo = []
            for x in range(3):
                memo.append(
                    torch.linspace(
                        start=from_point[x].item(), end=to_point[x].item(), steps=length
                    ).unsqueeze(-1)
                )
            memo = torch.cat(memo, dim=-1)[1:-1]
        whole_ulr_mask.append(ulr_mask)
        #####
        new_trans[ulr_mask, :] = memo
        tmp_rots = ulr_rigid._rots._rot_mats
        new_rots[ulr_mask] = tmp_rots  # 여기서는 아직 identity?
        #####
    new_rigid = Rigid(Rotation(rot_mats=new_rots), new_trans)
    trs_point = torch.stack(trs_point_s, dim=0).mean(dim=0)
    whole_ulr_mask = torch.stack(whole_ulr_mask, dim=-1).sum(dim=-1).bool()

    return new_rigid, trs_point, skip_ulr_type, whole_ulr_mask
