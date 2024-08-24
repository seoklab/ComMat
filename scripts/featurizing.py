import os, glob, sys
from collections import defaultdict
import openfold.np.protein as protein
import openfold.np.residue_constants as rc
import torch
import openfold.data.data_transforms as dt
import openfold.data.data_pipeline as dp
import openfold.data.input_pipeline as ip
import openfold.data.feature_pipeline as Fp
from openfold.np.residue_constants import restype_3to1, restypes
import time, pickle
import multiprocessing as mp
import tempfile
import esm
from pathlib import Path

CDR_DIC = {}
CDR_DIC["H"] = {
    "H_1": list(range(26, 33)),
    "H_2": list(range(52, 57)),
    "H_3": list(range(95, 103)),
}
CDR_DIC["L"] = {
    "L_1": list(range(24, 35)),
    "L_2": list(range(50, 57)),
    "L_3": list(range(89, 98)),
}
EXEC_ANARCI_PATH = "ANARCI"


def read_pdb_write_seq(pdb_fn, out_fn):
    fasta_dic = defaultdict(str)
    with open(pdb_fn, "r") as fp:
        for line in fp:
            if line.startswith("ATOM"):
                assert (
                    line[26] == " "
                ), "Only support normal residue numbered PDB, not Chothia, Kabat etc.. "
                if not line[12:16].strip() == "CA":
                    continue
                chain_id = line[21]
                res_name = line[17:20].strip()
                fasta_dic[chain_id] += restype_3to1[res_name]
    wrt = []
    for chain_id, seq in fasta_dic.items():
        wrt.append(f">{chain_id}\n")
        wrt.append(f"{seq}\n")
    with open(out_fn, "wt") as fp:
        fp.writelines(wrt)
    return fasta_dic


def chothia_number_to_cdr_type(chothia_number, chain):
    try:
        int_resno = int(chothia_number)
    except:
        int_resno = int(chothia_number[:-1])  # 100A, 100B etc
    for cdr_type, cdr_range in CDR_DIC[chain].items():
        if int_resno in cdr_range:
            return cdr_type


def parse_anarci_out_to_cdr_range(fn, input_pdb_seq, chain):
    dat = open(fn).readlines()
    ref = dat[0]
    query = dat[1]
    ref = ref.strip("\n").split(",")[13:]
    query = query.strip("\n").split(",")[13:]
    aligned_query_seq = "".join([x for x in query if not x == "-"])
    # compare sequence from ANARCI & sequence from input structure
    start_index = -1
    try:
        start_index = input_pdb_seq.index(aligned_query_seq)
    except:
        raise ValueError("ANARCI alignment failed with sequence from input structure")
    # numbering 1~
    cdr_type_resno_dic = defaultdict(list)
    residue_number = 0
    for idx, (chothia_number, aligned_qeury_aa_type) in enumerate(
        list(zip(ref, query))
    ):
        if not aligned_qeury_aa_type == "-":
            residue_number += 1
        cdr_type = chothia_number_to_cdr_type(chothia_number, chain)
        if not cdr_type is None:
            cdr_type_resno_dic[cdr_type].append(
                residue_number - 1
            )  # python indexing start with 0~
    return cdr_type_resno_dic, aligned_query_seq


def modify_pdb_with_anarci_aligened_sequence(
    pdb_fn,
    str_seq_dic,
    aligned_seq_dic,
    output_path,
):
    lines = open(pdb_fn).readlines()
    lines = [line for line in lines if line.startswith("ATOM")]
    out_dict = defaultdict(list)
    for line in lines:
        out_dict[line[21]].append(line)
    for chain in ["H", "L"]:
        lines = out_dict[chain]
        start_res_no = int(lines[0][22:26]) + str_seq_dic[chain].index(
            aligned_seq_dic[chain]
        )
        end_res_no = start_res_no + len(aligned_seq_dic[chain])
        filtered_lines = []
        for line in lines:
            res_no = int(line[22:26])
            if not res_no in range(start_res_no, end_res_no + 1):
                continue
            new_res_no = res_no - (start_res_no - 1)
            line = line[:22] + "%4i" % (new_res_no) + line[26:]
            filtered_lines.append(line)
        with open(f"{output_path}/{chain}.pdb", "wt") as fp:
            fp.writelines(filtered_lines)


def mapping_cdr_region_and_modify_pdb(pdb_fn, output_path):
    fasta_dic = read_pdb_write_seq(pdb_fn, out_fn=f"{output_path}/output.fa")
    print("hello")
    os.system(
        f"{EXEC_ANARCI_PATH} -i {output_path}/output.fa -s c -o {output_path}/anarci.out --csv"
    )
    print("bye")
    anarci_h_chain_fn = f"{output_path}/anarci.out_H.csv"
    anarci_l_chain_fn = f"{output_path}/anarci.out_KL.csv"
    cdr_H, aligned_H_seq = parse_anarci_out_to_cdr_range(
        anarci_h_chain_fn, fasta_dic["H"], "H"
    )
    cdr_L, aligned_L_seq = parse_anarci_out_to_cdr_range(
        anarci_l_chain_fn, fasta_dic["L"], "L"
    )
    cdr_resno_mapper = {}
    cdr_resno_mapper.update(cdr_H)
    cdr_resno_mapper.update(cdr_L)
    modify_pdb_with_anarci_aligened_sequence(
        pdb_fn=pdb_fn,
        str_seq_dic={"H": fasta_dic["H"], "L": fasta_dic["L"]},
        aligned_seq_dic={"H": aligned_H_seq, "L": aligned_L_seq},
        output_path=output_path,
    )
    return cdr_resno_mapper, {"H": aligned_H_seq, "L": aligned_L_seq}


def prep_single_chain(pdb_fn, output_path):
    ##
    chain_id = Path(pdb_fn).stem
    ##
    my_pipe = dp.DataPipeline(None)
    my_fp = Fp.FeaturePipeline()
    data = my_pipe.process_pdb(
        pdb_path=pdb_fn,
        alignment_dir=None,
        is_distillation=False,
        chain_id=chain_id,
    )
    feats = my_fp.process_features(data, "train")
    feats["hu_residue_index"] = torch.Tensor(data["hu_residue_index"])
    with open(f"{output_path}/single_chain.%s.pkl" % (chain_id), "wb") as fp:
        pickle.dump(feats, fp)
    return feats


def padding_for_missing(dic, aa_fa):
    hu_residue_index = dic["hu_residue_index"]
    ref = torch.arange(1, len(aa_fa) + 1)
    dic["miss_mask"] = torch.zeros_like(dic["aatype"]).fill_(1).bool()
    if hu_residue_index.shape[0] == ref.shape[0]:
        return dic
    else:
        if aa_fa is None:
            print("Require full length of sequence")
            sys.exit()
        ref = torch.arange(1, len(aa_fa) + 1)
        diff = ref[:, None] - hu_residue_index[None, :]  # N,n
        diff = (diff == 0).sum(dim=-1).bool()
        for key in dic.keys():
            new = torch.zeros(dic[key].shape[1:])[None, ...]  # 1 ,...
            new = new.expand(diff.shape[0], *new.shape[1:])  # full_len, ....
            new = new.type(dic[key].type())  # set type
            test = new.clone()
            test[diff] = dic[key]
            dic[key] = test
        aa_fa = torch.tensor([restypes.index(aa) for aa in aa_fa])
        dic["aatype"] = aa_fa
        dic["hu_residue_index"] = ref
        n_miss = (~diff).long().sum()
        return dic


def generate_ulr_range(dic, ulr_s=[], mode=None):
    out_dic = {}
    for ulr in dic.keys():
        tmp = dic[ulr].nonzero()
        from_index = torch.min(tmp).item()
        to_index = torch.max(tmp).item()
        out_dic[ulr] = [from_index, to_index]
    return out_dic


def ulr_range_to_mask(dat, templ, len_h):
    out_dic = {}
    for idx, key in enumerate(dat.keys()):
        if "H" in key:
            gap = 0
        elif "L" in key:
            gap = len_h
        mask = torch.zeros_like(templ).bool()
        ulr_range = list(dat[key])
        start_res_idx = ulr_range[0] + gap
        mask[start_res_idx : start_res_idx + len(ulr_range)] = True
        out_dic[key] = mask
    return out_dic


##About run ESM
def conversion_single(fa):
    fa = fa.tolist()
    sen = ""
    for x in fa:
        sen += restypes[x]
        # sen+=AA1[x]
    return sen


def run_by_chain(aatype, chain_idx, model, batch_converter, alphabet):
    max_chain_idx = torch.max(chain_idx)
    lang_out = []
    for x in range(max_chain_idx + 1):
        single_chain_fa = aatype[(chain_idx == x)]
        single_chain_fa = conversion_single(single_chain_fa)
        lang_out.append(
            run_single_chain(model, single_chain_fa, batch_converter, alphabet)
        )
    lang_out = torch.cat(lang_out, dim=0)
    return lang_out


def run_single_chain(model, seq, batch_converter, alphabet):
    data = [
        ("protein1", seq),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    with torch.no_grad():
        batch_tokens = batch_tokens.cuda()
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33][0][1:-1]
    return token_representations


def run_esm(feats):
    print("start, esm loading")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.cuda()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    aatype = feats["aatype"]
    chain_idx = feats["chain_id"]
    lang_out = run_by_chain(aatype, chain_idx, model, batch_converter, alphabet)
    feats["lang_out"] = lang_out.cpu()
    print("end, esm loading")
    return feats


def prep_af2_inp(pdb_fn, pdbname, output_path):
    pdb_fn = os.path.abspath(pdb_fn)
    tag = pdbname
    curr_path = os.popen("pwd").readlines()[0].strip("\n")
    # with tempfile.TemporaryDirectory() as tmpdir:
    #     os.chdir(tmpdir)
    cdr_resno_mapper, aligned_seq_dic = mapping_cdr_region_and_modify_pdb(
        pdb_fn, output_path
    )
    print(output_path)
    feats = {
        single_chain: prep_single_chain(
            f"{output_path}/{single_chain}.pdb", output_path
        )
        for single_chain in ["H", "L"]
    }
    os.chdir(curr_path)
    for chain_idx, chain in enumerate(["H", "L"]):
        feats[chain] = padding_for_missing(feats[chain], aligned_seq_dic[chain])
        feats[chain]["chain_id"] = (
            torch.zeros(feats[chain]["aatype"].shape[0]).long().fill_(chain_idx)
        )
    merged_feats = defaultdict(list)
    for key, item in feats["H"].items():
        merged_feats[key].append(feats["H"][key])
        merged_feats[key].append(feats["L"][key])
        merged_feats[key] = torch.cat(merged_feats[key], dim=0)
    #
    len_h = feats["H"]["aatype"].shape[0]
    ulr_dic = ulr_range_to_mask(cdr_resno_mapper, merged_feats["aatype"], len_h)
    ulr_range_dic = generate_ulr_range(ulr_dic)
    merged_feats["tot_ulr"] = ulr_dic
    merged_feats["tot_ulr_range"] = ulr_range_dic
    #
    for key in ["hu_residue_index", "chain_id", "rigidgroups_gt_frames", "miss_mask"]:
        merged_feats[f"IgFold_{key}"] = merged_feats[key]
    merged_feats = run_esm(merged_feats)
    with open(f"{output_path}/{tag}.pkl", "wb") as fp:
        pickle.dump(merged_feats, fp)
