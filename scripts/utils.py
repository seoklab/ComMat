import torch, os, json
import pandas as pd
import numpy as np
import pickle
from openfold.utils.rigid_utils import Rigid, Rotation
from openfold.np.protein import Protein, to_pdb


def split_output_loss_dic(loss_dic, batch_size):
    memo = [{} for x in range(batch_size)]
    for x in loss_dic.keys():
        for y in range(batch_size):
            memo[y][x] = loss_dic[x][y]
    return memo


def from_mask_to_resno(ulr_mask, hu_residue_index, header):
    tmp = hu_residue_index[ulr_mask].tolist()
    sen = f"REMARK {header} "
    for x in tmp:
        sen += f"{x} "
    sen += "\n"
    return sen


def get_post_prediction_pdb(input_dic, pred, tag_s, rmsd_s, header):
    out_string = ""
    seed_size = input_dic["aatype"].shape[1]
    batch_size = input_dic["aatype"].shape[0]
    ulr_mask = input_dic["ulr_mask"].bool()
    h3_ulr_mask = input_dic["raw_ulr_mask"].bool()
    all_cdr_mask = input_dic["all_cdr_mask"].bool()
    hu_residue_index = input_dic["hu_residue_index"].long()
    for batch_idx in range(batch_size):
        tag = tag_s[batch_idx]
        out_string = ""
        for seed_idx in range(seed_size):
            out_string += "MODEL %i\n" % (seed_idx + 1)
            out_string += "REMARK RMSD %8.3f\n" % (rmsd_s[batch_idx, seed_idx])
            out_string += from_mask_to_resno(
                h3_ulr_mask[batch_idx, seed_idx],
                hu_residue_index[batch_idx, seed_idx],
                header="H3",
            )
            out_string += from_mask_to_resno(
                all_cdr_mask[batch_idx, seed_idx],
                hu_residue_index[batch_idx, seed_idx],
                header="CDR",
            )
            out_string += from_mask_to_resno(
                ulr_mask[batch_idx, seed_idx],
                hu_residue_index[batch_idx, seed_idx],
                header="sampled",
            )
            pdb_out = Protein(
                aatype=input_dic["aatype"][batch_idx, seed_idx].cpu().numpy(),
                atom_positions=pred["final_atom_positions"][batch_idx, seed_idx]
                .cpu()
                .numpy(),
                atom_mask=pred["final_atom_mask"][batch_idx, seed_idx].cpu().numpy(),
                residue_index=input_dic["hu_residue_index"][batch_idx, seed_idx]
                .cpu()
                .numpy(),
                b_factors=torch.zeros_like(
                    pred["final_atom_mask"][batch_idx, seed_idx].cpu()
                ).numpy(),
                chain_index=input_dic["chain_id"][batch_idx, seed_idx].cpu().numpy(),
                remark=None,
                parents=None,
                parents_chain_index=None,
            )
            out_string += to_pdb(pdb_out)
            out_string += "ENDMDL\n"
        with open(f"{header}.{tag}.pdb", "wt") as fp:
            fp.writelines(out_string)
    return None


#
def get_ulr_including_chain_mask(input_dic):
    chain_id = input_dic["chain_id"]  # L
    ulr_mask = input_dic["all_cdr_mask"]  # L
    tmp_a = chain_id[ulr_mask].tolist()
    memo = []
    for x in list(set(tmp_a)):
        memo.append(chain_id == x)
    memo = torch.stack(memo, dim=0)
    memo = memo.sum(dim=0).bool()
    return memo


#
def do_pre_kabsch(input_dic, tag):
    ###
    kabsch_mask = input_dic["kabsch_mask"]
    ###
    # using non_ulr region
    target_chain_mask = get_ulr_including_chain_mask(input_dic)
    exclude_ag_mask = input_dic["chain_tag"] != 2
    non_cdr_only = (~input_dic["all_cdr_mask"]) & exclude_ag_mask
    mask = non_cdr_only & kabsch_mask
    mask = mask & target_chain_mask
    mask = mask[None, None, :]
    ###
    ca_from_rigids = input_dic["inp_gt"]._trans.clone()[None, None, ...]
    ca_from_rigids = ca_from_rigids[..., None, :]
    tmp_ref_coords = input_dic["all_atom_positions"][None, None, ...].clone()
    batch_size, seed_size, length = mask.shape
    tmp_mask = mask.long()
    tmp_mask = tmp_mask[:, 0, :]  # B,L
    tmp_mask = tmp_mask.sum(dim=0)  # L
    tmp_mask = tmp_mask == batch_size  # L
    aligned_coords = do_kabsch(
        ca_from_rigids[:, :, tmp_mask, :, :],
        tmp_ref_coords[:, :, tmp_mask, 1:2, :],
        ca_from_rigids.reshape(batch_size, seed_size, -1, 3),
        mode="pre",
        tag=tag,
    )
    aligned_coords = aligned_coords.reshape(batch_size, seed_size, length, 3)[0, 0, ...]
    input_dic["inp_gt"]._trans = aligned_coords
    #
    return input_dic

def do_post_kabsch2(input_dic, pred, tag_s):
    kabsch_mask = input_dic["kabsch_mask"].bool()
    # using non_ulr region
    exclude_ag_mask = input_dic["chain_tag"] != 2
    non_cdr_only = ~input_dic["all_cdr_mask"]
    mask = non_cdr_only & kabsch_mask
    batch_size, seed_size, length = mask.shape
    # common region
    tmp_mask = mask.long()
    tmp_mask = tmp_mask[:, 0, :]  # B,L: share among seed dimension
    tmp_mask = tmp_mask.bool()
    memo = []
    ca_from_rigids = input_dic["inp_gt"]._trans.clone()[..., None, :] * 10
    tmp_ref_coords = input_dic["all_atom_positions"].clone()
    for batch_idx in range(batch_size):  # lousy for loop... but safe kabsch alignment
        tag = tag_s[batch_idx]
        aligned_coords=do_kabsch(
                pred['final_atom_positions'][batch_idx:batch_idx+1,:,tmp_mask[batch_idx],1:2,:],
                input_dic['all_atom_positions'][batch_idx:batch_idx+1,:,tmp_mask[batch_idx],1:2,:],
                pred['final_atom_positions'][batch_idx:batch_idx+1].reshape(1,seed_size,-1,3),
                mode='post',
                tag=tag,
                )
        #aligned_coords = do_kabsch(
        #    input_dic["all_atom_positions"][
        #        batch_idx : batch_idx + 1, :, tmp_mask[batch_idx], 1:2, :
        #    ],
        #    pred["final_atom_positions"][
        #        batch_idx : batch_idx + 1, :, tmp_mask[batch_idx], 1:2, :
        #    ],
        #    input_dic["all_atom_positions"][batch_idx : batch_idx + 1].reshape(
        #        1, seed_size, -1, 3
        #    ),
        #    mode="post",
        #    tag=tag,
        #)
        aligned_coords = aligned_coords.reshape(1, seed_size, length, -1, 3)
        memo.append(aligned_coords)
    memo = torch.cat(memo, dim=0)
    return memo

def do_post_kabsch(input_dic, pred, tag_s):
    kabsch_mask = input_dic["kabsch_mask"].bool()
    # using non_ulr region
    exclude_ag_mask = input_dic["chain_tag"] != 2
    non_cdr_only = ~input_dic["all_cdr_mask"]
    mask = non_cdr_only & kabsch_mask
    batch_size, seed_size, length = mask.shape
    # common region
    tmp_mask = mask.long()
    tmp_mask = tmp_mask[:, 0, :]  # B,L: share among seed dimension
    tmp_mask = tmp_mask.bool()
    memo = []
    ca_from_rigids = input_dic["inp_gt"]._trans.clone()[..., None, :] * 10
    tmp_ref_coords = input_dic["all_atom_positions"].clone()
    for batch_idx in range(batch_size):  # lousy for loop... but safe kabsch alignment
        tag = tag_s[batch_idx]
        # aligned_coords=do_kabsch(
        #        pred['final_atom_positions'][batch_idx:batch_idx+1,:,tmp_mask[batch_idx],1:2,:],
        #        input_dic['all_atom_positions'][batch_idx:batch_idx+1,:,tmp_mask[batch_idx],1:2,:],
        #        pred['final_atom_positions'][batch_idx:batch_idx+1].reshape(1,seed_size,-1,3),
        #        mode='post',
        #        tag=tag,
        #        )
        aligned_coords = do_kabsch(
            input_dic["all_atom_positions"][
                batch_idx : batch_idx + 1, :, tmp_mask[batch_idx], 1:2, :
            ],
            pred["final_atom_positions"][
                batch_idx : batch_idx + 1, :, tmp_mask[batch_idx], 1:2, :
            ],
            input_dic["all_atom_positions"][batch_idx : batch_idx + 1].reshape(
                1, seed_size, -1, 3
            ),
            mode="post",
            tag=tag,
        )
        aligned_coords = aligned_coords.reshape(1, seed_size, length, -1, 3)
        memo.append(aligned_coords)
    memo = torch.cat(memo, dim=0)
    return memo


def do_kabsch(
    model, reference, full_model, mode, tag=None
):  # model:B,S,L,N,3 // reference:B,S,L,N,3
    batch_size, seed_size, length, atom_number, _ = model.shape
    model = model.reshape(batch_size, seed_size, length * atom_number, _).clone()
    reference = reference.reshape(
        batch_size, seed_size, length * atom_number, _
    ).clone()
    with torch.no_grad():
        R, t = find_rigid_alignment(model, reference)
    aligned_models = hu_bmm(R, full_model.transpose(-1, -2)).transpose(-1, -2) + t
    ###
    return aligned_models


def hu_bmm(a, b):
    return torch.einsum("bsik,bskj->bsij", a, b)


def find_rigid_alignment(A, B):
    A = A.double()
    B = B.double()
    # B=B.expand_as(A)
    a_mean = A.mean(axis=-2).unsqueeze(-2)
    b_mean = B.mean(axis=-2).unsqueeze(-2)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    A_c = A_c.transpose(-1, -2)
    H = hu_bmm(A_c, B_c)
    # H = A_c.bmm(B_c)
    U, S, V = torch.svd(H)
    # Rotation matrix
    U = U.transpose(-1, -2)
    R = hu_bmm(V, U)
    # Translation vector
    t = b_mean - hu_bmm(R, a_mean.transpose(-1, -2)).transpose(-1, -2)
    R = R.float()
    t = t.float()
    return R, t


def compute_rmsd(models, reference):
    diff = models - reference
    rmsd = (diff**2).sum(dim=-1)
    rmsd = rmsd.mean(dim=-1)
    rmsd = torch.sqrt(rmsd + 1e-8)
    return rmsd


def concat_dic(
    dic1, dic2
):  # two dictionary has same key and all value is Tensor object
    dic = {}
    for key in dic1.keys():
        dic[key] = torch.cat([dic1[key], dic2[key]], dim=0)
    return dic


def normalize_vector(x):
    y = torch.norm(x, dim=-1)
    y = y ** (-1)
    y = y.unsqueeze(-1)
    return x * y


def calc_torsion_from_4point(v1, v2, v3, v4):
    r1 = v2 - v1
    r2 = v3 - v2
    r3 = v4 - v3

    v1 = torch.cross(r1, r2)
    v2 = torch.cross(r2, r3)

    v1_norm = torch.norm(v1, dim=1)
    v2_norm = torch.norm(v2, dim=1)

    dot_prod = torch.sum(v1 * v2, dim=1)
    cross_prod = torch.cross(v1, v2)
    cross_prod_norm = torch.norm(cross_prod, dim=1)

    angle = torch.atan2(cross_prod_norm, dot_prod)
    return angle


def add_torsion_to_ndata(ndata):
    crd = cast_global_coord(ndata)
    # calc torsion N->CA->C->N_next
    n = crd[:-1, 1]
    ca = crd[:-1, 0]
    c = crd[:-1, 2]
    n_next = crd[1:, 1]
    tor1 = calc_torsion_from_4point(n, ca, c, n_next)
    # C_prev->N->CA->C
    n = crd[1:, 1]
    ca = crd[1:, 0]
    c = crd[1:, 2]
    c_prev = crd[:-1, 2]
    tor2 = calc_torsion_from_4point(c_prev, n, ca, c)

    ndata["tor"] = torch.zeros_like(ndata["l1"][:, 0:2, 0])  # N,2
    ndata["tor"][0:-1, 0] = tor1
    ndata["tor"][1:, 1] = tor2
    return ndata


def clone_dic(dic, skip=[], expansion_templat=None):
    out_dic = {}
    for key in dic.keys():
        if key in skip:
            continue
        out_dic[key] = dic[key].clone().detach()
    return out_dic


def rot_vec_mul(r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Applies a rotation to a vector. Written out by hand to avoid transfer
    to low-precision tensor cores.

    Args:
        r: [*, 3, 3] rotation matrices
        t: [*, 3] coordinate tensors
    Returns:
        [*, 3] rotated coordinates
    """
    x = t[..., 0]
    y = t[..., 1]
    z = t[..., 2]
    return torch.stack(
        [
            r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
            r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
            r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
        ],
        dim=-1,
    )


def _read_pickle(fn):
    with open(fn, "rb") as fp:
        dat = pickle.load(fp)
    return dat


def gen_bin_center(min_center, no_bin, bin_width, device="cuda"):
    return (torch.arange(no_bin) * bin_width + min_center).to(device=device)


def bin_index(value, bin_center_s):
    value = value.unsqueeze(-1)
    delta = torch.abs(value - bin_center_s)
    delta = torch.argmin(delta, dim=-1)
    return delta


def update_epoch_loss(epoch_loss, single_loss):
    for key in single_loss.keys():
        epoch_loss[key].append(single_loss[key])
    return epoch_loss


def epoch_loss_be_tensor(epoch_loss):
    for key in single_loss.keys():
        epoch_loss[key] = torch.Tensor(epoch_loss[key])
    return epoch_loss


def finalize_epoch_loss(epoch_loss, device="cuda"):
    for key in epoch_loss.keys():
        if key in ["topk_acc", "n_count"]:
            epoch_loss[key] = torch.Tensor(epoch_loss[key]).sum(dim=0).to(device=device)
        else:
            epoch_loss[key] = (
                torch.Tensor(epoch_loss[key]).mean(dim=0).to(device=device)
            )
    return epoch_loss


def report_epoch_loss(epoch_loss, epoch_idx, mode, out_tag=None, save_rmsd=False):
    log_fn = "%s.log" % mode
    if os.path.exists(log_fn):
        with open(log_fn) as fp:
            wrt = fp.readlines()
    else:
        header = "%10i " % epoch_idx
        for key in epoch_loss.keys():
            header += "%15s  " % key
        header += "\n"
        wrt = [header]
    if save_rmsd:
        tmp = epoch_loss["rmsd_loss"].cpu()
        with open("saved.rmsd.dat", "wb") as fp:
            pickle.dump(tmp, fp)

    sen = "%10i " % epoch_idx
    for key in epoch_loss.keys():
        sen += "%15.3f " % epoch_loss[key]
    sen += "\n"
    wrt.append(sen)
    with open(log_fn, "wt") as fp:
        fp.writelines(wrt)


def reduce_epoch_loss(epoch_loss, world_size, inference=False):
    if not inference:
        for loss_key in epoch_loss.keys():
            torch.distributed.all_reduce(epoch_loss[loss_key])
            epoch_loss[loss_key] = epoch_loss[loss_key] / world_size

        return epoch_loss
    else:
        for loss_key in list(epoch_loss.keys()):
            if "H3_rmsd_stack" in loss_key:
                gathered_tensor = [
                    torch.empty_like(epoch_loss[loss_key]) for _ in range(world_size)
                ]
                torch.distributed.all_gather(gathered_tensor, epoch_loss[loss_key])
                gathered_tensor = torch.cat(gathered_tensor, dim=0)
                median_value = torch.median(gathered_tensor)
                total_values = gathered_tensor.numel()
                count_2_or_less = (
                    torch.sum(gathered_tensor <= 2.0).item() / total_values
                )
                count_1_5_or_less = (
                    torch.sum(gathered_tensor <= 1.5).item() / total_values
                )
                count_1_or_less = (
                    torch.sum(gathered_tensor <= 1.0).item() / total_values
                )

                if loss_key == "best_H3_rmsd_stack":
                    epoch_loss["median_best_H3_rmsd"] = median_value
                    epoch_loss["best_H3<2.0"] = count_2_or_less
                    epoch_loss["best_H3<1.5"] = count_1_5_or_less
                    epoch_loss["best_H3<1.0"] = count_1_or_less
                elif loss_key == "top1_H3_rmsd_stack":
                    epoch_loss["median_top1_H3_rmsd"] = median_value
                    epoch_loss["top1_H3<2.0"] = count_2_or_less
                    epoch_loss["top1_H3<1.5"] = count_1_5_or_less
                    epoch_loss["top1_H3<1.0"] = count_1_or_less

            else:
                torch.distributed.all_reduce(epoch_loss[loss_key])
                epoch_loss[loss_key] = epoch_loss[loss_key] / world_size
        return epoch_loss
