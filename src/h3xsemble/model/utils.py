import torch
import torch.nn as nn
from h3xsemble.utils.rigid_utils import Rotation, Rigid, rot_matmul
import numpy as np
from h3xsemble.utils.residue_constants import rigid_group_atom_positions
from h3xsemble.utils.rigid_utils import Rotation, Rigid

def clone_rigid(input_r, do_detach=True):
    if do_detach:
        tmp_trans = input_r._trans.clone().detach()
        tmp_rots = input_r._rots.detach()
    else:
        print("Not now!")
    new_r = Rigid(tmp_rots, tmp_trans)
    return new_r


def prep_rel_pos(res_idx, chain_idx, max_rel_pos=32):
    ##
    rel_pos = res_idx[..., None] - res_idx[..., None, :]
    rel_pos = torch.clamp(rel_pos, min=-max_rel_pos, max=max_rel_pos)
    rel_pos = rel_pos + max_rel_pos
    ##
    rel_chain = chain_idx[..., None] - chain_idx[..., None, :]
    inter_chain_mask = (rel_chain != 0).bool()
    ##
    rel_pos[inter_chain_mask] = max_rel_pos * 2 + 1
    return rel_pos.long()


def rand_rotation_matrix(
    seed_size, deflection=1.0, randnums=None, device="cuda", random_seed=0
):
    """Ref: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.53.1357&rep=rep1&type=pdf"""
    torch.manual_seed(random_seed)
    randnums = torch.rand(seed_size, 3).to(device=device)
    randnums[:, 0] = randnums[:, 0] * 2.0 * deflection * torch.pi
    randnums[:, 1] = randnums[:, 1] * 2.0 * torch.pi
    randnums[:, 2] = randnums[:, 2] * 2.0 * deflection
    # theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    # phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    # z = z * 2.0 * deflection  # For magnitude of pole deflection
    r = torch.sqrt(randnums[:, 2])
    # Vx, Vy, Vz = V = (
    #    np.sin(phi) * r,
    #    np.cos(phi) * r,
    #    np.sqrt(2.0 - z)
    #    )
    ##
    V = torch.stack(
        [
            torch.sin(randnums[:, 1]) * r,
            torch.cos(randnums[:, 1]) * r,
            torch.sqrt(2.0 - randnums[:, 2]),
        ],
        dim=-1,
    )
    st = torch.sin(randnums[:, 0])
    ct = torch.cos(randnums[:, 0])
    ##
    R = torch.zeros(seed_size, 3, 3).to(device=device)
    R[:, 2, 2] = 1
    R[:, 0, 0] = ct
    R[:, 0, 1] = st
    R[:, 1, 0] = -st
    R[:, 1, 1] = ct
    ##
    M = torch.einsum("bi,bj->bij", (V, V)) - torch.eye(3).to(device=device)[None, ...]
    M = torch.einsum("bij,bjk->bik", (M, R))
    ##
    return M

def get_bb_pos(
    pred_frames: Rigid, #[B, S, L]
    trans_scale_factor: float,
    ):
    ala_ref_pos = rigid_group_atom_positions["ALA"]
    atom_1 = [*ala_ref_pos[0][2], 1.0]
    atom_2 = [*ala_ref_pos[1][2], 1.0]
    atom_3 = [*ala_ref_pos[2][2], 1.0]
    bb_ref_pos = torch.tensor([atom_1, atom_2, atom_3]).to(pred_frames.device)
    # [3, 4]
    # bb_ref_pos (3, 4) - N, CA, C atom positions with appended 1.0
    rot_trans_mat = pred_frames.to_tensor_4x4() #[B, S, L, 4, 4]
    rot_trans_mat[..., :-1, -1] = rot_trans_mat[..., :-1, -1] * 10
    final_frame = torch.einsum('bslij,cj->bslci', [rot_trans_mat, bb_ref_pos])
    final_frame = final_frame[..., :-1]
    return final_frame

class InitialPertTrs(nn.Module):
    def __init__(
        self,
        trans_pert=0.3,
        rot_pert=1.0,
        use_pert_non_ulr=False,
        trans_pert_non_ulr=0.1,
        rot_pert_non_ulr=0.3,
    ):
        super(InitialPertTrs, self).__init__()
        self.trans_pert = trans_pert
        self.rot_pert = rot_pert
        self.trans_pert_non_ulr = trans_pert_non_ulr
        self.rot_pert_non_ulr = rot_pert_non_ulr
        self.use_pert_non_ulr = use_pert_non_ulr

    def forward(self, s, z, rigids, ulr_mask, str_mask, train_mode=True, inference_mode=False):
        batch_size = rigids.shape[0]  # B
        seed_size = rigids.shape[1]  # S
        rigid_len = rigids.shape[2]
        #
        unit_vector = torch.zeros(batch_size, seed_size, rigid_len, 3).to(
            device=rigids.device
        )  # [B, S, L, 3]
        unit_vector[:, :, :, 0] = 1
        #
        if not train_mode and not inference_mode:
            random_seed = 7
        else:
            random_seed = np.random.randint(99999)
        pert_trs = rand_rotation_matrix(
            batch_size * seed_size * rigid_len, 1.0, random_seed=random_seed
        )  # [B*S*L, 3, 3]
        pert_trs = pert_trs.view(
            batch_size, seed_size, rigid_len, 3, 3
        )  # [B, S, L, 3, 3]
        pert_trs = torch.einsum(
            "bslij,bslj->bsli", (pert_trs, unit_vector)
        )  # [S, L, 3]
        #
        pert_rot = rand_rotation_matrix(
            batch_size * seed_size * rigid_len,
            self.rot_pert,
            random_seed=(
                random_seed * 7 + 7
            ),  # preventing some possible coupling between translational &rotational perturbation
        )  # [B*S*L, 3, 3]
        pert_rot = pert_rot.view(
            batch_size, seed_size, rigid_len, 3, 3
        )  # [B,S, L, 3, 3]
        ##
        pert_rot_non_ulr = rand_rotation_matrix(
            batch_size * seed_size * rigid_len,
            self.rot_pert_non_ulr,
            random_seed=(random_seed * 7 + 7),
        )
        pert_rot_non_ulr = pert_rot_non_ulr.view(batch_size, seed_size, rigid_len, 3, 3)

        ulr_mask = ulr_mask.bool()  # [B, S, L]
        new_trs = rigids._trans  # [B, S, L, 3]
        tmp = new_trs.clone()
        tmp[ulr_mask] = new_trs[ulr_mask] + pert_trs[ulr_mask] * self.trans_pert
        if self.use_pert_non_ulr:
            tmp[~ulr_mask] = (
                new_trs[~ulr_mask] + pert_trs[~ulr_mask] * self.trans_pert_non_ulr
            )

        new_trs = tmp
        #

        inp_rot_mats = rigids._rots._rot_mats  # [B,S, L, 3, 3]
        tmp = inp_rot_mats.clone()
        tmp[ulr_mask] = pert_rot[ulr_mask].float()
        if self.use_pert_non_ulr:
            tmp[~ulr_mask] = rot_matmul(tmp[~ulr_mask], pert_rot_non_ulr[~ulr_mask])
        inp_rot_mats = tmp
        #
        new_rots = Rotation(rot_mats=inp_rot_mats)
        return Rigid(new_rots, new_trs)


def rbf(D, D_min=0.0, D_count=64, D_sigma=0.5):
    # Distance radial basis function
    # D = [B, L, L]
    D_max = D_min + (D_count - 1) * D_sigma  # 64
    D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)  # [64]
    # [0, 0.5, 1.0, 1.5, ..., 31.5]
    D_mu = D_mu[None, :]
    # [1, 64]
    D_expand = torch.unsqueeze(D, -1)
    # [B, L, L, 1]
    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
    # [B, L, L, 1] - [1, 64] = [B, L, L, 64]

    return RBF
