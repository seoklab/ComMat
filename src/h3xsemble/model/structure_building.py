import torch
import torch.nn as nn
from h3xsemble.model.structure_module import AngleResnet, AngleResnet_backbone
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
from openfold.utils.rigid_utils import Rotation, Rigid


class Build_str(nn.Module):
    def __init__(self, config):
        super(Build_str, self).__init__()
        self.trans_scale_factor = config["trans_scale_factor"]
        self.angle_resnet = AngleResnet(**config["angle_resnet"])

    def forward(self, s, rigids, s_initial, aatype):
        backb_to_global = Rigid(
            Rotation(rot_mats=rigids.get_rots().get_rot_mats(), quats=None),
            rigids.get_trans(),
        )
        aatype = aatype.long()
        backb_to_global = backb_to_global.scale_translation(self.trans_scale_factor)
        #
        unnormalized_angles, angles = self.angle_resnet(s, s_initial)
        all_frames_to_global = self.torsion_angles_to_frames(
            backb_to_global,
            angles,
            aatype,
        )

        pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
            all_frames_to_global,
            aatype,
        )

        scaled_rigids = rigids.scale_translation(self.trans_scale_factor)

        preds = {
            "frames": scaled_rigids.to_tensor_7(),
            "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
            "unnormalized_angles": unnormalized_angles,
            "angles": angles,
            "positions": pred_xyz,
            "states": s,
        }
        return preds

    def _init_residue_constants(self, float_dtype, device):
        if not hasattr(self, "default_frames"):
            self.register_buffer(
                "default_frames",
                torch.tensor(
                    restype_rigid_group_default_frame,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "group_idx"):
            self.register_buffer(
                "group_idx",
                torch.tensor(
                    restype_atom14_to_rigid_group,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "atom_mask"):
            self.register_buffer(
                "atom_mask",
                torch.tensor(
                    restype_atom14_mask,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                torch.tensor(
                    restype_atom14_rigid_group_positions,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )

    def torsion_angles_to_frames(self, r, alpha, f):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype, alpha.device)
        # Separated purely to make testing less annoying
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(
        self, r, f  # [*, N, 8]  # [*, N]
    ):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(r.get_rots().dtype, r.get_rots().device)
        return frames_and_literature_positions_to_atom14_pos(
            r,
            f,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )


# mask : mask[(~ulr&~mask)] = False


class Build_str_all_frame(nn.Module):
    def __init__(self, config):
        super(Build_str, self).__init__()
        self.trans_scale_factor = config["trans_scale_factor"]
        self.angle_resnet = AngleResnet_backbone(**config["angle_resnet_backbone"])

    def forward(self, s, rigids, s_initial, aatype):
        backb_to_global = Rigid(
            Rotation(rot_mats=rigids.get_rots().get_rot_mats(), quats=None),
            rigids.get_trans(),
        )
        aatype = aatype.long()
        backb_to_global = backb_to_global.scale_translation(self.trans_scale_factor)
        #
        unnormalized_angles, angles = self.angle_resnet(s, s_initial)
        all_frames_to_global = self.torsion_angles_to_frames(
            backb_to_global,
            angles,
            aatype,
        )
        print("all_frames_to_global", all_frames_to_global.shape)
        print("aatype", aatype.shape)
        pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
            all_frames_to_global,
            aatype,
        )

        scaled_rigids = rigids.scale_translation(self.trans_scale_factor)

        preds = {
            "frames": scaled_rigids.to_tensor_7(),
            "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
            "unnormalized_angles": unnormalized_angles,
            "angles": angles,
            "positions": pred_xyz,
            "states": s,
        }
        return preds

    def _init_residue_constants(self, float_dtype, device):
        if not hasattr(self, "default_frames"):
            self.register_buffer(
                "default_frames",
                torch.tensor(
                    restype_rigid_group_default_frame,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "group_idx"):
            self.register_buffer(
                "group_idx",
                torch.tensor(
                    restype_atom14_to_rigid_group,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "atom_mask"):
            self.register_buffer(
                "atom_mask",
                torch.tensor(
                    restype_atom14_mask,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                torch.tensor(
                    restype_atom14_rigid_group_positions,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )

    def torsion_angles_to_frames(self, r, alpha, f):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype, alpha.device)
        # Separated purely to make testing less annoying
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(
        self, r, f  # [*, N, 8]  # [*, N]
    ):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(r.get_rots().dtype, r.get_rots().device)
        return frames_and_literature_positions_to_atom14_pos(
            r,
            f,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )
