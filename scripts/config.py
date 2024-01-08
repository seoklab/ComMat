import copy
import importlib
import ml_collections as mlc


def set_inf(c, inf):
    for k, v in c.items():
        if isinstance(v, mlc.ConfigDict):
            set_inf(v, inf)
        elif k == "inf":
            c[k] = inf


def enforce_config_constraints(config):
    def string_to_setting(s):
        path = s.split(".")
        setting = config
        for p in path:
            setting = setting[p]

        return setting

    mutually_exclusive_bools = [
        ("model.template.average_templates", "model.template.offload_templates"),
        (
            "globals.use_lma",
            "globals.use_flash",
        ),
    ]

    for s1, s2 in mutually_exclusive_bools:
        s1_setting = string_to_setting(s1)
        s2_setting = string_to_setting(s2)
        if s1_setting and s2_setting:
            raise ValueError(f"Only one of {s1} and {s2} may be set at a time")

    fa_is_installed = importlib.util.find_spec("flash_attn") is not None
    if config.globals.use_flash and not fa_is_installed:
        raise ValueError("use_flash requires that FlashAttention is installed")

    if config.globals.offload_inference and not config.model.template.average_templates:
        config.model.template.offload_templates = True


def model_config(name, train=False, low_prec=False, long_sequence_inference=False):
    c = copy.deepcopy(config)
    # TRAINING PRESETS
    if train:
        c.globals.blocks_per_ckpt = 1
        # c.globals.chunk_size = None
        c.globals.use_lma = False
        c.globals.offload_inference = False
        c.model.template.average_templates = False
        c.model.template.offload_templates = False

    enforce_config_constraints(c)

    return c


###

seed_size = 32 

# Dataloader
N_CROP = 100
FIX_STRUCTURE_MODE = None
PRE_KABSCH = False
POST_KABSCH = False
GP_RATIO = -1
AG_REMOVE_PROB = 0.7
ADD_ULR_PROB = 0.5
BUILD_ALL_CDR = False

### Model
IPA_TYPE = ["default", "default", "default", "default"]
MIN_N_RECYCLE = 8
MAX_N_RECYCLE = 8
SAMPLED_N_RECYCLE = 8
USE_NON_ULR = False
USE_PERT_NON_ULR = False
USE_CROSS_OVER = True
USE_UPDATE_Z = True  # update z between IPA blocks (recycle)
USE_UPDATE_Z_IPA = False  # update z between IPA layers
#USE_UPDATE_Z_IPA = [True,'naive',True] # use_update_z_ipa, update_z_ipa_mode, return_updated_z
STOP_ROT_GRAD = False
USE_RESET_STRUCTURE = False
### Loss32USE_INTERMEDIATE_STR = False
USE_INTERMEDIATE_STR = False
USE_CUMUL_CHI = False
USE_BB_MASK = False
USING_DISTANCE_WEIGHT = False
EPS_CHI = 1e-3
EPS_RMSD = 1e-6
WEIGHT_SUPERVISED_CHI = 0.5
MASK_BB_CRITERIA = 3
MASK_LOCAL_BB_CRITERIA = 1
WEIGHT_VIOLATION = 1.0
WEIGHT_PLDDT = 0.1

ipa_c_hidden = 16
ipa_no_heads = 12
ipa_no_qk_points = 4
ipa_no_v_points = 8
ipa_no_blocks = 1
ipa_no_transition_layers = 1
ipa_enc_div = 2
###
dgram_min_bin = mlc.FieldReference(2.3125, field_type=float)
dgram_max_bin = mlc.FieldReference(21.6875, field_type=float)
dgram_no_bins = mlc.FieldReference(64, field_type=int)
c_z = mlc.FieldReference(96, field_type=int)
c_m = mlc.FieldReference(256, field_type=int)
c_s = mlc.FieldReference(128, field_type=int)
##
raw_2d_in = mlc.FieldReference(2 * c_s + 73, field_type=int)  # set depend on dgram
trans_scale_factor = mlc.FieldReference(10.0, field_type=float)  # set depend on dgram
#
#
blocks_per_ckpt = mlc.FieldReference(None, field_type=int)
chunk_size = mlc.FieldReference(4, field_type=int)
aux_distogram_bins = mlc.FieldReference(64, field_type=int)
tm_enabled = mlc.FieldReference(False, field_type=bool)
eps = mlc.FieldReference(1e-8, field_type=float)
templates_enabled = mlc.FieldReference(True, field_type=bool)
embed_template_torsion_angles = mlc.FieldReference(True, field_type=bool)
tune_chunk_size = mlc.FieldReference(True, field_type=bool)

NUM_RES = "num residues placeholder"
NUM_MSA_SEQ = "msa placeholder"
NUM_EXTRA_SEQ = "extra msa placeholder"
NUM_TEMPLATES = "num templates placeholder"

config = mlc.ConfigDict(
    {
        # Recurring FieldReferences that can be changed globally here
        "globals": {
            "blocks_per_ckpt": blocks_per_ckpt,
            # "chunk_size": chunk_size,
            "c_z": c_z,
            "c_m": c_m,
            "c_s": c_s,
            "eps": eps,
        },
        "dataloader": {
            "n_crop": N_CROP,
            "fix_structure_mode": FIX_STRUCTURE_MODE,  # crystal, IgFold,pertMD
            "fix_model_index": None,  # int
            "build_from_scratch": False,  # build with blackhole initializtion,
            "gp_ratio": GP_RATIO,
            "ag_remove_prob": AG_REMOVE_PROB,
            "add_ulr_prob": ADD_ULR_PROB,
            "using_pre_kabsch": PRE_KABSCH,
            "build_all_cdr": BUILD_ALL_CDR,
        },
        "model": {
            "use_no_recycle":False,
            "gradient_clip_value":None,
            "run_mode": None,
            "raw_2d_in": raw_2d_in,
            "seed_size": seed_size,
            "trans_scale_factor": trans_scale_factor,
            "no_recycle": 8,
            "sampled_n_recycle": SAMPLED_N_RECYCLE,
            "min_n_recycle": MIN_N_RECYCLE,
            "max_n_recycle": MAX_N_RECYCLE,
            "c_z": c_z,
            "use_cross_over": USE_CROSS_OVER,
            "cross_over_no_block": 1,
            "use_update_z": USE_UPDATE_Z,
            "InitialPertTrs": {
                "trans_pert": 0.3,
                "rot_pert": 1.0,
                "trans_pert_non_ulr": 0.1,
                "rot_pert_non_ulr": 0.3,
                "use_pert_non_ulr": USE_PERT_NON_ULR,
            },
            "using_post_kabsch": POST_KABSCH,
            "reset_structure": USE_RESET_STRUCTURE,
            "_mask_trans": False,
            "input_embedder": {
                "tf_dim": 22,
                "msa_dim": 49,
                "c_z": c_z,
                "c_m": c_m,
                "relpos_k": 32,
            },
            "recycling_embedder": {
                "c_z": c_z,
                "c_m": c_s,
                "min_bin": 3.25,
                "max_bin": 20.75,
                "no_bins": 15,
                "inf": 1e8,
            },
            "WorkingZ": {
                "c_z": c_z,
                "raw_2d_in": raw_2d_in,
                "dgram_min_bin": dgram_min_bin,
                "dgram_max_bin": dgram_max_bin,
                "dgram_no_bins": dgram_no_bins,
                "extra_activation": True,
                "bottle_neck": False,
                "add_prev": False,
                "rel_pos_dim": 32 * 2 + 1 + 1,
                "rel_pos_add": "cat",
                "c_in_tri_att": c_z,
                "c_hidden_tri_att": 16,
                "c_hidden_tri_mul": 64,
                "no_head": 4,
                "no_blocks": 2,
                "pair_transition_n": 2,
                "blocks_per_ckpt": 2,
                "chunk_size": 1,
                "use_tri_attn": USE_UPDATE_Z,
                "partial_use_tri_attn":True,
                "partial_use_tri_mul":True,
            },
            "seq_feat_embedder": {
                "tf_dim": 20,
                "out_dim": c_s,
                "use_torsion_mode": None,
                # "use_torsion_mode": "bb",
                "lang_model_stat": True,
                "lang_model_dim": 1280,
                "mode": "cat",
            },
            "bb_update": {"c_s": c_s},
            "pair_feat_embedder": {
                "tf_dim": 20,
                "c_z": c_z,
                "out_dim": 20,
                "max_rel_pos": 32,
            },
            "recycle_embedder": {
                "c_s": c_s,
                "c_z": c_z,
                "min_bin": 20,
                "max_bin": 20,
                "no_bins": 20,
                "use_dist_bin": True,
            },
            "IPA_block": {
                "c_s": c_s,
                "c_z": c_z,
                "c_hidden": ipa_c_hidden,
                "no_heads": ipa_no_heads,
                "no_qk_points": ipa_no_qk_points,
                "no_v_points": ipa_no_v_points,
                "no_blocks": ipa_no_blocks,
                "dropout_rate": 0.1,
                "no_transition_layers": ipa_no_transition_layers,
                "inf": 1e5,
                "use_cross_over": USE_CROSS_OVER,
                "cross_over_interval": 1,
                "use_update_z": USE_UPDATE_Z,
                "use_update_z_ipa": USE_UPDATE_Z_IPA,
                "build_str_interval": 1,
                "no_ipa_s": len(IPA_TYPE),
                "build_str_type": "torsion",  # torsion or frame
                "ipa_type_s": IPA_TYPE,
                "update_rigids": True,
                "use_non_ulr": USE_NON_ULR,
                "stop_rot_gradient": STOP_ROT_GRAD,
            },
            "IPA_enc": {
                "c_s": c_s,
                "c_z": c_z,
                "c_hidden": int(ipa_c_hidden / ipa_enc_div),
                "no_heads": int(ipa_no_heads / ipa_enc_div),
                "no_qk_points": int(ipa_no_qk_points / ipa_enc_div),
                "no_v_points": int(ipa_no_v_points / ipa_enc_div),
                "no_blocks": 1,
                "no_transition_layers": 2,
                "dropout_rate": 0.1,
                "inf": 1e5,
                "use_cross_over": False,
                "cross_over_interval": 2,
                "use_update_z": False,
                "no_ipa_s": 1,
                "ipa_type_s": ["default"],
                "update_rigids": False,
                "use_non_ulr": USE_NON_ULR,
            },
            "Cross_over_module": {
                "c_s": c_s,
                "c_z": c_z,
                "c_hidden": int(ipa_c_hidden / ipa_enc_div),
                "no_heads": int(ipa_no_heads / ipa_enc_div),
                "no_qk_points": int(ipa_no_qk_points / ipa_enc_div),
                "no_v_points": int(ipa_no_v_points / ipa_enc_div),
                "no_transition_layers": 1,
                "dropout_rate": 0.1,
                "use_gloabal_feature": False,
                "use_point_attention": False,
                "use_triangle_attention": False,
                "use_distance_bias":False,
                "point_attention_weight": 0.0,
                "use_non_ulr": True,
                "tri_attn_config":{
                    "c_t": c_s,
                    "c_hidden_tri_att": 16,
                    "c_hidden_tri_mul": 64,
                    "no_blocks": 2,
                    "no_heads": 4,
                    "pair_transition_n": 2,
                    "dropout_rate": 0.1,
                    "blocks_per_ckpt": 2,
                    "use_tri_attn": True,
                    "use_tri_mul":True,
                }
            },
            "Build_str": {
                "trans_scale_factor": trans_scale_factor,
                "angle_resnet": {
                    "c_in": c_s,
                    "c_hidden": c_s,
                    "no_blocks": 2,
                    "no_angles": 7,
                    "epsilon": eps,
                },
                "uniform_build_str":True,
            },
            "Build_str_all": {
                "trans_scale_factor": trans_scale_factor,
                "angle_resnet": {
                    "c_in": c_s,
                    "c_hidden": c_s,
                    "no_blocks": 2,
                    "no_angles": 7,
                    "no_blocks": 2,
                    "no_angles": 7,
                    "epsilon": eps,
                },
            },
            "heads": {
                "lddt": {
                    "no_bins": 50,
                    "c_in": c_s,
                    "c_hidden": 128,
                },
                "res_eng": {
                    "no_bins": 50,
                    "c_in": c_s,
                    "c_hidden": 128,
                },
                "distogram": {
                    "c_z": c_z,
                    "no_bins": aux_distogram_bins,
                },
                "masked_msa": {
                    "c_m": c_m,
                    "c_out": 23,
                },
                #"finetuning_freeze":{
                #    'enable_grad_plddt':True,
                #    'enable_grad_distogram':False,
                #    'enable_grad_res_eng':False,
                #},
            },
        },
    }
)
