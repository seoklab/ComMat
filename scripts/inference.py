import logging
import pathlib
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.multiprocessing
import os
import pandas as pd
from collections import defaultdict
from pathlib import Path
from openfold.utils.tensor_utils import dict_multimap

##
# from apex.optimizers import FusedAdam, FusedLAMB
####
from config import config as hu_config  # merge with af2 line
from af2_loss import AlphaFoldLoss

from qm9_inference import TestDataModule
from runtime import gpu_affinity
from arguments import PARSER
from runtime.loggers import LoggerCollection, DLLogger
from runtime.utils import (
    to_cuda,
    get_local_rank,
    seed_everything,
    increase_l2_fetch_granularity,
)

from h3xsemble.utils import *
from h3xsemble.model.H3xsembleModel import H3xsembleModule
from utils import *

torch.multiprocessing.set_sharing_strategy("file_system")


def load_state(
    model: nn.Module,
    path: pathlib.Path,
):
    """Loads model, optimizer and epoch states from path"""
    checkpoint = torch.load(
        str(path), map_location={"cuda:0": f"cuda:{get_local_rank()}"}
    )
    # checkpoint['scheduler_state_dict']['base_eta_max']=0.005
    checkpoint["scheduler_state_dict"]["gamma"] = 0.75
    model.load_state_dict(checkpoint["state_dict"])

    logging.info(f"Loaded checkpoint from {str(path)}")
    return checkpoint["epoch"]


def test_epoch(
    bench_set_tag,
    model,
    dataloader,
    loss_fn,
    # epoch_idx,
    local_rank,
    sampled_n_recycle,
    reset_structure,
    args,
):
    result_s = []
    for batch_idx, batch in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        unit="batch",
        # desc=f"{bench_set_tag} Epoch {epoch_idx}",
        disable=(local_rank != 0),
    ):
        input_dic = batch[0]
        tag = batch[1]
        target_mode = batch[2]
        input_dic = to_cuda(input_dic)
        input_dic["train_mode"] = False
        input_dic["inference_mode"] = True
        with torch.cuda.amp.autocast(enabled=args.amp):
            pred = model(
                input_dic,
                sampled_n_recycle=sampled_n_recycle,
                reset_structure=reset_structure,
            )
            if args.using_post_kabsch:
                pred["final_atom_positions"] = do_post_kabsch2(input_dic, pred, tag)
            loss, loss_dic = loss_fn(tag, pred, input_dic)
            if args.write_pdb:
                if not Path(f'{args.file_name}').exists():
                    Path(args.file_name).mkdir(exist_ok = True)
                get_post_prediction_pdb(
                    input_dic,
                    pred,
                    tag,
                    rmsd_s=loss_dic["obs_H3_rmsd"],
                    header=f'{args.file_name}/{bench_set_tag}',
                )
            result_s.append(loss_dic)
    result_s = dict_multimap(torch.stack, result_s)
    #total_shape = result_s["total_rmsd"].shape
    #result_s["total_rmsd"] = result_s["total_rmsd"].view(total_shape[0], total_shape[-1])
    result_s["top1_H3_rmsd_stack"] = result_s["top1_H3_rmsd"]
    result_s["best_H3_rmsd_stack"] = result_s["best_H3_rmsd"]
    for key in result_s.keys():
        if "H3_rmsd_stack" in key or "total_rmsd" in key:
            continue
        result_s[key] = result_s[key].mean()

    return result_s


def run_bench(
    bench_set_tag,
    model,
    dataloader,
    loss_fn,
    # epoch_idx,
    local_rank,
    world_size,
    sampled_n_recycle,
    reset_structure,
    args,
):
    print("@@@@@@@")
    with torch.no_grad():
        epoch_loss = test_epoch(
            bench_set_tag,
            model,
            dataloader,
            loss_fn,
            # epoch_idx,
            local_rank,
            sampled_n_recycle,
            reset_structure,
            args,
        )

    for loss_key in list(epoch_loss.keys()):
        if "H3_rmsd_stack" in loss_key:
            gathered_tensor = epoch_loss[loss_key]
            median_value = torch.median(gathered_tensor)
            total_values = gathered_tensor.numel()
            count_2_or_less = torch.sum(gathered_tensor <= 2.0).item() / total_values
            count_1_5_or_less = torch.sum(gathered_tensor <= 1.5).item() / total_values
            count_1_or_less = torch.sum(gathered_tensor <= 1.0).item() / total_values
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
            epoch_loss[loss_key] = epoch_loss[loss_key]

    return epoch_loss


def inference(
    model: nn.Module,
    loss_fn: nn.Module,
    benchloader_dic,
    sampled_n_recycle: int,
    reset_structure: bool,
    args,
):
    ##
    device = torch.cuda.current_device()
    model.to(device=device)
    local_rank = get_local_rank()
    world_size = 1
    epoch_start = 0
    if os.path.exists(args.epoch_test):
        epoch_start = load_state(model, args.epoch_test)

    model.eval()
    ###
    if args.file_name is None:
        args.file_name = args.epoch_test.split(".")[0]
    if Path(f"result_{args.file_name}.csv").exists():
        with open(f"result_{args.file_name}.csv", "r") as f:
            result_dict = pd.read_csv(f).to_dict(orient="list")
        print(result_dict)
    else:
        result_dict = defaultdict(list)
    print(result_dict)
    for bench_set_tag in benchloader_dic.keys():
        print(bench_set_tag)
        epoch_loss = run_bench(
            bench_set_tag,
            model,
            benchloader_dic[bench_set_tag],
            loss_fn,
            # epoch_idx,
            local_rank,
            world_size,
            sampled_n_recycle,
            reset_structure,
            args,
        )
        result_dict["bench_set"].append(bench_set_tag)
        result_dict["best_H3_rmsd"].append(epoch_loss["best_H3_rmsd"].cpu().item())
        result_dict["median_best_H3_rmsd"].append(
            epoch_loss["median_best_H3_rmsd"].cpu().item()
        )
        result_dict["top1_H3_rmsd"].append(epoch_loss["top1_H3_rmsd"].cpu().item())
        result_dict["median_top1_H3_rmsd"].append(
            epoch_loss["median_top1_H3_rmsd"].cpu().item()
        )
        result_dict["pair_rmsd"].append(epoch_loss["prmsd_loss"].cpu().item())
        result_dict["best_H3<2.0"].append(epoch_loss["best_H3<2.0"])
        result_dict["best_H3<1.5"].append(epoch_loss["best_H3<1.5"])
        result_dict["best_H3<1.0"].append(epoch_loss["best_H3<1.0"])
        result_dict["top1_H3<2.0"].append(epoch_loss["top1_H3<2.0"])
        result_dict["top1_H3<1.5"].append(epoch_loss["top1_H3<1.5"])
        result_dict["top1_H3<1.0"].append(epoch_loss["top1_H3<1.0"])

        result_dict["n_recycle"].append(sampled_n_recycle)
        result_dict["seed_size"].append(args.test_seed_size)
        result_dict["n_crop"].append(args.test_n_crop)
        test_ulr_type_str = "/".join([str(i) for i in args.test_ulr_type])
        result_dict["ulr_type"].append(test_ulr_type_str)
        result_dict["model_or_crystal"].append(args.test_structure_mode)

    df = pd.DataFrame.from_dict(result_dict)
    df.to_csv(f"result_{args.file_name}.csv", index=False)


def print_parameters_count(model):
    num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Number of trainable parameters: {num_params_trainable}")


if __name__ == "__main__":
    local_rank = get_local_rank()
    args = PARSER.parse_args()

    logging.getLogger().setLevel(
        logging.CRITICAL if local_rank != 0 or args.silent else logging.INFO
    )

    logging.info("======Loop_CSA======")
    logging.info("|      Inference procedure     |")
    logging.info("===============================")

    # if args.seed is not None:
    #     logging.info(f"Using seed {args.seed}")
    #     seed_everything(args.seed)

    logger = LoggerCollection(
        [DLLogger(save_dir=args.log_dir, filename=args.dllogger_name)]
    )

    hu_config.model.seed_size = args.test_seed_size
    hu_config.loss.seed_size = args.test_seed_size
    if "build_all_cdr" in hu_config.dataloader:
        build_all_cdr = hu_config.dataloader.build_all_cdr
    else:
        build_all_cdr = False
    bench_datamodule = TestDataModule(
        **vars(args),
        seed_size=args.test_seed_size,
        use_gp=True,
        use_chain_tag=True,
        use_multi_cdr=True,
        use_pert_crop_center=False,
        pert_crop_center=0.0,
        fix_tag_debug=None,
        fix_structure_mode=args.test_structure_mode,
        build_from_scratch=hu_config.dataloader.build_from_scratch,
        ulr_type=args.test_ulr_type,
        n_crop=args.test_n_crop,
        build_all_cdr = build_all_cdr,
    )

    benchloader_dic = bench_datamodule.get_benchloader_dic(sel_list=["ig", "ib", "test"])
    model = H3xsembleModule(hu_config.model)
    loss_fn = AlphaFoldLoss(hu_config.loss)
    print_parameters_count(model)
    logger.log_hyperparams(vars(args))
    increase_l2_fetch_granularity()

    inference(
        model,
        loss_fn,
        benchloader_dic,
        args.test_n_recycle,
        hu_config.model.reset_structure,
        args,
    )

    logging.info("Inference finished successfully")
