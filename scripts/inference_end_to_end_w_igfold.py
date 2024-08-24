import logging
import pathlib
import torch
import torch.nn as nn
import torch.multiprocessing
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict
from featurizing import prep_af2_inp
from align_to_modeled import get_fasta_dict, get_new_pdb

##
# from apex.optimizers import FusedAdam, FusedLAMB
####
from config import config as hu_config  # merge with af2 line

# from af2_loss import AlphaFoldLoss
from preprocess import data_preprocess_temp
from arguments import PARSER
from runtime.loggers import LoggerCollection, DLLogger
from runtime.utils import (
    to_cuda,
    get_local_rank,
    seed_everything,
    increase_l2_fetch_granularity,
)

from igfold import IgFoldRunner
from h3xsemble.model.H3xsembleModel import H3xsembleModule
from utils import (
    do_post_kabsch2,
    get_post_prediction_pdb,
    get_post_prediction_pdb_ranked,
)
from align_to_modeled import write_aligned_pdb
from local_optimize import run_input_multiple, run_input_indiv

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


def inference(
    model: nn.Module,
    benchloader_dic,
    sampled_n_recycle: int,
    reset_structure: bool,
    args,
):
    ##
    device = torch.cuda.current_device()
    model.to(device=device)
    epoch_start = 0
    print(args.weight)
    path_weight = pathlib.Path(__file__).parent.parent.resolve()
    path_weight = path_weight / "weights" / Path(args.weight)
    print(path_weight)
    if Path.exists(path_weight):
        epoch_start = load_state(model, path_weight)
    if not Path.exists(path_weight):
        print("weight load failed!!!")
    model.eval()

    input_dic, tag, target_mode = benchloader_dic
    print(tag)
    print(target_mode)
    with torch.no_grad():
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
            if args.write_pdb:
                if not Path(f"{args.output_path}").exists():
                    Path(args.output_path).mkdir(exist_ok=True)
                get_post_prediction_pdb_ranked(
                    input_dic,
                    pred,
                    tag_s=None,
                    header=f"{args.output_path}/{tag}",
                )


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

    if args.seed is not None:
        logging.info(f"Using seed {args.seed}")
        seed_everything(args.seed)

    logger = LoggerCollection(
        [DLLogger(save_dir=args.log_dir, filename=args.dllogger_name)]
    )

    hu_config.model.seed_size = args.test_seed_size
    # hu_config.loss.seed_size = args.test_seed_size
    if "build_all_cdr" in hu_config.dataloader:
        build_all_cdr = hu_config.dataloader.build_all_cdr
    else:
        build_all_cdr = False

    ### if no test_pdb structure is given, IgFold will be used to generate structure ###

    if args.test_pdb is not None:
        test_pdb = args.test_pdb
    else:  # Get IgFold structure
        fasta_file = Path(args.fasta_path)
        output_folder = args.output_path
        sequences: DefaultDict[str, str] = defaultdict(str)
        with open(fasta_file) as f_out:
            lines = f_out.readlines()
            for line in lines:
                if line.startswith(">"):
                    chain = line[1:].strip()
                else:
                    sequences[chain] += line.strip()
        pdbname = Path(args.fasta_path).stem
        if not Path(output_folder).exists():
            Path(output_folder).mkdir(exist_ok=True)
        igfold = IgFoldRunner()
        igfold.fold(
            f"{output_folder}/{pdbname}_igfold.pdb",
            sequences=sequences,
            do_refine=False,
            do_renum=False,
        )
        hchain_length = len(
            get_fasta_dict(f"{output_folder}/{pdbname}_igfold.fasta")["H"]
        )
        new_pdb = get_new_pdb(f"{output_folder}/{pdbname}_igfold.pdb", hchain_length)
        with open(f"{output_folder}/igfold_renumbered.pdb", "w") as f_out:
            f_out.writelines(new_pdb)
        test_pdb = f"{output_folder}/{pdbname}_igfold.pdb"

    prep_af2_inp(test_pdb, pdbname, output_path=output_folder)
    #### TODO: Make data processing part ###
    pdbname = Path(args.fasta_path).stem
    input_dic, tag, mode = data_preprocess_temp(
        pdbname,
        output_folder,
        ag_remove=True,
        seed_size=args.test_seed_size,
        trans_scale_factor=10.0,
        n_crop=args.test_n_crop,
        build_from_scratch=False,
    )

    model = H3xsembleModule(hu_config.model)
    #    loss_fn = AlphaFoldLoss(hu_config.loss)
    print_parameters_count(model)
    logger.log_hyperparams(vars(args))
    increase_l2_fetch_granularity()

    inference(
        model,
        [input_dic, tag, mode],
        args.test_n_recycle,
        hu_config.model.reset_structure,
        args,
    )

    ### Align predicted structure to IgFold structure ###

    igfold_file = f"{output_folder}/igfold_renumbered.pdb"
    commat_file = f"{output_folder}/{tag}.pdb"
    write_aligned_pdb(
        pdbname,
        commat_file,
        igfold_file,
        output_folder,
    )

    ### relax with GalaxyLocalOptimize ###
    seed_size = args.test_seed_size
    for i in range(seed_size):
        run_input_indiv(pdbname, i, output_folder)
    # run_input_multiple(pdbname, output_folder)

    ### ranking with AF2Rank ###
    #os.system('python3 scripts/0424_get_tmscore.py --name {pdbname} --chain H,L --decoy_dir {output_folder}/relaxed --model_num 5')

    logging.info("Inference finished successfully")
