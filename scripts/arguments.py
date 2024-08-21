import argparse
import pathlib

from runtime.utils import str2bool

PARSER = argparse.ArgumentParser(description="HU_AF2")

paths = PARSER.add_argument_group("Paths")
paths.add_argument(
    "--data_dir",
    type=pathlib.Path,
    default=pathlib.Path("./data"),
    help="Directory where the data is located or should be downloaded",
)
paths.add_argument(
    "--log_dir",
    type=pathlib.Path,
    default=pathlib.Path("./results"),
    help="Directory where the results logs should be saved",
)
paths.add_argument(
    "--dllogger_name",
    type=str,
    default="dllogger_results.json",
    help="Name for the resulting DLLogger JSON file",
)
paths.add_argument(
    "--param_name",
    type=str,
    default="hmmm",
    help="Name for the resulting parameter_file",
)
paths.add_argument(
    "--sch_type",
    type=str,
    default="cosine",
    help="Name for the resulting parameter_file",
)
paths.add_argument(
    "--save_ckpt_path",
    type=pathlib.Path,
    default=None,
    help="File where the checkpoint should be saved",
)
paths.add_argument(
    "--load_ckpt_path",
    type=pathlib.Path,
    default=None,
    help="File of the checkpoint to be loaded",
)

optimizer = PARSER.add_argument_group("Optimizer")
optimizer.add_argument("--optimizer", choices=["adam", "sgd", "lamb"], default="adam")
optimizer.add_argument(
    "--learning_rate", "--lr", dest="learning_rate", type=float, default=0.002
)
optimizer.add_argument(
    "--min_learning_rate",
    "--min_lr",
    dest="min_learning_rate",
    type=float,
    default=None,
)
optimizer.add_argument(
    "--max_lr", "--max_lr", dest="max_learning_rate", type=float, default=0.1
)
optimizer.add_argument("--t_up", "--t_up", dest="warm_up_cycle", type=int, default=10)
optimizer.add_argument("--t_0", "--t_0", dest="cycle_period", type=int, default=100)
optimizer.add_argument("--momentum", type=float, default=0.9)
optimizer.add_argument("--weight_decay", type=float, default=0.01)

PARSER.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
PARSER.add_argument("--batch_size", type=int, default=1, help="Batch size")
PARSER.add_argument("--seed", type=int, default=None, help="Set a seed globally")
PARSER.add_argument(
    "--num_workers", type=int, default=6, help="Number of dataloading workers"
)

PARSER.add_argument(
    "--amp",
    type=str2bool,
    nargs="?",
    const=True,
    default=False,
    help="Use Automatic Mixed Precision",
)
PARSER.add_argument(
    "--gradient_clip", type=float, default=None, help="Clipping of the gradient norms"
)
PARSER.add_argument(
    "--accumulate_grad_batches", type=int, default=1, help="Gradient accumulation"
)
PARSER.add_argument(
    "--ckpt_interval", type=int, default=-1, help="Save a checkpoint every N epochs"
)
PARSER.add_argument(
    "--eval_interval",
    dest="eval_interval",
    type=int,
    default=2,
    help="Do an evaluation round every N epochs",
)
PARSER.add_argument(
    "--silent",
    type=str2bool,
    nargs="?",
    const=True,
    default=False,
    help="Minimize stdout output",
)
PARSER.add_argument(
    "--wandb",
    type=str2bool,
    nargs="?",
    const=True,
    default=False,
    help="Enable W&B logging",
)
PARSER.add_argument(
    "--wandb_name", type=str, default=False, help="Name of WanDB logging"
)
PARSER.add_argument(
    "--benchmark",
    type=str2bool,
    nargs="?",
    const=True,
    default=False,
    help="Benchmark mode",
)
PARSER.add_argument("--val_fn_path", type=str, nargs="?", default=None)
PARSER.add_argument("--out_data_tag", type=str, nargs="?", default=None)
PARSER.add_argument(
    "--use_rsa_feat", type=str2bool, nargs="?", const=True, default=False
)
PARSER.add_argument(
    "--use_l0_aux_loss", type=str2bool, nargs="?", const=True, default=False
)
PARSER.add_argument(
    "--use_l1_loss", type=str2bool, nargs="?", const=True, default=False
)
PARSER.add_argument(
    "--use_subunit_act", type=str2bool, nargs="?", const=True, default=False
)
PARSER.add_argument(
    "--use_subunit_act_aux_loss", type=str2bool, nargs="?", const=True, default=False
)
PARSER.add_argument("--graph_method", type=str, nargs="?", const=True, default="hu")
PARSER.add_argument("--aux_type", type=str, nargs="?", const=True, default="hu")


## inference mode!!
PARSER.add_argument(
    "--write_pdb",
    type=str2bool,
    nargs="?",
    const=True,
    default=False,
    help="write_pdb with test_epoch",
)

PARSER.add_argument(
    "--weight", type=str, default="last.pt", help="weight to use in inference"
)

PARSER.add_argument(
    "--test_structure_mode", type=str, default="crystal", help="test_structure_mode"
)

PARSER.add_argument(
    "--test_seed_size", type=int, default=32, help="seed_size for test_mode"
)
PARSER.add_argument(
    "--test_n_recycle", type=int, default=8, help="n_recycle for test_mode"
)
PARSER.add_argument("--test_n_crop", type=int, default=100, help="n_crop for test_mode")
PARSER.add_argument(
    "--test_ulr_type",
    type=str,
    nargs="+",
    default=["H_3"],
    help="ulr_types for test_mode",
)
PARSER.add_argument(
    "--test_pdb", type=str, help="chothia-numbered antibody PDB to inference"
)
PARSER.add_argument(
    "--output_path", type=str, default=None, help="path for output files"
)
PARSER.add_argument(
    "--using_post_kabsch",
    type=str2bool,
    nargs="?",
    const=True,
    default=False,
    help="using target post kabsch",
)
