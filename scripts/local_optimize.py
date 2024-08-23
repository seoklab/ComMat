import os
from pathlib import Path


def run_input_indiv(pdbname, num, output_folder):
    Path(f"{output_folder}/relaxed").mkdir(exist_ok=True)
    Path(f"{output_folder}/relaxation_scripts").mkdir(exist_ok=True)
    input_line = """data_directory        /applic/Galaxy/current/data
top_type              polarh
weight_type           Prot2016
print_level           30
energy_print_level    30
read_multiple_models  yes
infile_pdb            @P
outfile_prefix        @O
energy_on_full        yes
read_multiple_models  yes
local_optimize_relax     yes
local_optimize_sidechain no """

    input_line = input_line.replace(
        "@P", f"{output_folder}/unrelaxed/{pdbname}_{num}.pdb"
    )
    input_line = input_line.replace(
        "@O", f"{output_folder}/relaxed/{pdbname}_relaxed_{num}"
    )
    with open(
        f"{output_folder}/relaxation_scripts/local_optimize_{num}.in", "w"
    ) as f_out:
        f_out.write(input_line)
    os.system(
        f"mpiexec src/galaxylocalopt/bin/local_optimize.mpi {output_folder}/relaxation_scripts/local_optimize_{num}.in"
    )


def run_input_multiple(pdbname, output_folder):
    input_line = """data_directory        /home/yubeen/ComMat/src/galaxylocalopt/data
top_type              polarh
weight_type           Prot2016
print_level           30
energy_print_level    30
read_multiple_models  yes
infile_pdb            @P
outfile_prefix        @O
energy_on_full        yes
read_multiple_models  yes
local_optimize_relax     yes
local_optimize_sidechain no """

    input_line = input_line.replace(
        "@P", f"{output_folder}/{pdbname}_multiple_aligned.pdb"
    )
    input_line = input_line.replace("@O", f"{output_folder}/{pdbname}_relaxed")
    with open(f"{output_folder}/local_optimize.in", "w") as f_out:
        f_out.write(input_line)
    os.system(
        f"mpiexec src/galaxylocalopt/bin/local_optimize.mpi {output_folder}/local_optimize.in"
    )
