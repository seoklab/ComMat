from collections import defaultdict
from pathlib import Path


def get_fasta_dict(fasta_file):
    seq_dict = defaultdict(str)
    with open(fasta_file) as f_fasta:
        lines = f_fasta.readlines()
        for line in lines:
            if line.startswith(">"):
                chain = line[1:].strip()
            else:
                seq_dict[chain] += line.strip()
    return seq_dict


def align_commat_to_igfold(pdbname, commat_file, igfold_file):
    pdb, hchain, lchain, agchain = pdbname.split("_")

    commat_coord = {}
    f_commat = open(commat_file)
    commat_lines = f_commat.readlines()

    if hchain > lchain:
        chain_match = {"A": hchain, "B": lchain}
    else:
        chain_match = {"A": hchain, "B": lchain}

    commat_coord_one_model = defaultdict(lambda: defaultdict(list))
    for line in commat_lines:
        if line.startswith("MODEL"):
            model_num = int(line.strip().split()[-1])
            if model_num > 1:
                commat_coord[model_num - 1] = commat_coord_one_model
            commat_coord_one_model = defaultdict(lambda: defaultdict(list))
        else:
            if line.startswith("ATOM"):
                chain = line[21]
                resno = int(line[22:26].strip())
                new_chain = chain_match[chain]
                new_line = f"{line[:21]}{new_chain}{line[22:]}"
                if not new_line.endswith("\n"):
                    new_line += "\n"
                commat_coord_one_model[new_chain][resno].append(new_line)
    commat_coord[model_num] = commat_coord_one_model

    igfold_coord = defaultdict(lambda: defaultdict(list))
    # chain_match_igfold = {'H':hchain, 'L':lchain}
    f_igfold = open(igfold_file)
    igfold_lines = f_igfold.readlines()
    for line in igfold_lines:
        if line.startswith("ATOM"):
            chain = line[21]
            resno = int(line[22:26].strip())
            if not line.endswith("\n"):
                line += "\n"
            igfold_coord[chain][resno].append(line)

    commat_aligned_coord = {}
    for k, v in commat_coord.items():  # MODEL
        commat_aligned_mono = defaultdict(lambda: defaultdict(list))
        for k1, v1 in v.items():  # CHAIN
            length = len(igfold_coord[k1])
            for i in range(1, length + 1):
                if i in v1:
                    commat_aligned_mono[k1][i] = v1[i]
                else:
                    commat_aligned_mono[k1][i] = igfold_coord[k1][i]
        commat_aligned_coord[k] = commat_aligned_mono

    return commat_aligned_coord


def write_aligned_pdb(pdbname, commat_file, igfold_file, output_folder):
    commat_aligned_coord = align_commat_to_igfold(pdbname, commat_file, igfold_file)

    # for k, v in commat_aligned_coord.items():  # MODEL No
    #     Path(pdbname).mkdir(exist_ok=True)
    #     with open(f"{pdbname}/{pdbname}_{k}.pdb", "w") as f_out:
    #         for ch, v1 in v.items():
    #             for resno, v2 in v1.items():
    #                 f_out.writelines(v2)
    #             f_out.write("TER\n")
    with open(f"{output_folder}/{pdbname}_multiple_aligned.pdb", "w") as f_out:
        for k, v in commat_aligned_coord.items():
            f_out.write(f"MODEL {k}\n")
            for ch, v1 in v.items():
                for resno, v2 in v1.items():
                    f_out.writelines(v2)
            f_out.write("TER\n")
            f_out.write("END\n")
            f_out.write("ENDMDL\n")
