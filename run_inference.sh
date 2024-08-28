#!/bin/sh
##
##
#SBATCH -p gpu-super.q
#SBATCH --exclude=nova002,nova003,nova00[5-7],nova01[1-4]
#SBATCH --cpus-per-task=6
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --job-name=ep_100
#SBATCH --output=output.log
#SBATCH --gres=gpu:1
export NCCL_P2P_DISABLE=1


python scripts/inference_end_to_end_w_igfold.py --batch_size 1 --weight community_size_32.pt --test_seed_size 32 \
    --fasta_path 7sn1_H_L_#.fasta --test_n_recycle 8 --test_n_crop 100 --test_ulr_type H_3 \
    --output_path test_output_final --using_post_kabsch --write_pdb --local_optimize --localopt_data_path \
    src/galaxylocalopt/data --localopt_exec_path src/galaxylocalopt/bin/local_optimize
