#!/bin/sh
##
##
#SBATCH -p gpu-super.q
#SBATCH --exclude=nova002,nova003,nova00[5-7],nova01[1-4]
#SBATCH --cpus-per-task=6
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --job-name=ep_100
#SBATCH --output=ep_100.log
#SBATCH --gres=gpu:1
export NCCL_P2P_DISABLE=1


python scripts/inference.py --batch_size 1 --weight community_size_32.pt --test_seed_size 32 \
    --test_pdb 7sn1_H_L_#.pdb --test_n_recycle 8 --test_n_crop 100 --test_ulr_type H_3 \
    --file_name test_0108 --using_post_kabsch --write_pdb
