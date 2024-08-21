#!/usr/bin/env python3

import os
import argparse
import json
import pandas as pd
from AF2rank import af2rank
from AF2rank import plot_me
from collections import defaultdict
from pathlib import Path

def get_af2rank(name, CHAIN, NATIVE_PATH, DECOY_DIR, recycles=1, iterations=1, model_num=1, model_mode="alphafold-multimer"):
    #@markdown ### **settings**
    recycles = 1 #@param ["0", "1", "2", "3", "4"] {type:"raw"}
    iterations = 1

    # decide what model to use
    model_mode = "alphafold-multimer" #@param ["alphafold", "alphafold-multimer"]
    model_num = model_num #@param ["1", "2", "3", "4", "5"] {type:"raw"}

    if model_mode == "alphafold":
      model_name = f"model_{model_num}_ptm"
    if model_mode == "alphafold-multimer":
      model_name = f"model_{model_num}_multimer_v2"

    print(model_name)
    save_output_pdbs = False #@param {type:"boolean"}

    #@markdown ### **advanced**
    mask_sequence = True #@param {type:"boolean"}
    mask_sidechains = True #@param {type:"boolean"}
    mask_interchain = False #@param {type:"boolean"}

    SETTINGS = {"rm_seq":mask_sequence,
                "rm_sc":mask_sidechains,
                "rm_ic":mask_interchain,
                "recycles":int(recycles),
                "iterations":int(iterations),
                "model_name":model_name}

    if save_output_pdbs:
        os.makedirs(f'{NAME}_output', ok_exists=True)

    af = af2rank(NATIVE_PATH, name, CHAIN, model_name=SETTINGS["model_name"])

    score_dict = {}
    SCORES = []
    # score native structure
    SCORES.append(af.predict(pdb=NATIVE_PATH, **SETTINGS, extras={"id":NATIVE_PATH}, types='native'))

    # score the decoy sctructures
    for decoy_pdb in os.listdir(DECOY_DIR):
      idx = Path(decoy_pdb).stem.split('_')[-1]
      idx = int(idx) - 1
      input_pdb = os.path.join(DECOY_DIR, decoy_pdb)
      if save_output_pdbs:
        output_pdb = os.path.join(f"{NAME}_output",decoy_pdb)
      else:
        output_pdb = None
      value = af.predict(pdb=input_pdb, output_pdb=output_pdb, **SETTINGS, extras={"id":decoy_pdb}, types=idx)
      SCORES.append(value)
      score_dict[idx] = value
    return SCORES, score_dict

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='rank structure with AF2')

    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--chain', type=str, required=True)
    parser.add_argument('--native_path', type=str, required=True)
    parser.add_argument('--decoy_dir', type=str, required=True)
    parser.add_argument('--model_num', type=int, required=True)
    args = parser.parse_args()

    scores, scores_dict = get_af2rank(args.name, args.chain, args.native_path, args.decoy_dir, model_num = args.model_num)
    f_out = open(f'{args.name}/result_v2_{args.model_num}.json', 'w')
    json.dump(scores_dict, f_out)
    #pdbname = args.name
    #h3_rmsd = defaultdict(float)
    #df = pd.read_csv('/home/yubeen/commat_revision/analyze/rmsd_commat_refine_re.csv')
    #for index, row in df.iterrows():
    #    if row['name'] == pdbname:
    #        for i in range(32):
    #            scores_dict[f'rmsd_{i}'] = row[f'score_{i}']


    #plot_me(args.name, scores, x="tm_i", y="composite",
    #    title=f"{args.name}: ranking INPUT decoys using composite score")
