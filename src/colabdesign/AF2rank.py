#!/usr/bin/env python3

#@title import libraries
import warnings
import sys
import pickle
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.append('/home/yubeen/code/ColabDesign')
from colabdesign import clear_mem, mk_af_model
from colabdesign.shared.utils import copy_dict

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import jax
from pathlib import Path

ref_dest = '/home/yubeen/commat_revision/features'

def tmscore(x,y, pdbname, model_name, types, which):
  # save to dumpy pdb files
  pdb_name = []
  for n, z in enumerate([x, y]):
      which_type = which[n]
      pdb_name.append(f"temp/{pdbname}_{model_name}_{types}_{which_type}.pdb")
  for n,z in enumerate([x,y]):
    out = open(pdb_name[n], 'w')
    for k,c in enumerate(z):
      out.write("ATOM  %5d  %-2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.2f\n"
                  % (k+1,"CA","ALA","A",k+1,c[0],c[1],c[2],1,0))
    out.close()
  # pass to TMscore
  output = os.popen(f'TMscore {pdb_name[0]} {pdb_name[1]}')

  # parse outputs
  parse_float = lambda x: float(x.split("=")[1].split()[0])
  o = {}
  for line in output:
    line = line.rstrip()
    if line.startswith("RMSD"): o["rms"] = parse_float(line)
    if line.startswith("TM-score"): o["tms"] = parse_float(line)
    if line.startswith("GDT-TS-score"): o["gdt"] = parse_float(line)

  os.system(f'rm {pdb_name[0]}')
  os.system(f'rm {pdb_name[1]}')
  return o

def plot_me(name, scores, x="tm_i", y="composite",
            title=None, diag=False, scale_axis=False, dpi=100, **kwargs):
  def rescale(a,amin=None,amax=None):
    a = np.copy(a)
    if amin is None: amin = a.min()
    if amax is None: amax = a.max()
    a[a < amin] = amin
    a[a > amax] = amax
    return (a - amin)/(amax - amin)

  plt.figure(figsize=(5,5), dpi=dpi)
  if title is not None: plt.title(title)
  x_vals = np.array([k[x] for k in scores])
  y_vals = np.array([k[y] for k in scores])
  c = rescale(np.array([k["plddt"] for k in scores]),0.5,0.9)
  plt.scatter(x_vals, y_vals, c=c*0.75, s=5, vmin=0, vmax=1, cmap="gist_rainbow",
              **kwargs)
  min_vals = min(min(x_vals), min(y_vals))
  max_vals = max(max(x_vals), max(y_vals))
  if diag:
    plt.plot([0,1],[0,1],color="black")

  labels = {"tm_i":"TMscore of Input",
            "tm_o":"TMscore of Output",
            "tm_io":"TMscore between Input and Output",
            "ptm":"Predicted TMscore (pTM)",
            "i_ptm":"Predicted interface TMscore (ipTM)",
            "plddt":"Predicted LDDT (pLDDT)",
            "composite":"Composite"}

  plt.xlabel(labels.get(x,x));  plt.ylabel(labels.get(y,y))
  if scale_axis:
    if x in labels: plt.xlim(-0.1, 1.1)
    if y in labels: plt.ylim(-0.1, 1.1)

  print(spearmanr(x_vals,y_vals).correlation)
  plt.savefig(f'{name}_correlation.png')

class af2rank:
  def __init__(self, pdb, pdbname, chain=None, model_name="model_1_ptm", model_names=None):
    self.args = {"pdb":pdb, "pdbname": pdbname, "chain":chain,
                 "use_multimer":("multimer" in model_name),
                 "model_name":model_name,
                 "model_names":model_names}
    self.reset()

  def reset(self):
    self.model = mk_af_model(protocol="fixbb",
                             use_templates=True,
                             use_multimer=self.args["use_multimer"],
                             debug=False,
                             model_names=self.args["model_names"], 
                             data_dir = '/applic/AlphaFold/current/data')

    self.model.prep_inputs(self.args["pdb"], chain=self.args["chain"])
    self.model.set_seq(mode="wildtype")
    self.wt_batch = copy_dict(self.model._inputs["batch"])
    self.wt = self.model._wt_aatype

  def set_pdb(self, pdb, chain=None):
    if chain is None: chain = self.args["chain"]
    self.model.prep_inputs(pdb, chain=chain)
    self.model.set_seq(mode="wildtype")
    self.wt = self.model._wt_aatype

  def set_seq(self, seq):
    self.model.set_seq(seq=seq)
    self.wt = self.model._params["seq"][0].argmax(-1)

  def _get_score(self, model_name, types):
    score = copy_dict(self.model.aux["log"])

    score["plddt"] = score["plddt"]
    print(score["plddt"])
    print('loop_plddt', score['plddt_loop'])
    score["pae"] = 31.0 * score["pae"]
    score["pae_loop"] = 31.0 * score["pae_loop"]
    score["rmsd_io"] = score.pop("rmsd",None)

    i_xyz = self.model._inputs["batch"]["all_atom_positions"][:,1]
    o_xyz = np.array(self.model.aux["atom_positions"][:,1])

    # TMscore to input/output
    if hasattr(self,"wt_batch"):
      n_xyz = self.wt_batch["all_atom_positions"][:,1]
      score["tm_i"] = tmscore(n_xyz,i_xyz, self.args["pdbname"], model_name, types, which=['native', 'input'])["tms"]
      score["tm_o"] = tmscore(n_xyz,o_xyz, self.args["pdbname"], model_name, types, which = ['native', 'output'])["tms"]

    # TMscore between input and output
    score["tm_io"] = tmscore(i_xyz,o_xyz, self.args["pdbname"], model_name, types, which= ['input', 'output'])["tms"]

    # composite score
    score["composite"] = score["ptm"] * score["plddt"] * score["tm_io"]
    return score

  def predict(self, pdb=None, seq=None, chain=None,
              input_template=True, model_name=None,
              rm_seq=True, rm_sc=True, rm_ic=False,
              recycles=1, iterations=1,
              output_pdb=None, extras=None, verbose=True, types=None):

    if model_name is not None:
      self.args["model_name"] = model_name
      if "multimer" in model_name:
        if not self.args["use_multimer"]:
          self.args["use_multimer"] = True
          self.reset()
      else:
        if self.args["use_multimer"]:
          self.args["use_multimer"] = False
          self.reset()

    print(self.args)
    if pdb is not None: self.set_pdb(pdb, chain)
    if seq is not None: self.set_seq(seq)

    # set template sequence
    self.model._inputs["batch"]["aatype"] = self.wt

    # set other options
    self.model.set_opt(
        template=dict(rm_ic=rm_ic),
        num_recycles=recycles)
    self.model._inputs["rm_template"][:] = not input_template
    self.model._inputs["rm_template_sc"][:] = rm_sc
    self.model._inputs["rm_template_seq"][:] = rm_seq

    # get h3 loop masks
    pdbname = self.args['pdbname']
    feature_file = Path(f'/home/yubeen/commat_revision/af2rank/h3index/{pdbname}_h3loop.npy')
    
    with open(feature_file, 'rb') as f_in:
        h3_mask = jax.numpy.load(f_in)

    sequence_length = len(self.model._pdb['batch']['aatype'])
    self.model._inputs['h3_mask'] = h3_mask[:sequence_length]

    # "manual" recycles using templates
    ini_atoms = self.model._inputs["batch"]["all_atom_positions"].copy()
    for i in range(iterations):
      self.model.predict(models=self.args["model_name"], verbose=False)
      if i < iterations - 1:
        self.model._inputs["batch"]["all_atom_positions"] = self.model.aux["atom_positions"]
      else:
        self.model._inputs["batch"]["all_atom_positions"] = ini_atoms

    score = self._get_score(self.args["model_name"], types)
    if extras is not None:
      score.update(extras)

    if output_pdb is not None:
      self.model.save_pdb(output_pdb)

    if verbose:
      print_list = ["tm_i","tm_o","tm_io","composite","ptm","i_ptm","plddt","fitness","id"]
      print_score = lambda k: f"{k} {score[k]:.4f}" if isinstance(score[k],float) else f"{k} {score[k]}"
      print(*[print_score(k) for k in print_list if k in score])

    return score
