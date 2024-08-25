# ComMat
Community-based Deep Learning and Antibody H3 Loop Modeling

## Installation

Create a conda environment with this command. It will automatically create an
environment named `h3_loop`. You could also optionally provide the name of
the environment.

```bash
conda env create -f environment.yml [-n <name>]
```
If conda takes too long, it's recommended to use [mamba](https://github.com/conda-forge/miniforge) instead.

```bash
mamba env create -f environment.yml [-n <name>]
```

Then run the following command.

```bash
conda activate commat
python3 setup.py install
```

Also, we need the ANARCI tool to renumber antibody sequences. After cloning the repository from the [ANARCI](https://github.com/oxpig/ANARCI), run the following command:
```bash
cd ANARCI
python setup.py install
```

The run_inference.sh script contains an example execution command. 
When you provide a FASTA file and a seed size, the script generates different structures up to the specified seed size and outputs them in the output_folder/relaxed directory with ranking labels. For instance, if the file is named relaxed/7sn1_H_L_#_1.pdb, it indicates that this structure has a ranking of 1.

Due to technical issues, the AF2rank tool could not be included, but you can download and use it from [ColabDesign](https://github.com/sokrypton/ColabDesign) if needed.

This related
[SO question](https://stackoverflow.com/questions/35064426/when-would-the-e-editable-option-be-useful-with-pip-install)
describes the differences between the two commands.

## License
All code, except for the code in the "src/galaxylocalopt" directory, is licensed under the MIT license. Code in the "src/galaxylocalopt" directory is licensed under the CC BY-NC-ND 4.0 license.
