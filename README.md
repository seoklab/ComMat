# ComMat
Community-based Deep Learning and Antibody H3 Loop Modeling

## Installation

Create a conda environment with this command. It will automatically create an
environment named `h3_loop`. You could also optionally provide the name of
the environment.

```bash
conda env create -f environment.yml [-n <name>]
```

Then run the following command.

```bash
pip3 install .
```

Or for "editable" install,

```bash
pip3 install -e .
```

```bash
python3 setup.py install
```


This related
[SO question](https://stackoverflow.com/questions/35064426/when-would-the-e-editable-option-be-useful-with-pip-install)
describes the differences between the two commands.


