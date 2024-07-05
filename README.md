# README
This is the codebase for FabHacks: Transform Everyday Objects into Home Hacks Leveraging a Solver-aided DSL. Please expect there to be bugs/issues and let me know as you encounter them.

Related repository: [Primitive Tagging](https://github.com/merlinyx/primtag)

## Getting Started

1. Install dependencies (described below)
2. Activate the fabhacks conda environment: `conda activate fabhacks`
3. Install the fabhacks package: `pip install -e .`
4. Setup the OpenSCAD path by setting the environmental variable `OPENSCAD_EXEC`, e.g. `export OPENSCAD_EXEC=path_to_openscad`. Preferably add to terminal profiles like `.bash_profile`
5. Test with the viewer program: `python ui/viewer.py`

## Install Dependencies

This code is in Python and depends on [openscad](https://openscad.org/downloads.html).

It'd be best to set up a separate Python environment using [miniforge](https://github.com/conda-forge/miniforge).

Use the command below to create a conda environment (tested with py38).

`conda create --name=fabhacks python=3.8`

`conda activate fabhacks`

Check that pip is installed using `conda list`. Then install the packages with

`pip install -e .`

`python -m pip install numpy`

`python -m pip install importlib-resources`

`python -m pip install scipy`

`python -m pip install solidpython`

`python -m pip install libigl`

`python -m pip install dill`

`python -m pip install networkx`

`python -m pip install ete3`

`python -m pip install six`

`python -m pip install xxhash`

`conda install -c conda-forge scikit-sparse`

`conda install pyopengl` (needed for running explorer)

`python -m pip install pillow` (needed for running explorer)

For polyscope, please clone [this fork](https://github.com/merlinyx/polyscope-py.git) with `ImageButton` bindings for running the `explorer.py` UI, and install that version with `python -m pip install -e .`. If not intending to use explorer, feel free to do `python -m pip install polyscope`.
