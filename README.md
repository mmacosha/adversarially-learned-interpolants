# Multi-Marginal Flow Matching with Adversarially Learnt Interpolants

<p align="center">
    📃 <a href="https://arxiv.org/abs/" target="_blank">Paper</a>  <br>
</p>

<!-- <p align="center">
  <img src="assets/d2e_cifar10_main.png" alt="Project Screenshot/Logo" width="700"/>
</p> -->

---
> **Multi-Marginal Flow Matching with Adversarially Learnt Interpolants**<br>
> Oskar Kviman, Kirill Tamogashev, Nicola Branchini, Víctor Elvira, Jens Lagergren, Nikolay Malkin<br><br>
>**Abstract:** Learning the dynamics of a process given sampled observations at several time points is an important but difficult task in many scientific applications. When no ground-truth trajectories are available, but one has only snapshots of data taken at discrete time steps, the problem of modelling the dynamics, and thus inferring the underlying trajectories, can be solved by multi-marginal generalisations of flow matching algorithms. This paper proposes a novel flow matching method that overcomes the limitations of existing multi-marginal trajectory inference algorithms. Our proposed method, ALI-CFM, uses a GAN-inspired adversarial loss to fit neurally parametrised interpolant curves between source and target points such that the marginal distributions at intermediate time points are close to the observed distributions. The resulting interpolants are smooth trajectories that, as we show, are unique under mild assumptions.  These interpolants are subsequently marginalised by a flow matching algorithm, yielding a trained vector field for the underlying dynamics. We showcase the versatility and scalability of our method by outperforming the existing baselines on spatial transcriptomics and cell tracking datasets, while performing on par with them on single-cell trajectory prediction.

## Project structure
```
ali/
├── 🔬 configs/           # Experiment configs
│   └── ...
├── 💾 data/              
│   ├── cell_tracking/    # Cell tracking data
│   ├── single_cell/      # Single cell data
│   └── st/               # Spatial transcriptomics data
├── 📖 docs/              # Project documentation
│   └── TRAINING.md
├── 🚀 scripts/           # Standalone scripts for running experiments or tasks
│   └── ...
├── ali_cfm/              # Your Python package
│   ├── scripts           # Scripts for running all the experiments
│   ├── training/         # Training  logic
│   └── ...
├── 🙈 .gitignore                
├── 🛠️ pyproject.toml     # Project metadata and build configuration
└── 👋 README.md          # Your project's welcome page!
```

## Installation

Here's how you can install this repository and reproduce the experiments

* Python 3.10+
* We use [uv](https://docs.astral.sh/uv/) package manager

```bash
# This example assumes that uv is installed. 
# If not, follow the link above to install it or use a package mangaer of your choice.

# 1. Clone the repository into a folder named 'ali' and navigate into it
git https://github.com/mmacosha/adversarially-learnt-interpolants.git ali
cd ali

# 2. Create a virtual environment using Python 3.11 with uv
uv venv -n alienv --python 3.10

# 3. Activate the newly created virtual environment
source ./alienv/bin/activate

# 4. Install the project and all its dependencies in editable mode
pip install -e .
```

## Single cell experiments

### Data
Single cell RNA data can be downloaded with the following [url](https://data.mendeley.com/datasets/hhny5ff7yj/1).
The webpage contains three files: `ebdata_v3.h5ad`, `op_cite_inputs_0.h5ad`, `op_train_multi_targets_0.h5ad`. In order to reproce single cell experiments you should download the files and save them to `./data/single_cell`.

### Experiments
```bash
# Cite 100D
train_ali_single_cell --config configs/single_cell/100D/cite-100D.yaml

# Cite 50D
train_ali_single_cell --config configs/single_cell/50D/cite-50D.yaml

# Multi 100D
train_ali_single_cell --config configs/single_cell/100D/multi-100D.yaml

# Cite 50D
train_ali_single_cell --config configs/single_cell/50D/multi-50D.yaml
```

Baselines for the single cell experiments are run using MFM [repository](https://github.com/kksniak/metric-flow-matching/).

## Spatial transcriptomics

### Data
The data was originally published in [Mo et al. (2024)](https://www.nature.com/articles/s41586-024-08087-4) under a CC BY-NC-ND 4.0 license, why we only share the raw versions of the relevant dataset files here. In Mo et al., the datasets are referred to HT206B1-U2, HT206B1-U3, HT206B1-U4 and HT206B1-U5. See the ST data folder in this repo to find the necessary raw data files. In our paper, when we refer to slide one ($t=0$) we are refering to slide U2, while $t=1/3$ corresponds to U3, and so on. 

To preprocess the data, go to data/st_data, and run first ```common_reference_alignment.py```, followed by ```visualize_aligned_tumor_clusters.py```. For each slide U2, ..., U5 there now exists a ```f"{slide}_tumor_coordinates.csv"``` containing the tumour coordinates in the aligned coordinate system which was used as training data, and a ```{slide}_aligned_to_{reference}.tif``` image file which can be used as sanity checks by overlaying the inferred tumour positions on the corresponding images.

### Experiments
```bash
# Train ali-cfm
train_ali_st --config configs/st/ali_st.yaml

# Train OT-CFM or OT-MMFM (specified in the config)
train_cfm_baseline_st --config configs/st/cfm_baseline_st.yaml
```

More details on running baselines for spatial transcriptomics experiments can be found in [here](/docs/TRAINING.md).

## Cell tracking 
### Data
The data can be found on this website, or explicitly via [this zip downloading address](https://data.
celltrackingchallenge.net/training-datasets/PhC-C2DH-U373.zip). Alernatively, go to the [Cell Tracking Challenge dataset page](https://celltrackingchallenge.net/2d-datasets/) and download the training dataset in the "Glioblastoma-astrocytoma U373 cells on a polyacrylamide substrate" row. We use the frames available in folder ```01``` and the corresponding segmentation masks from folder ```01 ST/SEG```. We used the segmentation masks for cell with label four.

To create the dataset used in our paper, move the zip content to data/cell_tracking/PhC-C2DH-U373, and run ```data/cell_tracking/create_dataset.py```. The segmentation masks overlayed on the cell can also be visualised using ```data/cell_tracking/visualize_single_frame.py```.


### Experiments
```bash
# Train ali-cfm
train_ali_cell_tracking --config configs/cell_tracking/ali_cell_tracking.yaml

# Train OT-CFM (OT-MMFM not implemented!)
train_cfm_baseline_cell_tracking --config configs/cell_tracking/cfm_baseline_cell_tracking.yaml
```

More details on running baselines for cell tracking experiments can be found in [here](/docs/TRAINING.md).

## Citation
Please, cite this work as follows:
```
@misc{kviman2025adversarial,
    author    = {Kviman, Oskar and Tamogashev, Kirill and Branchini, Nicola and Elvira, Víctor and Lagergren, Jens and Malkin, Nikolay},
    title     = {Multi-Marginal Flow Matching with Adversarially Learnt Interpolants},
    year      = {2025},
    notes     = {Submitted to ICLR 2026.}
}
```
