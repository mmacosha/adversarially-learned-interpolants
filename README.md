# Multi-Marginal Flow Matching with Adversarially Learnt Interpolants

<p align="center">
    📃 <a href="https://arxiv.org/abs/" target="_blank">Paper</a>  <br>
</p>

<!-- <p align="center">
  <img src="assets/d2e_cifar10_main.png" alt="Project Screenshot/Logo" width="700"/>
</p> -->

---
> **Multi-Marginal Flow Matching with Adversarially Learnt Interpolants**<br>
> Kirill Tamogashev & Nikolay Malkin<br><br>
>**Abstract:** Learning the dynamics of a process given sampled observations at several time points is an important but difficult task in many scientific applications. When no ground-truth trajectories are available, but one has only snapshots of data taken at discrete time steps, the problem of modelling the dynamics, and thus inferring the underlying trajectories, can be solved by multi-marginal generalisations of flow matching algorithms. This paper proposes a novel flow matching method that overcomes the limitations of existing multi-marginal trajectory inference algorithms. Our proposed method, ALI-CFM, uses a GAN-inspired adversarial loss to fit neurally parametrised interpolant curves between source and target points such that the marginal distributions at intermediate time points are close to the observed distributions. The resulting interpolants are smooth trajectories that, as we show, are unique under mild assumptions.  These interpolants are subsequently marginalised by a flow matching algorithm, yielding a trained vector field for the underlying dynamics. We showcase the versatility and scalability of our method by outperforming the existing baselines on spatial transcriptomics and cell tracking datasets, while performing on par with them on single-cell trajectory prediction.

## Project structure
```
energy-sb/
├── 📜 configs/           # Hydra experiment configs
├── 🖼️ assets/            # Static assets like images, plots, tables
├── 🐍 sb/                # Main source code for the Schrödinger Bridge package
│   ├── data/             # Data modules
│   ├── buffers/          # Replay buffers for off-policy training
│   ├── losses/           # Loss functions
│   ├── nn/               # Neural network architectures
│   ├── samplers/         # Core training loops and logic
│   └── ...
├── 🚀 py_scripts/        # Executable Python scripts for training and generation
│   ├── train.py
│   └── ...
├── 셸 sh_scripts/         # Shell scripts for running experiments
│   └── ...
├── 🧪 tests/             # Unit and integration tests
├── 📄 pyproject.toml     # Project configuration and dependencies
└── 📖 README.md          # Project overview and documentation
```


## Installation

Here's how you can install this repository and reproduce the experiments

* Python 3.11+
* We use [uv](https://docs.astral.sh/uv/) package manager

```bash
# This example assumes that uv is installed. 
# If not, follow the link above to install it or use a package mangaer of your choice.

# 1. Clone the repository into a folder named 'sb' and navigate into it
git https://github.com/mmacosha/d2e-stochastic-dynamics.git sb
cd sb

# 2. Create a virtual environment using Python 3.11 with uv
uv venv -n sbenv --python 3.11

# 3. Activate the newly created virtual environment
source ./sbenv/bin/activate

# 4. Install the project and all its dependencies in editable mode
pip install -e .
```

## 2D Experiments
To reproduce 2D experiments you should just to run the scripts located in `sh-scripts/...'. For example:
```bash
./sh_scripts/d2d-benchmarks/ablate-d2d-normal_to_gmm.sh 1
```
All the experiments suppot `cpu`, `mps` and `cuda` training.

## Image experiments
To reproduce image experiments you would need to install the following models
1. [SN-GAN](https://github.com/GongXinyuu/sngan.pytorch)
2. [StyleGAN](https://github.com/NVlabs/stylegan3)
3. [CIFAR-10 Classifier](https://github.com/huyvnphan/PyTorch_CIFAR10)
<!-- 4. [CelebA Classifier]() -->

All models should be installed into the `sb/external`. It is possible to load these dependences using 
`git clone --recurse-submodules`, however, some modifications to the code (mostly paths for imports) might be needed.

The experiments can be run as follows: 
```bash
train --cfg=sb-sg --device=0 --wandb=online --name=EXPERIMENT-NAME \
      --overrides="data.p_1.args.target_classes=[5],sampler.off_policy_fraction=0.8"
```

## Citation
Please, cite this work as follows
```
@misc{kviman@ali,
    author    = {Kviman, Oskar and Tamogashev, Kirill and Branchini, Nicola and Elvira, Víctor and Lagergren, Jens and Malkin, Nikolay},
    title     = {Multi-Marginal Flow Matching with Adversarially Learnt Interpolants},
    year      = {2025},
    notes     = {Submitted to ICLR 2026.}
}
```