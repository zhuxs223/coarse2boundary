# Recovering internal parcel boundaries from coarse agricultural annotations

Research code for recovering missing internal parcel boundaries from RGB imagery and coarse agricultural region masks, then evaluating the resulting refined labels with independently trained parcel-segmentation models.

The implementation combines a shallow SMALL initializer, iterative teacher–student training, structural screening, fixed-reference distribution mapping, and supervision updates. The repository includes the reported comparison methods and the three-label-source downstream experiment.

## Setup

Use a Python environment with a matching PyTorch/torchvision installation for your GPU. Install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

The reported architecture uses `segmentation-models-pytorch==0.5.0`. The paper reports PyTorch 2.12.0; the release implementation checks are recorded separately in [validation](docs/validation.md). Full training requires a GPU and is started explicitly by the user.

## Data and encoder weights

1. Obtain GTPBD from its authors; see the [dataset paper](https://arxiv.org/abs/2507.14697v3).
2. Edit `config.json` to point to your data. Keep the provided cohort and spatial split files.
3. Obtain the declared torchvision ResNet-50 encoder checkpoint:

```bash
python tools/download_encoder.py
```

4. If the coarse-derived `mask_labels_3px` layer is missing, construct it:

```bash
python tools/prepare_weak_labels.py
```

Images, annotations, pretrained weights, trained checkpoints, and historical experimental outputs are not included. [Data documentation](docs/data.md) gives the directory layout and label definitions.

## Main workflow

Run these commands from the repository root. The default paths are relative to the checkout, and all long-running processes remain in the foreground.

```bash
# 1. SMALL H/4 initializer: fixed 10 epochs on all 5,227 target tiles.
python run_experiment.py initialize --precheck
python run_experiment.py initialize

# 2. Full recovery: 20 rounds × 10 student epochs; U-Net student.
python run_experiment.py recovery --precheck
python run_experiment.py recovery

# 3. Independently train and evaluate fine / recovered / coarse label arms.
python run_experiment.py downstream --precheck
python run_experiment.py downstream
```

Initialization automatically registers the terminal SMALL checkpoint and its actual provenance for later experiments. To use an existing compatible checkpoint, set `small_teacher` in `config.json`, then run `python tools/register_teacher.py`.

The common recovery comparison endpoint is epoch 100, corresponding to `round_010`; downstream label construction uses the fixed final recovery checkpoint at epoch 200, `round_020`. Upstream recovery uses the complete target collection transductively. Downstream model training, validation, and testing use the supplied spatially separated partitions.

## Comparison experiments

```bash
python run_experiment.py s2l
python run_experiment.py nnpu
python run_experiment.py s2l-small
python run_experiment.py probability-lut
python run_experiment.py structure-no-lut
python run_experiment.py probability-no-lut
python run_experiment.py reference-student
```

The default comparison budget is 100 student epochs. The main recovery also includes the separately reported 10-epoch SMALL initialization. `nnpu` is the **nnPU-Oracle** condition and explicitly uses fine annotations to calculate per-image class proportions. `reference-student` requires the baseline's completed first round. Each command accepts `--precheck`; `--show-command` prints a launch plan without starting training.

For the model-size experiments:

```bash
# Initial-prior experiment: requires the completed H/4 initializer.
python run_experiment.py h8-initialization
# Separate student-architecture experiment: retains the H/4 initial teacher.
python run_experiment.py recovery --student h8
```

The [experiment guide](docs/experiments.md) maps implementations, budgets, and outputs to the manuscript. The [portability notes](docs/portability.md) describe release-only changes to paths, launch defaults, and provenance registration.

## Outputs and checks

Generated files go to `out/`. The main generator is `out/unet_abc_fullteacher/round_020/final_r020.pth`; downstream comparison scores are written to `out/downstream_parcel_3in1_ep100/comparison_test.csv`.

```bash
python tools/validate_release.py
python tools/validate_release.py --models
```

The first command validates source syntax, manifests, launch plans, and synthetic pixel protocols. `--models` additionally checks the declared model parameter counts, output shapes, and a small weak-supervised gradient update. These checks do not run the reported full training experiments.

## Sources and license

Original implementation lineage is recorded in [source provenance](docs/source_provenance.json). References for GTPBD, U-Net, ResNet, Scribble2Label, nnPU, and third-party software are listed in [notices](THIRD_PARTY_NOTICES.md).

This repository is released under the [MIT License](LICENSE). Dataset and dependency terms are provided by their respective authors. The associated manuscript title is given above; publication and code-archive identifiers can be added when available.
