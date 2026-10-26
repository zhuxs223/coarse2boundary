# Experiment map

All commands are launched from the repository root. Use `--precheck` before starting a full run. `--show-command` prints the selected script and budget without importing the ML stack or starting a child process.

| Experiment | Command | Default budget | Output under `out/` |
|---|---|---|---|
| SMALL H/4 initialization | `python run_experiment.py initialize` | 10 epochs, batch 12 | `initialization/full_h2_on_ep10/` |
| Complete recovery | `python run_experiment.py recovery` | 20 rounds / 200 student epochs | `unet_abc_fullteacher/` |
| S2L-Adapted | `python run_experiment.py s2l` | 100 epochs, includes warm-up 20 | `unet_s2l/` |
| nnPU-Oracle | `python run_experiment.py nnpu` | 100 epochs | `unet_pu/` |
| S2L with SMALL prior | `python run_experiment.py s2l-small` | 100 epochs | `unet_s2l_small_prior/` |
| No structural group, LUT on | `python run_experiment.py probability-lut` | 10 rounds / 100 epochs | `unet_fullteacher_probability/` |
| Structural group on, LUT off | `python run_experiment.py structure-no-lut` | 10 rounds / 100 epochs | `unet_abc_fullteacher_nolut/` |
| No structural group, LUT off | `python run_experiment.py probability-no-lut` | 10 rounds / 100 epochs | `unet_fullteacher_probability_nolut/` |
| Fixed reference from first student round | `python run_experiment.py reference-student` | 10 rounds / 100 epochs | `unet_fullteacher_r1_reference/` |
| Three downstream label sources | `python run_experiment.py downstream` | 100 epochs per arm | `downstream_parcel_3in1_ep100/` |
| H/8 initial-prior experiment | `python run_experiment.py h8-initialization` | 10 epochs | `h8_initialization/` |
| H/8 student, same H/4 teacher | `python run_experiment.py recovery --student h8` | 20 rounds / 200 epochs | `h8_abc_fullteacher/` |

## Reading the reported endpoints

Main recovery comparisons use epoch 100 (`round_010`), and the downstream label generator uses the fixed epoch-200 checkpoint (`round_020`). The main method's SMALL initialization is additional to student epochs. Setting `--rounds 10` starts a 100-epoch recovery run; it can later be extended to 20 rounds with the same source and input files. Use the same experiment directory and initialization when continuing a run.

S2L and nnPU can be extended with `--epochs 200`; this selects a new budget, not the main table's common endpoint. Independently run S2L initializations should be interpreted by within-run trends.

## Component definitions

The structural group includes A/B/C and contact screening, coupled score adjustments, historical anchors and displacement exclusion, and repeated-support confidence enhancement. The probability-based comparisons remove that group together. Independent broad-response supervision retention and the negative policy remain active.

LUT matching retains the initial SMALL score-distribution reference. `structure-no-lut` bypasses matching. `reference-student` imports the complete first-round model, optimizer, and histories from `unet_abc_fullteacher`, then changes the fixed reference from round 2 onward. It checks the baseline state and source provenance before reuse.

## Downstream experiment

Fine, recovered, and coarse supervision arms initialize independently from the same ImageNet encoder and random decoder. They use identical 100-epoch schedules, D4 image augmentation, BCE plus soft Dice, and validation selection by boundary-band parcel IoU. Final testing starts after all arms finish training. The test threshold is 0.5.

Refined parcel labels are `coarse foreground & ~(raw final boundary probability >= 0.5)`. The final generator output is not remapped or structurally postprocessed at this stage. The recovered-label model is a separately trained segmentation model, not the generator checkpoint.

## Checkpoint and state handling

Recovery and comparison runners check source, configuration, and input signatures when resuming. Completed rounds are reused; interrupted rounds restart from their beginning. The downstream runner resumes from a committed epoch. SMALL initialization is a short fixed 10-epoch training script and does not resume a partially created initializer directory. Existing output is retained on failure; choose a separate run location before restarting it.

`--smoke` is available for recovery and the comparison runners. It uses small fixtures and a short schedule; its numbers are implementation checks rather than manuscript results.
