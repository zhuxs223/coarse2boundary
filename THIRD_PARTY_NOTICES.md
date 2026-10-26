# Research and software sources

This repository contains the authors' task-specific implementation and adaptations. The original training/reference snapshots are identified in `docs/source_provenance.json`.

Research sources:

- [GTPBD: A Fine-Grained Global Terraced Parcel and Boundary Dataset](https://arxiv.org/abs/2507.14697v3).
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
- [Scribble2Label](https://arxiv.org/abs/2006.12890v2), with the [original implementation](https://github.com/hvcl/scribble2label). The supplied S2L comparison is adapted to boundary recovery.
- [Positive-Unlabeled Learning with Non-Negative Risk Estimator](https://proceedings.neurips.cc/paper_files/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html).
- [A Positive/Unlabeled Approach for the Segmentation of Medical Sequences Using Point-Wise Supervision](https://arxiv.org/abs/2107.08394). The supplied nnPU-Oracle condition additionally uses true per-image class proportions.

Dependencies are installed separately and retain their own licenses and notices:

- [PyTorch](https://github.com/pytorch/pytorch), [torchvision](https://github.com/pytorch/vision), and the torchvision ResNet-50 checkpoint.
- [Segmentation Models PyTorch](https://github.com/qubvel-org/segmentation_models.pytorch), version 0.5.0.
- [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [scikit-image](https://scikit-image.org/), [OpenCV](https://opencv.org/), [Pillow](https://python-pillow.org/), and [Matplotlib](https://matplotlib.org/).

The repository MIT license applies to the authors' code. GTPBD rasters, third-party source repositories, and pretrained/trained model weights are not bundled or relicensed here.
