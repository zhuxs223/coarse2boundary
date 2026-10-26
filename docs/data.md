# Data layout and label protocol

The study uses a fixed collection of **5,227** original RGB tiles from Central China (2,376) and Southwest China (2,851), each 512 × 512 pixels. Obtain the dataset from the GTPBD authors. The release contains tile identifiers and split metadata, without raster data.

```text
data/GTPBD_enhenced_png/
  images/train/Central China/<tile>.png
  images/train/Southwest China/<tile>.png
  mask_labels/train/<region>/<tile>.png
  mask_labels_3px/train/<region>/<tile>.png
  boundary_labels/train/<region>/<tile>.png
  parcel_labels/train/<region>/<tile>.png      # optional
```

`GTPBD_enhenced_png` preserves the experiment's folder spelling. Any alternative location can be selected through `data_root` in `config.json`. The `train/` storage directory is part of the source data layout; it does not define the downstream split.

- `images`: RGB data.
- `mask_labels`: coarse agricultural foreground, nonzero for agricultural area.
- `mask_labels_3px`: coarse-derived supervision, stored as **0 = source background, 1 = unlabeled interior, 2 = outline positive**. The preparation helper uses coarse masks only.
- `boundary_labels`: fine boundary reference, nonzero for boundary.
- `parcel_labels`: optional fine parcel foreground. If absent, downstream evaluation and the fine-label training arm derive it as `coarse foreground & ~fine boundary`.

Pixels with all three RGB channels equal to zero and coarse background are treated as no-data. The outermost one-pixel image frame is excluded. Original 0/1 and 0/255 binary masks are supported; coarse-derived supervision must keep its 0/1/2 encoding.

## Provided manifests

- `meta/splits_full5227.json`: the exact ordered target cohort and four source groups. All 5,227 tiles participate in upstream recovery.
- `meta/splits_v4_sd.json`: downstream train **4,127**, validation **414**, test **375**, and excluded **311** tiles.
- `meta/paper_teacher_manifest.json`: initializer settings, tile order, and fixed diagnostic probes.
- `meta/paper_teacher_provenance.json`: the original manuscript teacher's provenance, for comparison. A new training run generates its own `meta/full_teacher_provenance.json`.

The target collection is a study-specific subset, not a replacement for the official GTPBD split. Do not randomly resplit overlapping tiles for the downstream experiment. The downstream precheck audits spatial overlap from the source coordinates.

Fine boundaries are read for recovery monitoring and diagnostics; they are not upstream training targets. The fine-label downstream arm and nnPU-Oracle condition have their explicitly documented additional fine-label information.
