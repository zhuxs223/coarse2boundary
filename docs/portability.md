# Release adaptation notes

The scientific source snapshots were taken from the implementations used for the manuscript. `source_provenance.json` records original relative paths, original file hashes, release hashes, and each changed file's adaptation notes.

Release-specific changes are limited to:

1. A shared, foreground launcher and a relative-path configuration file.
2. A default U-Net recovery student and 200-epoch generator budget; comparison defaults use the reported common 100-epoch budget. Earlier workstation launchers also supported longer budgets and defaulted to an H/8 student.
3. Initializer orchestration that runs the full-cohort H/4 model directly. The development-only old 4,127-tile teacher comparison is omitted. Data loading, model, loss, optimizer, and training-loop implementations are retained.
4. Explicit registration of a locally regenerated initializer. Hash checks use that checkpoint's recorded provenance instead of requiring the original author's exact checkpoint bytes. Architecture, H/2 skip, ordered full cohort, epoch 10, parameter count, and the 4,360 attempted-update budget are still checked. Actual optimizer updates are retained, including any AMP skips.
5. Configurable data-loader workers. The fixed-reference experiment's source-provenance map is regenerated for this portable source tree, preserving its consistency checks.

Model weights and train/validation/test metrics depend on the training environment. Source and manifest byte fingerprints are preserved by `.gitattributes` for checkout consistency. GitHub browser uploads preserve the supplied files directly.

Generated teacher provenance, run outputs, datasets, and model weights are ignored by Git. The archive is source-only.
