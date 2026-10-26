# Release validation

The release is validated with static source/manifest checks and small implementation checks. Full 5,227-tile recovery, baseline training, and 100-epoch downstream runs remain user-launched experiments.

`python tools/validate_release.py` checks Python syntax, the ordered 5,227-tile cohort, the downstream split membership, source-file fingerprints, synthetic coarse-supervision encoding, and every public launch plan.

`python tools/validate_release.py --models` additionally checks:

- SMALL H/4: 557,505 parameters, full-resolution output, finite weak-supervised loss, encoder gradients, and one synthetic optimizer step.
- Recovery U-Net: 32,521,105 parameters, full-resolution output, bilinear decoder.
- H/8 initializer: 2,145,857 parameters and full-resolution output.

The package's `release_validation.json` records the checks actually executed by the release preparer, including the local environment and additional input checks. Synthetic outputs are not paper performance scores.
