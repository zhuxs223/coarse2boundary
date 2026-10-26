"""Register a fixed-cohort SMALL checkpoint; preserve provenance and strict checks."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from release_utils import ROOT, load_config, sha, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--metadata-dir", type=Path, default=ROOT / "meta")
    args = parser.parse_args()
    import torch
    cfg = load_config(args.config)
    teacher = Path(cfg["small_teacher"])
    ck = torch.load(teacher, map_location="cpu", weights_only=True)
    manifest = ck["manifest"]
    cohort_path = ROOT / "meta/splits_full5227.json"
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    attempts = ck.get("train_attempts", ck["optimizer_updates"])
    expected = dict(epoch=10, architecture="SMALL", params=557505, tiles=5227, attempted_updates=4360)
    if (ck["epoch"] != expected["epoch"] or manifest["n_train"] != expected["tiles"]
            or manifest["train_keys"] != cohort["train_full"] or not manifest["h2_skip"]
            or manifest["params"] != expected["params"] or attempts != expected["attempted_updates"]
            or ck["model_config"] != dict(architecture="SMALL", h2_skip=True)):
        raise ValueError("Expected full5227, H/4 SMALL with H/2 skip, fixed epoch10 and 4360 attempted updates")
    if sha(cfg["resnet_weights"]) != "8c28bce1adf71c9dbb928e91724832100e67dc6ff68af1a3a79f53cc39833bbf":
        raise ValueError("Use the declared torchvision ResNet-50 IMAGENET1K_V1 checkpoint")
    metadata = args.metadata_dir.resolve()
    manifest_path = metadata / "small_full5227_h4_manifest.json"
    atomic_json(manifest_path, manifest)
    provenance = dict(architecture="SMALL H/4 with H/2 skip", epoch=10, train_tiles=5227,
                      teacher_file=str(teacher), teacher_sha256=sha(teacher),
                      manifest_file="meta/small_full5227_h4_manifest.json",
                      manifest_sha256=sha(manifest_path), resnet_sha256=sha(cfg["resnet_weights"]),
                      split_sha256=sha(cohort_path), teacher_training_updates=ck["optimizer_updates"],
                      teacher_training_attempts=attempts, teacher_training_batch=12,
                      role="Initial teacher only; students start with an ImageNet encoder and new decoder",
                      fine_gt="Scoring only; fixed epoch10, no fine-GT checkpoint selection")
    atomic_json(metadata / "full_teacher_provenance.json", provenance)
    print(json.dumps({"registered_teacher": str(teacher), "sha256": provenance["teacher_sha256"],
                      "metadata": str(metadata), "optimizer_updates": ck["optimizer_updates"]}, indent=2))


if __name__ == "__main__":
    main()
