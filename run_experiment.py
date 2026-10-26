"""Foreground entry points for the manuscript experiments."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from release_utils import ROOT, environment, load_config, atomic_json

EXPERIMENTS = ("initialize", "h8-initialization", "recovery", "s2l", "nnpu", "s2l-small",
               "probability-lut", "structure-no-lut", "probability-no-lut", "reference-student", "downstream")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument("--config", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--precheck", action="store_true", help="Input checks without full training")
    mode.add_argument("--smoke", action="store_true", help="Small implementation check; not a paper result")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--student", choices=("unet", "h8"), default="unet")
    parser.add_argument("--show-command", action="store_true", help="Print the launch plan without starting a process")
    args = parser.parse_args()
    if args.rounds is not None and args.rounds < 1:
        parser.error("rounds must be positive")
    if args.epochs is not None and args.epochs < 1:
        parser.error("epochs must be positive")
    cfg = load_config(args.config)
    env = environment(cfg)
    mode_flag = ["--precheck"] if args.precheck else ["--smoke"] if args.smoke else []
    extra = []
    followup = None
    if args.experiment == "initialize":
        if args.smoke:
            parser.error("Use tools/validate_release.py for the small initializer check")
        script = ROOT / "experiments/initialize_small/run_pair.py"
        extra = ["--workers", str(cfg["workers"]), "--out", str(ROOT / "out/initialization")]
        if not args.precheck:
            followup = [sys.executable, str(ROOT / "tools/register_teacher.py")]
            if args.config:
                followup += ["--config", str(args.config.resolve())]
    elif args.experiment == "h8-initialization":
        if args.smoke:
            parser.error("H/8 initialization supports --precheck or the full fixed 10 epochs")
        script = ROOT / "experiments/initialize_small/run_h8.py"
        extra = ["--workers", str(cfg["workers"]), "--out", str(ROOT / "out/h8_initialization"),
                 "--reference", str(ROOT / "out/initialization/full_h2_on_ep10")]
    elif args.experiment == "recovery":
        script = ROOT / "experiments/recovery/run_windows.py"
        env.update(EXP810_ROUNDS=str(args.rounds or 20), EXP810_STUDENT=args.student)
        extra = ["--student", args.student]
    elif args.experiment in ("s2l", "nnpu", "s2l-small"):
        folder = "s2l_small" if args.experiment == "s2l-small" else args.experiment
        script = ROOT / "experiments" / folder / "run_windows.py"
        env["EXP810_EPOCHS"] = str(args.epochs or 100)
        if folder == "s2l_small":
            extra = ["--epochs", str(args.epochs or 100)]
    elif args.experiment == "downstream":
        if args.smoke:
            parser.error("Use downstream/smoke_test.py as documented, or --precheck")
        script = ROOT / "downstream/run_all.py"
        dcfg = json.loads((ROOT / "downstream/config.json").read_text(encoding="utf-8"))
        dcfg.update(data_root=cfg["data_root"], split_file=str(ROOT / "meta/splits_v4_sd.json"),
                    resnet_weights=cfg["resnet_weights"], pseudo_checkpoint=cfg["upstream_checkpoint"],
                    output_root=str(ROOT / "out/downstream_parcel_3in1_ep100"), workers=cfg["workers"])
        runtime = ROOT / "out/release_configs/downstream.json"
        if not args.show_command:
            atomic_json(runtime, dcfg)
        extra = ["--config", str(runtime)]
    else:
        folder = {"probability-lut": "probability_lut", "structure-no-lut": "structure_nolut",
                  "probability-no-lut": "probability_nolut", "reference-student": "reference_student"}[args.experiment]
        script = ROOT / "experiments" / folder / "run_windows.py"
        extra = ["--rounds", str(args.rounds or 10)]
        if folder == "structure_nolut":
            extra += ["--lut", "off"]
        if folder == "reference_student":
            extra += ["--baseline", str(ROOT / "out/unet_abc_fullteacher")]
    command = [sys.executable, "-u", str(script), *mode_flag, *extra]
    print(json.dumps({"experiment": args.experiment, "command": command,
                      "root": str(ROOT), "data_root": cfg["data_root"],
                      "rounds": env.get("EXP810_ROUNDS", args.rounds),
                      "epochs": env.get("EXP810_EPOCHS", args.epochs),
                      "initializer_registration": followup}, indent=2), flush=True)
    if args.show_command:
        return 0
    result = subprocess.run(command, cwd=ROOT, env=env)
    if result.returncode:
        return result.returncode
    if followup:
        return subprocess.run(followup, cwd=ROOT, env=env).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
