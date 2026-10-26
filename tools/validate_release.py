"""Fast release checks; no full training, cohort inference, or downloads."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from release_utils import ROOT, environment, load_config, sha


def model_check(name, cfg):
    snippets = {
        "small": '''
import run_pair as R
R.seed_all()
net=R.Small(True).to(device)
assert sum(p.numel() for p in net.parameters())==557505
net.train()
x=torch.randn(2,3,64,64,device=device)
labels=torch.ones(2,64,64,dtype=torch.uint8)
labels[:,:8,:]=0
labels[:,20:23,8:56]=2
batch=dict(mk3=labels,nod=torch.zeros_like(labels,dtype=torch.bool))
optimizer=torch.optim.AdamW([
    dict(params=[p for p in net.parameters() if p.ndim!=1],weight_decay=.01),
    dict(params=[p for p in net.parameters() if p.ndim==1],weight_decay=0)],lr=1e-4)
y=net(x)
assert tuple(y.shape)==(2,1,64,64)
loss=R.loss_fn(y,batch,device)
assert torch.isfinite(loss)
optimizer.zero_grad(set_to_none=True)
loss.backward()
assert net.conv1.weight.grad is not None and net.conv1.weight.grad.abs().sum()>0
optimizer.step()
result=dict(model='SMALL.H4',parameters=557505,shape=list(y.shape),weak_loss=float(loss.detach()),encoder_gradient=True,optimizer_step=True)
''',
        "unet": '''
import model_student_unet as M
net=M.build_model(weights,verbose=False).to(device).eval()
assert sum(p.numel() for p in net.parameters())==32521105
with torch.inference_mode():y=net(torch.randn(1,3,64,64,device=device))
assert tuple(y.shape)==(1,1,64,64)
assert all(block.interpolation_mode=='bilinear' for block in net.net.decoder.blocks)
result=dict(model='ResNet50.UNet',parameters=32521105,shape=list(y.shape),decoder='bilinear')
''',
        "h8": '''
import model_h8 as M
net=M.SmallH8(weights).to(device).eval()
assert sum(p.numel() for p in net.parameters())==2145857
with torch.inference_mode():y=net(torch.randn(1,3,64,64,device=device))
assert tuple(y.shape)==(1,1,64,64)
result=dict(model='SMALL.H8',parameters=2145857,shape=list(y.shape))
'''}
    folder = "recovery" if name == "unet" else "initialize_small"
    prefix = "import sys,json,torch\nfrom pathlib import Path\n"
    prefix += f"sys.path.insert(0,{str(ROOT / 'experiments' / folder)!r})\n"
    prefix += f"weights={cfg['resnet_weights']!r}\n"
    prefix += "torch.set_num_threads(2)\ndevice=torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
    suffix = "\nresult['device']=str(device)\nprint('VALIDATION_JSON='+json.dumps(result))\n"
    env = environment(cfg)
    env.update(EXP810_STUDENT="unet", PYTHONDONTWRITEBYTECODE="1")
    process = subprocess.run([sys.executable, "-B", "-c", prefix + snippets[name] + suffix],
                             cwd=ROOT, env=env, text=True, encoding="utf-8", capture_output=True)
    if process.returncode:
        raise RuntimeError(process.stdout + process.stderr)
    line = next(x for x in process.stdout.splitlines() if x.startswith("VALIDATION_JSON="))
    return json.loads(line.split("=", 1)[1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", action="store_true")
    p.add_argument("--config", type=Path)
    p.add_argument("--report", type=Path)
    a = p.parse_args()
    sources = [x for x in ROOT.rglob("*.py") if not any(t in x.relative_to(ROOT).parts for t in ("out", "data", "weights", "__pycache__"))]
    for file in sources:
        ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
    cohort = json.loads((ROOT / "meta/splits_full5227.json").read_text(encoding="utf-8"))
    groups = cohort["source_groups"]
    keys = [k for group in ("train_sd", "val_sd", "test_sd", "dropped_buffer") for k in groups[group]]
    assert len(keys) == len(set(keys)) == 5227
    assert cohort["train_full"] == cohort["target_full"] == keys
    assert hashlib.sha256(("\n".join(keys) + "\n").encode()).hexdigest() == cohort["keys_sha256"]
    assert [len(groups[g]) for g in ("train_sd", "val_sd", "test_sd", "dropped_buffer")] == [4127, 414, 375, 311]
    spatial = json.loads((ROOT / "meta/splits_v4_sd.json").read_text(encoding="utf-8"))
    assert all(spatial[g] == groups[g] for g in ("train_sd", "val_sd", "test_sd"))
    provenance = json.loads((ROOT / "docs/source_provenance.json").read_text(encoding="utf-8"))
    assert all(sha(ROOT / x["release"]) == x["release_sha256"] for x in provenance)
    from tools.prepare_weak_labels import weak_mask
    import numpy as np
    coarse = np.zeros((32, 32), np.uint8)
    coarse[5:27, 4:28] = 255
    weak = weak_mask(coarse)
    assert set(np.unique(weak)) == {0, 1, 2}
    assert weak[16, 16] == 1 and weak[0, 0] == 0 and weak[5, 10] == 2
    import run_experiment as entry
    launchers = {}
    for experiment in entry.EXPERIMENTS:
        proc = subprocess.run([sys.executable, "-B", str(ROOT / "run_experiment.py"), experiment, "--show-command"],
                              cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        if proc.returncode:
            raise RuntimeError(proc.stdout + proc.stderr)
        launch = json.loads(proc.stdout)
        assert Path(launch["command"][2]).is_file()
        launchers[experiment] = True
    result = dict(python_sources=len(sources), syntax=True, cohort=True, downstream_membership=True,
                   source_hashes=True, synthetic_coarse_supervision=True, launch_plans=launchers,
                   full_training_started=False, full_inference_started=False)
    if a.models:
        cfg = load_config(a.config)
        if not Path(cfg["resnet_weights"]).is_file():
            raise FileNotFoundError("Set resnet_weights in config.json or run tools/download_encoder.py first")
        result["models"] = [model_check(name, cfg) for name in ("small", "unet", "h8")]
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
