"""Download the explicitly declared torchvision ImageNet encoder weights."""
import argparse
from pathlib import Path
import sys
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from release_utils import load_config, sha

URL = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
SHA256 = "8c28bce1adf71c9dbb928e91724832100e67dc6ff68af1a3a79f53cc39833bbf"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path)
    a = p.parse_args()
    target = Path(load_config(a.config)["resnet_weights"])
    if target.exists():
        if sha(target) != SHA256:
            raise ValueError(f"Existing file differs; keep it and choose a new resnet_weights path: {target}")
        print(f"Verified: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".download")
    if temporary.exists():
        raise FileExistsError(f"Previous download exists; inspect it before retrying: {temporary}")
    with urlopen(URL, timeout=60) as source, temporary.open("wb") as output:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            output.write(block)
    if sha(temporary) != SHA256:
        raise ValueError(f"Download hash mismatch; retained for inspection: {temporary}")
    temporary.replace(target)
    print(f"Downloaded and verified: {target}")


if __name__ == "__main__":
    main()
