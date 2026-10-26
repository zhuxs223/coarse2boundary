"""Construct 0/1/2 source supervision from coarse masks; no fine-label reads."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from release_utils import ROOT, load_config


def weak_mask(coarse):
    import cv2
    import numpy as np
    foreground = coarse > 0
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(foreground.astype(np.uint8), kernel,
                       borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    contour = foreground & ~eroded
    contour[[0, -1], :] = False
    contour[:, [0, -1]] = False
    outline = cv2.dilate(contour.astype(np.uint8), kernel,
                         borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    result = foreground.astype(np.uint8)
    result[outline] = 2
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path)
    p.add_argument("--limit", type=int, default=0, help="0: all cohort masks")
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    import numpy as np
    from PIL import Image
    root = Path(load_config(a.config)["data_root"])
    keys = json.loads((ROOT / "meta/splits_full5227.json").read_text(encoding="utf-8"))["train_full"]
    if a.limit:
        keys = keys[:a.limit]
    written = checked = 0
    for key in keys:
        with Image.open(root / "mask_labels/train" / (key + ".png")) as im:
            expected = weak_mask(np.asarray(im.convert("L")))
        target = root / "mask_labels_3px/train" / (key + ".png")
        if target.exists():
            with Image.open(target) as im:
                actual = np.asarray(im.convert("L"))
            # The image frame is ignored by every loss/evaluation domain.
            if not np.array_equal(actual[1:-1, 1:-1], expected[1:-1, 1:-1]):
                raise ValueError(f"Existing supervision differs inside the valid frame: {target}")
            checked += 1
        elif a.check_only:
            raise FileNotFoundError(target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(expected).save(target)
            written += 1
    print(json.dumps(dict(tiles=len(keys), checked=checked, written=written)))


if __name__ == "__main__":
    main()
