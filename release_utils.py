"""Shared path handling for public entry points; no ML imports at startup."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_config(path=None):
    source = Path(path).resolve() if path else ROOT / "config.json"
    values = json.loads(source.read_text(encoding="utf-8"))
    for key in ("data_root", "resnet_weights", "small_teacher", "upstream_checkpoint"):
        p = Path(values[key]).expanduser()
        values[key] = str((source.parent / p).resolve() if not p.is_absolute() else p.resolve())
    values["workers"] = int(values.get("workers", 6))
    if values["workers"] < 0:
        raise ValueError("workers must be nonnegative")
    return values


def environment(config):
    env = os.environ.copy()
    env.update(EXP810_ROOT=str(ROOT), EXP810_DATA=config["data_root"],
               EXP810_SPLIT=str(ROOT / "meta/splits_full5227.json"),
               EXP810_RESNET=config["resnet_weights"],
               EXP810_TEACHER=config["small_teacher"],
               EXP810_SMALL_TEACHER=config["small_teacher"],
               EXP810_WORKERS=str(config["workers"]),
               PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    return env


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 ** 2), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
