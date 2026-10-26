"""Atomic epoch commits and a Windows-compatible process lock."""
import csv
import hashlib
import json
import os
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def torch_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(value, tmp)
    # Flush file buffers before publishing the commit pointer/checkpoint.
    with tmp.open('r+b') as f:
        os.fsync(f.fileno())
    tmp.replace(path)


def csv_write(path, rows, fields=None):
    if not rows:
        return
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def seed(n):
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(n)


@contextmanager
def run_lock(root):
    path = Path(root) / 'run.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open('a+b')
    if path.stat().st_size == 0:
        f.write(b'0')
        f.flush()
    f.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        raise RuntimeError('Another process is using this experiment output')
    try:
        yield
    finally:
        f.seek(0)
        if os.name == 'nt':
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()
