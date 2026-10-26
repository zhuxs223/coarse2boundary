# -*- coding: utf-8 -*-
"""
Evaluation module -- the single place where the measurement conventions live.

This module is shared on purpose: the reference run and every later
supervision-construction run must be scored by exactly the same code, otherwise
the comparison is meaningless ("口径老是对不齐" was a recurring failure mode).

Domain definitions (design doc, section 4.1; updated 9-18 23:14)
---------------------------------------------------------------
    interior   = image minus the ring touching the image border  (ONLY that ring)
    valid      = interior minus nodata                  standard segmentation figure
    agri       = (mask | bnd) & valid                   MAIN domain (most sensitive)
    inner_apx  = (mask_labels_3px != 2) & mask & valid  diagnostic, IMPRECISE

    `agri` includes the boundary layer itself, so GT coverage inside it is complete
    by construction; only the border ring is removed.

    `interior` is a geometric region and is NOT reported: it still contains nodata,
    which would silently count as background. `valid` is the reported stratum. Do
    not confuse `interior` with Steger's `BORDER_PX` pad -- the latter is a
    ridge-extraction parameter and has nothing to do with any domain here.

    `inner_apx` REPLACES the old `parcel` domain, which was degenerate: because
    `parcel_labels == (mask & ~boundary_labels)` (verified pixel-wise), the GT
    `boundary & parcel` was empty by construction and every row read R=nan / IoU=0.

    WARNING -- `inner_apx` is knowingly imprecise (user ruling 9-18 23:14):
    `mask_labels_3px == 2` does NOT strictly correspond to the outer contour band,
    and it also swallows part of the real ridges -- GT coverage inside it is only
    CC 85.4% / SW 36.4%. Use it as a diagnostic only, never as the headline domain,
    and always read it together with its reported base_rate.

Why `parcel` was dropped (kept here for the record)
---------------------------------------------------
`parcel_labels == (mask & ~boundary_labels)` holds pixel-wise (verified 20/20).
Hence `parcel` and `boundary_labels` are DISJOINT, and with
`gt = (boundary_labels == 1) & domain` the GT inside a parcel domain is empty *by
construction* -- every row read base_rate=0, R=nan, IoU=0, AUC=nan (confirmed on
the real ep1/ep2 run). That is why the diagnostic slot is now `inner_apx`.

Object / metrics
----------------
    gt   = (boundary_labels == 1) & domain
    pred = (prob >= THR) & domain
    TP = |pred & gt|   FP = |pred & ~gt|   FN = |~pred & gt|
    P = TP/(TP+FP)   R = TP/(TP+FN)   IoU = TP/(TP+FP+FN)

    * NO tolerance, at the metric level (widen the anchor itself if needed)
    * threshold 0.5 only
    * P / R / IoU only -- F1 is a composite and is deliberately not reported
    * base_rate = |gt|/|domain| must always be reported alongside R and P
      (the 0.7 m domain has a very high base rate)
    * AUC was REMOVED (user, 9-19 11:07). It answers "did the discriminative power
      change", NOT "is this good enough"; its conventional grading scale does not
      transfer to a 33% base rate; and it cost ~65 s/epoch (~20% of an epoch, because
      it was a Python-level tie loop over ~250k pixels x 3 domains x 414 tiles).
      Diagnosis is done by eye on the timeline and by the IoU/R trajectory.
      Do not add it back without asking.
"""

import numpy as np

DOMAINS = ("valid", "agri", "inner_apx")
GROUPS = ("ALL", "CC", "SW")


def edge_ring(h, w, k):
    """Boolean border ring of width k."""
    e = np.zeros((h, w), bool)
    if k <= 0:
        return e
    e[:k, :] = True
    e[-k:, :] = True
    e[:, :k] = True
    e[:, -k:] = True
    return e


def interior_region(h, w, edge_trim):
    """`interior` -- remove ONLY the ring touching the image border.

    User ruling (9-18 22:50): `interior` means "the image minus the outermost ring",
    nothing more. `valid` is then `interior` minus nodata.

    Retired: `interior = mask & (mask_labels_3px != 2)`. `mask_labels_3px == 2` does
    not strictly correspond to the outer contour band, so that definition was never
    precise. Do not bring it back.
    """
    return ~edge_ring(h, w, edge_trim)


def build_domains(h, w, msk, bnd, nod, mk3, edge_trim):
    m = msk.astype(bool)
    b = bnd.astype(bool)
    n = nod.astype(bool)
    interior = interior_region(h, w, edge_trim)
    valid = interior & (~n)
    return {
        "valid": valid,
        "agri": (m | b) & valid,
        # IMPRECISE interior approximation: m3==2 does not strictly equal the outer
        # contour band (see module docstring). Diagnostic only.
        "inner_apx": (mk3 != 2) & m & valid,
    }


def cell_stats(pred, gt, dom):
    """ONE pass over the domain for one (image, domain) pair.

    Returns (tp, fp, fn, dom_px, gt_px).

    fp/fn come from counts instead of building `p & ~g` / `~p & g` temporaries:
    with p = pred & dom and g = gt & dom we have |p| = tp + fp and |g| = tp + fn,
    so the two full-size negations are avoidable.

    This is the SINGLE implementation of the confusion counts; every caller (the
    aggregate meters and the per-image rows) goes through it, so those two paths
    cannot drift apart.
    """
    p = pred & dom
    g = gt & dom
    n_p = int(np.count_nonzero(p))
    n_g = int(np.count_nonzero(g))
    tp = int(np.count_nonzero(p & g))
    return tp, n_p - tp, n_g - tp, int(np.count_nonzero(dom)), n_g


def safe_div(a, b):
    return float(a) / float(b) if b > 0 else float("nan")


class Meter:
    """Accumulate one (group, domain) cell over the whole split."""

    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.dom_px = 0
        self.gt_px = 0
        self.n_img = 0

    def add_cell(self, tp, fp, fn, dom_px, gt_px):
        """Merge one already-computed (image, domain) cell -- no pixel work here."""
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.dom_px += dom_px
        self.gt_px += gt_px
        self.n_img += 1

    def row(self):
        return {
            "n_img": self.n_img,
            "dom_px": self.dom_px,
            "gt_px": self.gt_px,
            "base_rate": safe_div(self.gt_px, self.dom_px),
            "P_seg": safe_div(self.tp, self.tp + self.fp),
            "R_seg": safe_div(self.tp, self.tp + self.fn),
            "IoU_seg": safe_div(self.tp, self.tp + self.fp + self.fn),
        }


def evaluate_batch(meters, prob, gt, doms, thr, region):
    """Feed one image into the ALL/CC/SW x valid/agri/inner_apx meters.

    Each (image, domain) cell is computed ONCE and merged into BOTH the ALL meter and
    the region meter. Those two hold identical pixel sums by construction (same image,
    same domain), so computing them separately -- as an earlier version did -- only
    doubled the work.
    """
    pred = prob >= thr
    for dom_name, dom in doms.items():
        cell = cell_stats(pred, gt, dom)
        meters[("ALL", dom_name)].add_cell(*cell)
        meters[(region, dom_name)].add_cell(*cell)


CSV_COLUMNS = [
    "epoch", "split", "reg", "domain", "n_img", "dom_px", "gt_px", "base_rate",
    "P_seg", "R_seg", "IoU_seg",
]


def rows_for_csv(epoch, split, meters):
    out = []
    for dom in DOMAINS:
        for grp in GROUPS:
            r = meters[(grp, dom)].row()
            row = {"epoch": epoch, "split": split, "reg": grp, "domain": dom}
            row.update(r)
            out.append(row)
    return out


def fmt(v):
    if v is None:
        return "-"
    if v != v:            # NaN
        return "nan"
    return "%.4f" % v
