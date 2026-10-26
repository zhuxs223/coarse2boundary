"""Frozen, single-path pseudo-outline anchors; no GT or training targets."""
import numpy as np
from scipy import ndimage as ndi
from skimage.draw import line
from skimage.morphology import skeletonize
from geometry import split_paths, align_path


def single_anchor(p, prior, valid, radius=2):
    paths, _ = split_paths(prior)
    rendered = np.zeros(p.shape, bool)
    confirmations = 0
    max_step = 0.0
    for ids in paths:
        xy = np.column_stack(np.divmod(ids, p.shape[1])).astype(float)
        if len(ids) >= 20:
            d = align_path(p, ids, radius)
            confirmations += int(d['trusted'].sum())
            # Only confirmed teacher locations propose a correction. A short
            # blend at confirmed/unconfirmed transitions returns to the prior.
            shift = np.where(d['trusted'], d['shift'], 0.0)
            shift = ndi.gaussian_filter1d(shift, 1.0, mode='nearest')
            shift[:3] = 0.0
            shift[-3:] = 0.0
            xy = d['prior'] + d['normal'] * shift[:, None]
        coords = np.rint(xy).astype(int)
        coords[:, 0] = np.clip(coords[:, 0], 0, p.shape[0]-1)
        coords[:, 1] = np.clip(coords[:, 1], 0, p.shape[1]-1)
        original = np.column_stack(np.divmod(ids, p.shape[1]))
        bad = ~valid[coords[:, 0], coords[:, 1]]
        coords[bad] = original[bad]
        if len(coords) == 1:
            rendered[tuple(coords[0])] = True
        else:
            max_step = max(max_step, float(np.linalg.norm(np.diff(coords, axis=0), axis=1).max()))
            for a, b in zip(coords[:-1], coords[1:]):
                yy, xx = line(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
                rendered[yy, xx] = True
    center = skeletonize(rendered & valid)
    # Collapse newly created raster-scale loops only. Original small loops and
    # all parcel-sized loops remain, including interiors of real closed paths.
    original_holes = ndi.binary_fill_holes(prior) & ~prior
    old_labels, _ = ndi.label(original_holes)
    old_sizes = np.bincount(old_labels.ravel())
    old_keep = old_sizes <= 9
    old_keep[0] = False
    old_small = old_keep[old_labels]
    near_old_small = ndi.binary_dilation(old_small, iterations=3)
    collapsed = 0
    for _ in range(3):
        hole_labels, _ = ndi.label(ndi.binary_fill_holes(center) & ~center)
        sizes = np.bincount(hole_labels.ravel())
        fill = np.zeros(center.shape, bool)
        for label in np.flatnonzero((sizes > 0) & (sizes <= 9)):
            if label == 0:
                continue
            region = hole_labels == label
            if not (region & near_old_small).any():
                fill |= region
                collapsed += 1
        if not fill.any():
            break
        center = skeletonize(center | fill)
    # Euclidean 1px, not a square 3x3 dilation and not an extra touch dilation.
    band = (ndi.distance_transform_edt(~center) <= 1.0) & valid if center.any() else center.copy()
    return center, band, dict(confirmed_samples=confirmations, max_raster_step=max_step,
                             collapsed_new_microloops=collapsed)


def first_touch_segments(ridge, band):
    paths, _ = split_paths(ridge)
    selected = np.zeros(ridge.shape, bool)
    touched = 0
    records = []
    flat_band = band.ravel()
    for ids in paths:
        n_touch = int(flat_band[ids].sum())
        take = n_touch > 0
        if take:
            selected.ravel()[ids] = True
            touched += 1
        records.append(dict(pixels=len(ids), anchor_touch_pixels=n_touch, selected=take))
    # Compare with unlimited component inheritance, but do not use it as gate.
    labels, _ = ndi.label(ridge, np.ones((3, 3)))
    contacted = np.unique(labels[ridge & band])
    contacted = contacted[contacted > 0]
    component_selected = ridge & np.isin(labels, contacted)
    return selected, component_selected, records


def synthetic_checks():
    shape = (100, 120)
    valid = np.ones(shape, bool)
    prior = np.zeros(shape, bool)
    prior[40, 10:110] = True
    yy = np.arange(shape[0])[:, None]
    p = np.broadcast_to(.95*np.exp(-((yy-42)/1.5)**2), shape).copy().astype(np.float32)
    center, band, _ = single_anchor(p, prior, valid)
    assert ndi.label(center, np.ones((3, 3)))[1] == 1
    assert not (ndi.binary_fill_holes(center) & ~center).any()
    assert center[40, 10] and center[40, 109]
    assert ndi.distance_transform_edt(~prior)[center].max() <= 2.0
    assert np.all(band == (ndi.distance_transform_edt(~center) <= 1))
    blank, _, _ = single_anchor(np.zeros(shape, np.float32), prior, valid)
    assert np.array_equal(blank, prior)
    ring = np.zeros(shape, bool)
    ring[10, 10:91] = ring[90, 10:91] = True
    ring[10:91, 10] = ring[10:91, 90] = True
    _, ring_band, _ = single_anchor(np.zeros(shape, np.float32), ring, valid)
    assert not ring_band[50, 50]
    graph = np.zeros(shape, bool)
    graph[50, 10:91] = True
    graph[50:81, 50] = True
    start = np.zeros(shape, bool)
    start[49:52, 9:12] = True
    selected, component, _ = first_touch_segments(graph, start)
    assert selected[50, 20] and not selected[50, 80] and not selected[70, 50]
    assert component[50, 80] and component[70, 50]
    return dict(shifted_line_is_single_connected=True, no_new_small_loop_on_straight_line=True,
                endpoints_pinned=True, blank_keeps_prior=True, parcel_interior_not_filled=True,
                band_distance_exactly_one=True, no_recursive_fork_expansion=True)
