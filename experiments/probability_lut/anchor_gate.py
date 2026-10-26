"""F + ABC: original Steger touching OR ABC; historical exclusion stays authoritative."""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from geometry import split_paths
from steger_safe import steger_fixed
from abc_gate import build_abc


def ridge_identity(p,valid,mask,mk):
    domain=valid&(mask|(mk==2))
    ridge=steger_fixed(p,sigma=2.,lam_th=.01,pad=0)&domain
    labels,_=ndi.label(ridge,np.ones((3,3)))
    sizes=np.bincount(labels.ravel());keep=sizes>=5;keep[0]=False
    closed=ndi.binary_closing(keep[labels],np.ones((3,3)))&domain
    internal=valid&(mk==1)
    support=ndi.binary_dilation(closed,ndi.generate_binary_structure(2,1))&internal
    positive=skeletonize(closed)&domain&(p>.5)
    return support,positive


def center_band(center,radius,valid):
    if not center.any():return np.zeros(center.shape,bool)
    return (ndi.distance_transform_edt(~center)<=radius)&valid


def select(p,valid,mask,mk,anchor_center,extra_radius=3,band_radius=1):
    support,ridge=ridge_identity(p,valid,mask,mk)
    internal=valid&(mk==1)
    anchor_center=np.asarray(anchor_center,bool)&valid
    band=center_band(anchor_center,band_radius,valid)
    exclusion=center_band(band,extra_radius,valid)
    gates,abc=build_abc(p,internal,band,exclusion)
    abc_center=gates['A']|gates['B']|gates['C']
    # ABC expands touching, never the historical drift exclusion. No same-round
    # recursive growth: newly touching Steger paths enter next round's anchors.
    touch_band=band|center_band(abc_center,band_radius,valid)
    paths,_=split_paths(ridge)
    accepted=np.zeros(p.shape,bool);contained_center=accepted.copy()
    repeated=accepted.copy();extending=accepted.copy()
    touching=covered=eligible=overlapping=abc_touching=0
    for ids in paths:
        touch=bool(touch_band.ravel()[ids].any())
        old_touch=bool(band.ravel()[ids].any())
        contained=bool(exclusion.ravel()[ids].all())
        overlap=bool(band.ravel()[ids].all())
        touching+=touch;overlapping+=overlap
        if contained and not overlap:contained_center.ravel()[ids]=True
        if touch and contained and not overlap:covered+=1
        if touch and (not contained or overlap):
            accepted.ravel()[ids]=True;eligible+=1
            abc_touching+=int(not old_touch)
            (repeated if overlap else extending).ravel()[ids]=True
    allow=np.zeros(p.shape,bool);maintained=allow.copy()
    if ridge.any():
        distance,nearest=ndi.distance_transform_edt(~ridge,return_indices=True)
        allow=support&(p>.5)&(distance<=1.)&accepted[nearest[0],nearest[1]]
        allow&=~(contained_center&~accepted)
        repeat_only=repeated&~extending
        maintained=allow&repeat_only[nearest[0],nearest[1]]
    original_bonus=allow&~maintained
    abc_support=center_band(abc_center,1,internal)&(p>.5)
    abc_maintained=np.zeros(p.shape,bool)
    if abc_center.any():
        _,nearest=ndi.distance_transform_edt(~abc_center,return_indices=True)
        repeat_only=abc['repeated_center']&~abc['extending_center']
        abc_maintained=abc_support&repeat_only[nearest[0],nearest[1]]
    # Union once before target construction: hitting multiple gates is one +.1.
    # An extending path takes priority at a shared endpoint, as in the old gate.
    bonus=original_bonus|(abc_support&~abc_maintained)
    allow|=abc_support
    maintained=(maintained|abc_maintained)&~bonus
    support|=abc_support
    steger_accepted=accepted.copy();accepted|=abc_center
    repeated|=abc['repeated_center'];extending|=abc['extending_center']
    contained_center|=abc['contained_center']
    next_center=anchor_center|accepted
    assert not (allow&~support).any()
    assert not (allow&(p<=.5)).any()
    assert not (maintained&~allow).any()
    assert not (anchor_center&~next_center).any()
    assert not (accepted&~valid).any()
    return dict(support=support,positive_ridge=ridge|abc_center,allowed=allow,
        maintained=maintained,repeated_center=repeated,extending_center=extending,
        accepted_center=accepted,next_center=next_center,band=band,exclusion=exclusion,
        touch_band=touch_band,contained_center=contained_center,abc_center=abc_center,
        abc_A=gates['A'],abc_B=gates['B'],abc_C=gates['C'],
        stats=dict(paths=len(paths),touching_paths=touching,contained_touching_paths=covered,
            accepted_paths=eligible,fully_overlapping_paths=overlapping,
            raw_ridge_px=int(ridge.sum()),accepted_center_px=int(accepted.sum()),
            bonus_support_px=int((allow&~maintained).sum()),
            qualified_support_px=int(allow.sum()),maintained_support_px=int(maintained.sum()),
            incoming_anchor_px=int(anchor_center.sum()),next_anchor_px=int(next_center.sum()),
            new_anchor_px=int((next_center&~anchor_center).sum()),
            abc_center_px=int(abc_center.sum()),abc_support_px=int(abc_support.sum()),
            steger_accepted_center_px=int(steger_accepted.sum()),
            abc_seeded_touching_paths=abc_touching,**abc['stats']))


