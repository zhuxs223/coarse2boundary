"""Reviewed independent A/B/C gates, current raw p and legal internal domain only.

Ported unchanged geometric thresholds from the five-probe review. Existing
anchors may veto contained drift, but never provide probability or synthetic
bridge pixels. No GT, historical probe cache or model confidence veto.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from geometry import split_paths

PARAMETERS=dict(
    source='current raw p>=.5, skeletonized inside valid internal nonheld domain',
    A=dict(length=64,chord_arc_min=.93,orthogonal_p95_fraction=.04,orthogonal_p95_floor=3),
    B=dict(length=96,tangent_step=8,turn_p95_max_degrees=20,turn_max_degrees=45),
    C=dict(max_gap=12,max_endpoint_turn=25,min_total_length=96,mutual_best=True,synthetic_bridge_admitted=False),
    junction=dict(turn_max=25,runner_margin=10),
    combine='original touching OR A OR B OR C; ABC seeds one touching pass',
    drift='historical band/exclusion only; contained nonoverlap candidates rejected before seeding',
    support='one-pixel Euclidean band, internal and raw p>.5',
    hold='previous finalized supervision takes priority',
    repeat='fully historical-band-overlapping candidate holds q; original cross-round repeat applies')

def stitched_paths(skel):
    """Join only uniquely aligned path pairs; ambiguous junctions stay split."""
    paths,_=split_paths(skel);ends={};vectors={};links={}
    for i,ids in enumerate(paths):
        if len(ids)<8 or ids[0]==ids[-1]:continue
        xy=np.column_stack(np.divmod(ids,skel.shape[1])).astype(float)
        for side in [0,1]:
            seq=xy if side==0 else xy[::-1]
            v=seq[min(10,len(seq)-1)]-seq[0];v/=max(1e-10,np.linalg.norm(v))
            e=(i,side);vectors[e]=v;ends.setdefault(int(ids[0 if side==0 else -1]),[]).append(e)
    for node,es in ends.items():
        best={}
        for e in es:
            options=sorted((float(np.degrees(np.arccos(np.clip(-vectors[e]@vectors[f],-1,1)))),f)
                           for f in es if f[0]!=e[0])
            if not options:continue
            if options[0][0]<=25 and (len(options)==1 or options[1][0]-options[0][0]>=10):best[e]=options[0][1]
        for e,f in best.items():
            if best.get(f)==e:links[e]=f
    seen=set();chains=[]
    seeds=[(i,s) for i in range(len(paths)) for s in [0,1] if (i,s) not in links]
    seeds+=[(i,0) for i in range(len(paths))]
    for initial in seeds:
        if initial[0] in seen:continue
        parts=[];entry=initial
        while entry[0] not in seen:
            i,side=entry;seen.add(i);ids=paths[i] if side==0 else paths[i][::-1]
            parts.append(ids if not parts else ids[1:]);exit=(i,1-side)
            if exit not in links:break
            entry=links[exit]
        chains.append(np.concatenate(parts))
    return chains


def raster(ids,shape):
    mask=np.zeros(np.prod(shape),bool)
    if len(ids):mask[np.concatenate(ids)]=True
    return mask.reshape(shape)


def measure(ids,shape):
    xy=np.column_stack(np.divmod(ids,shape[1])).astype(float)
    smooth=ndi.gaussian_filter1d(xy,1.5,axis=0,mode='nearest')
    dist=np.r_[0,np.cumsum(np.linalg.norm(np.diff(smooth,axis=0),axis=1))]
    return smooth,dist


def windows(ids,shape,length):
    if len(ids)<length*.8:return
    smooth,dist=measure(ids,shape)
    if dist[-1]<length:return
    starts=list(np.arange(0,dist[-1]-length+.001,16.))
    if not starts or dist[-1]-length-starts[-1]>1:starts.append(dist[-1]-length)
    for start in starts:
        grid=np.arange(start,start+length+.01,1.)
        xy=np.column_stack([np.interp(grid,dist,smooth[:,j]) for j in [0,1]])
        picked=ids[(dist>=start)&(dist<=start+length)]
        yield xy,picked


def straight(xy):
    arc=np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()
    ratio=np.linalg.norm(xy[-1]-xy[0])/max(arc,1e-10)
    centered=xy-xy.mean(axis=0);_,_,v=np.linalg.svd(centered,full_matrices=False)
    deviation=float(np.quantile(np.abs(centered@v[1]),.95))
    return ratio>=.93 and deviation<=max(3,.04*arc)


def smooth_curve(xy):
    # Tangent at 8px scale discounts raster stair steps while retaining bends.
    sampled=xy[::8]
    if len(sampled)<4:return False
    delta=np.diff(sampled,axis=0)
    angles=np.unwrap(np.arctan2(delta[:,0],delta[:,1]))
    turn=np.abs(np.diff(angles))*180/np.pi
    return float(np.quantile(turn,.95))<=20 and float(turn.max())<=45


def path_gate(chains,shape,kind,length,admit=None):
    accepted=[];n=0
    for ids in chains:
        for xy,picked in windows(ids,shape,length):
            if (straight(xy) if kind=='A' else smooth_curve(xy)) and (admit is None or admit(picked)):
                accepted.append(picked);n+=1
    return raster(accepted,shape),n


def gap_gate(chains,skel,domain,admit=None):
    usable=[ids for ids in chains if len(ids)>=10 and ids[0]!=ids[-1]]
    # A gap joins true endpoints only; ordinary junction continuation is already
    # handled by the common stitched-path extractor, not by this gate.
    _,adj=split_paths(skel)
    endpoints=[];points=[];inward=[]
    for i,ids in enumerate(usable):
        xy=np.column_stack(np.divmod(ids,skel.shape[1])).astype(float)
        for side in [0,1]:
            if len(adj.get(int(ids[0 if side==0 else -1]),[]))!=1:continue
            seq=xy if side==0 else xy[::-1]
            vec=seq[min(10,len(seq)-1)]-seq[0];vec/=max(np.linalg.norm(vec),1e-10)
            endpoints.append((i,side));points.append(seq[0]);inward.append(vec)
    if len(points)<2:return np.zeros(skel.shape,bool),np.zeros(skel.shape,bool),0
    points=np.array(points);inward=np.array(inward);tree=cKDTree(points);options={}
    for a,b in sorted(tree.query_pairs(12)):
        if endpoints[a][0]==endpoints[b][0]:continue
        gap=points[b]-points[a];distance=float(np.linalg.norm(gap))
        if distance<1.5:continue
        direction=gap/distance
        aa=float(np.degrees(np.arccos(np.clip((-inward[a])@direction,-1,1))))
        bb=float(np.degrees(np.arccos(np.clip(inward[b]@direction,-1,1))))
        if max(aa,bb)>25:continue
        bridge=np.zeros(skel.shape,np.uint8)
        cv2.line(bridge,tuple(points[a][::-1].astype(int)),tuple(points[b][::-1].astype(int)),1,1)
        if np.any((bridge>0)&~domain):continue
        cost=distance+.2*(aa+bb)
        options.setdefault(a,[]).append((cost,b));options.setdefault(b,[]).append((cost,a))
    best={a:min(opts)[1] for a,opts in options.items()};links={}
    for a,b in best.items():
        if best.get(b)==a:links[endpoints[a]]=endpoints[b]
    seen=set();accepted=[];bridges=np.zeros(skel.shape,np.uint8);n=0
    seeds=[(i,s) for i in range(len(usable)) for s in [0,1] if (i,s) not in links]
    seeds.extend((i,0) for i in range(len(usable)))
    for seed in seeds:
        if seed[0] in seen:continue
        entry=seed;parts=[];used=[];gap_pairs=[]
        while entry[0] not in seen:
            i,side=entry;seen.add(i);ids=usable[i] if side==0 else usable[i][::-1]
            parts.append(ids);used.append(i);exit=(i,1-side)
            if exit not in links:break
            target=links[exit]
            if target[0] in seen:break
            endpoint=usable[target[0]][0 if target[1]==0 else -1]
            gap_pairs.append((int(ids[-1]),int(endpoint)));entry=target
        if not gap_pairs:continue
        ids=np.concatenate(parts);xy,dist=measure(ids,skel.shape)
        if dist[-1]<96 or not (straight(xy) or smooth_curve(xy)):continue
        if admit is not None and not admit(ids):continue
        accepted.append(ids);n+=1
        for a,b in gap_pairs:
            ay,ax=divmod(a,skel.shape[1]);by,bx=divmod(b,skel.shape[1])
            cv2.line(bridges,(ax,ay),(bx,by),1,1)
    # Only existing r1 skeleton pixels are admitted. Bridges are review-only.
    return raster(accepted,skel.shape),bridges.astype(bool)&~skel,n


def build_abc(p,domain,band=None,exclusion=None):
    """Candidate-level historical drift veto before ABC becomes a touch anchor."""
    domain=np.asarray(domain,bool)
    skel=skeletonize((p>=.5)&domain);chains=stitched_paths(skel)
    repeated=np.zeros(p.shape,bool);extending=repeated.copy();contained_center=repeated.copy()
    counters=dict(abc_contained_candidates_rejected=0,abc_overlapping_candidates=0)
    def admit(ids):
        if not len(ids):return False
        overlap=bool(band.ravel()[ids].all()) if band is not None else False
        contained=bool(exclusion.ravel()[ids].all()) if exclusion is not None else False
        if contained and not overlap:
            contained_center.ravel()[ids]=True
            counters['abc_contained_candidates_rejected']+=1
            return False
        (repeated if overlap else extending).ravel()[ids]=True
        counters['abc_overlapping_candidates']+=int(overlap)
        return True
    a,na=path_gate(chains,p.shape,'A',64,admit)
    b,nb=path_gate(chains,p.shape,'B',96,admit)
    c,_,nc=gap_gate(chains,skel,domain,admit)
    gates=dict(A=a,B=b,C=c)
    for center in gates.values():assert not np.any(center&(~domain|~skel))
    return gates,dict(source_skeleton=skel,repeated_center=repeated,
        extending_center=extending,contained_center=contained_center,
        stats=dict(abc_A_windows=na,abc_B_windows=nb,abc_C_chains=nc,
            abc_A_center_px=int(a.sum()),abc_B_center_px=int(b.sum()),
            abc_C_center_px=int(c.sum()),**counters))
