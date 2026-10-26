"""Pseudo-outline guided probability-band alignment; analysis only, no GT."""
import numpy as np
from scipy import ndimage as ndi

def split_paths(skel):
    h,w=skel.shape;nodes=set(np.flatnonzero(skel));adj={n:[] for n in nodes}
    for n in nodes:
        y,x=divmod(n,w)
        for dy,dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
            yy,xx=y+dy,x+dx
            if not (0<=yy<h and 0<=xx<w):continue
            j=yy*w+xx
            if j not in nodes:continue
            if dx and dy and (skel[y,xx] or skel[yy,x]):continue
            adj[n].append(j)
    seen=set();paths=[]
    def edge(a,b):return (min(a,b),max(a,b))
    def walk(a,b):
        path=[a,b];seen.add(edge(a,b));prev,cur=a,b
        while len(adj[cur])==2:
            nxt=next(j for j in adj[cur] if j!=prev)
            if edge(cur,nxt) in seen:break
            seen.add(edge(cur,nxt));path.append(nxt);prev,cur=cur,nxt
        return np.asarray(path)
    for n in sorted(nodes):
        if len(adj[n])==0:paths.append(np.asarray([n]))
        elif len(adj[n])!=2:
            for j in adj[n]:
                if edge(n,j) not in seen:paths.append(walk(n,j))
    for n in sorted(nodes):
        for j in adj[n]:
            if edge(n,j) not in seen:paths.append(walk(n,j))
    return paths,adj

def path_frame(ids,shape):
    y,x=np.divmod(ids,shape[1]);xy=np.column_stack([y,x]).astype(float)
    # Broad tangent estimate follows the pseudo-outline, rather than noisy ridges.
    z=ndi.gaussian_filter1d(xy,1.5,axis=0,mode='nearest')
    tangent=np.gradient(z,axis=0);length=np.linalg.norm(tangent,axis=1)
    unit=tangent/np.maximum(length[:,None],1e-8)
    normal=np.column_stack([-unit[:,1],unit[:,0]])
    return xy,normal

def profile_centers(profile,offsets,radius):
    n=len(profile);center=np.zeros(n);peak=np.zeros(n);width=np.zeros(n);amb=np.zeros(n,bool);matched=np.zeros(n,bool);reason=np.zeros(n,np.uint8)
    for i,p in enumerate(profile):
        bits=p>=.5
        edges=np.flatnonzero(np.diff(np.r_[False,bits,False].astype(np.int8)))
        candidates=[];allbands=[]
        for lo,hi in edges.reshape(-1,2):
            ss=offsets[lo:hi];pp=p[lo:hi]
            c=float(np.average(ss,weights=pp**2));pk=float(pp.max());wd=float((hi-lo)*.5)
            allbands.append((abs(c),c,pk,wd,lo==0 or hi==len(p)))
            if abs(c)>radius or lo==0 or hi==len(p) or wd>8:continue
            candidates.append((abs(c),c,pk,wd))
        if not candidates:
            if not allbands:reason[i]=1 # no threshold response
            else:
                near=min(allbands)
                reason[i]=3 if near[4] else (2 if near[3]>8 else 4)
            continue
        candidates.sort();_,c,pk,wd=candidates[0]
        center[i]=c;peak[i]=pk;width[i]=wd;matched[i]=True
        # Separate response bands of similar strength near the same prior are ambiguous.
        amb[i]=any(abs(other[2]-pk)<.1 and abs(other[1]-c)>=2 for other in candidates[1:])
    return center,peak,width,amb,matched,reason

def smooth_offsets(proposals,confidence,matched,radius):
    states=np.arange(-radius,radius+.25,.5);n=len(proposals);k=len(states)
    # Unmatched positions follow the prior locally but remain explicitly untrusted.
    cost=np.where(matched[:,None],confidence[:,None]*(states[None,:]-proposals[:,None])**2,.1*states[None,:]**2)
    cost+=.02*states[None,:]**2
    delta=states[:,None]-states[None,:]
    trans=np.where(np.abs(delta)<=1,.35*delta**2,np.inf)
    score=cost[0].copy();back=np.zeros((n,k),np.int16)
    for i in range(1,n):
        v=score[:,None]+trans;back[i]=np.argmin(v,axis=0);score=cost[i]+v[back[i],np.arange(k)]
    ix=int(score.argmin());path=np.empty(n)
    for i in range(n-1,-1,-1):path[i]=states[ix];ix=int(back[i,ix])
    return path

def align_path(p,ids,radius=2):
    xy,normal=path_frame(ids,p.shape)
    profile_offsets=np.arange(-6,6.25,.5)
    coords=xy[:,None,:]+normal[:,None,:]*profile_offsets[None,:,None]
    profiles=ndi.map_coordinates(p,[coords[:,:,0].ravel(),coords[:,:,1].ravel()],order=1,mode='nearest').reshape(len(ids),-1)
    c,peak,width,amb,matched,reason=profile_centers(profiles,profile_offsets,radius)
    shift=smooth_offsets(c,peak,matched,radius)
    fitted=xy+normal*shift[:,None]
    fitted_p=ndi.map_coordinates(p,[fitted[:,0],fitted[:,1]],order=1,mode='nearest')
    trusted=matched&~amb&(fitted_p>=.7)&(np.abs(shift-c)<=1)
    trusted[:3]=False;trusted[-3:]=False # endpoints/junctions retained as transition, not trusted anchors
    return dict(prior=xy,normal=normal,fitted=fitted,shift=shift,proposal=c,
        matched=matched,ambiguous=amb,trusted=trusted,prior_p=p.ravel()[ids],fitted_p=fitted_p,
        profile_width=width,profile_peak=peak,reject_reason=reason)
