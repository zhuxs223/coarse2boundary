"""Numerically safe Steger extraction preserving the existing accepted pixels."""
import numpy as np
from scipy.ndimage import gaussian_filter

def steger_fixed(P,sigma=2.0,lam_th=0.01,pad=0):
    P=np.asarray(P,dtype=np.float32)
    if P.ndim!=2 or not np.isfinite(P).all():
        raise ValueError('Steger expects a finite 2D probability field')
    H,W=P.shape
    Ixx=gaussian_filter(P,sigma,order=(0,2))
    Ixy=gaussian_filter(P,sigma,order=(1,1))
    Iyy=gaussian_filter(P,sigma,order=(2,0))
    Ix=gaussian_filter(P,sigma,order=(0,1))
    Iy=gaussian_filter(P,sigma,order=(1,0))
    tr=Ixx+Iyy
    disc=np.sqrt(np.maximum((Ixx-Iyy)**2+4*Ixy**2,0))
    lam1=(tr-disc)/2.0
    resp=np.abs(lam1)-np.abs((tr+disc)/2.0)
    vx,vy=Ixy,lam1-Ixx
    norm=np.sqrt(vx**2+vy**2)+1e-10
    nx,ny=vx/norm,vy/norm
    denominator=lam1+1e-10
    numerator=-(Ix*nx+Iy*ny)
    eligible=(resp>lam_th)&np.isfinite(denominator)&(denominator!=0)&np.isfinite(numerator)
    # Keep the original division and epsilon for eligible points. Flat/rejected
    # points need no division, and will never be converted to integer coordinates.
    t=np.full(P.shape,np.inf,dtype=np.float32)
    np.divide(numerator,denominator,out=t,where=eligible)
    valid=eligible&np.isfinite(t)&(np.abs(t)<=0.5)
    if pad>0:
        valid[:pad,:]=False;valid[-pad:,:]=False
        valid[:,:pad]=False;valid[:,-pad:]=False
    y,x=np.nonzero(valid)
    yy=np.clip(np.round(y+t[valid]*ny[valid]).astype(np.intp),0,H-1)
    xx=np.clip(np.round(x+t[valid]*nx[valid]).astype(np.intp),0,W-1)
    ridge=np.zeros((H,W),bool)
    ridge.flat[np.unique(yy*W+xx)]=True
    return ridge
