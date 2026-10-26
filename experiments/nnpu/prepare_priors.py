"""One-time Oracle preprocessing; serialize scalars and hashes, never GT positions."""
import runtime_setup
import argparse
import json
import time
import numpy as np
import cfg_standard as C
import data as D
import utils as U
from dataset_protocol import load_targets, key_digest

DOMAIN = 'valid=(1px-interior AND NOT(rgb-black AND NOT coarse-mask)); U=valid AND mk3==1'


def prepare(keys, path):
    start = time.perf_counter()
    records = []
    for i,key in enumerate(keys):
        item = D.Tiles([key],evaluation=True)[0]
        valid, mk3, gt = item['valid'].numpy(), item['mk3'].numpy(), item['bnd'].numpy()
        u = valid & (mk3==1)
        count, truth = int(u.sum()), int((u&gt).sum())
        hashes = {name: U.sha(C.DATA_ROOT/layer/'train'/(key+'.png')) for name,layer in C.LAYERS.items()}
        records.append(dict(key=key,positive=int((valid & (mk3==2)).sum()),unlabeled=count,
                            gt_in_U=truth,pi=truth/count if count else 0.,inputs_sha256=hashes))
        if (i+1)%256==0 or i+1==len(keys):
            print(f'[Oracle prepare] {i+1}/{len(keys)} {time.perf_counter()-start:.1f}s',flush=True)
    payload = dict(format=1,protocol='nnpu-oracle-U-v1',domain=DOMAIN,
        keys_sha256=key_digest(keys),split_sha256=U.sha(C.SPLIT_JSON),
        image_size=C.IMG_SIZE,edge_trim=C.EDGE_TRIM,extra_information='true per-tile boundary prevalence in U',
        fine_positions_serialized=False,records=records,
        summary=dict(tiles=len(keys),no_P_tiles=sum(r['positive']==0 for r in records),
            no_U_tiles=sum(r['unlabeled']==0 for r in records),
            pi_mean=float(np.mean([r['pi'] for r in records if r['unlabeled']])) if any(r['unlabeled'] for r in records) else 0.,
            U_pixels=sum(r['unlabeled'] for r in records),gt_in_U=sum(r['gt_in_U'] for r in records)))
    U.json_write(path,payload)
    print('[Oracle summary]',json.dumps(payload['summary']),flush=True)
    return load_priors(keys,path)


def load_priors(keys,path):
    payload = json.loads(path.read_text(encoding='utf-8'))
    if (payload['protocol']!='nnpu-oracle-U-v1' or payload['domain']!=DOMAIN
        or payload['keys_sha256']!=key_digest(keys) or payload['split_sha256']!=U.sha(C.SPLIT_JSON)
        or payload['image_size']!=C.IMG_SIZE or payload['edge_trim']!=C.EDGE_TRIM):
        raise RuntimeError('Oracle prior protocol/keys/domain differs')
    records=payload['records']
    if [r['key'] for r in records]!=list(keys):
        raise RuntimeError('Oracle prior records/order differs')
    for r in records:
        n,gt,p = r['unlabeled'],r['gt_in_U'],r['positive']
        if not all(isinstance(v,int) for v in (n,gt,p)) or not 0<=gt<=n or p<0:
            raise ValueError('Invalid Oracle counts')
        expected=gt/n if n else 0.
        if not np.isfinite(r['pi']) or abs(r['pi']-expected)>1e-12:
            raise ValueError('Oracle pi/count mismatch')
    # Only these scalars are passed into training workers; GT hashes/counts are provenance.
    return {r['key']:{k:r[k] for k in ('pi','positive','unlabeled')} for r in records}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true',help='Recompute all input hashes against prepared priors')
    args=parser.parse_args()
    runtime_setup.configure()
    keys,_,_,_=load_targets(C.SPLIT_JSON)
    with U.run_lock(C.output()):
        path=C.priors_path()
        if path.exists():
            load_priors(keys,path)
            if args.verify:
                payload=json.loads(path.read_text(encoding='utf-8'))
                for r in payload['records']:
                    for name,layer in C.LAYERS.items():
                        if U.sha(C.DATA_ROOT/layer/'train'/(r['key']+'.png'))!=r['inputs_sha256'][name]:
                            raise RuntimeError('Input changed: '+r['key']+' '+name)
            print('[Oracle] existing priors verified; not overwritten:',path,flush=True)
        else:
            if (C.output()/'latest.pth').exists():
                raise RuntimeError('Prior file missing from existing experiment; restore original rather than rebuilding')
            prepare(keys,path)


if __name__=='__main__':
    main()
