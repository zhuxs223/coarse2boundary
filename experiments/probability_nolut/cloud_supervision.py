"""Teacher inference and identity-mapped probability-only supervision."""
import runtime_setup
import csv,json,time,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
import cfg_standard as C
import run_pair as R
import cloud_data as D
import cloud_geometry as G
from state_store import Reader,Writer

def identity_lut():
    vals=np.arange(65536,dtype=np.uint16).view(np.float16)
    return np.where(np.isfinite(vals)&(vals>=0)&(vals<=1),vals,np.float16(0)).astype(np.float16)

def distribution_lut(original,hist,reference,previous):
    """No-LUT arm always uses the same float16 identity mapping."""
    if original or not C.LUT_ENABLED:return identity_lut()
    if hist.sum():return R.quantile_lut(reference,hist)
    return previous.lut

def mapping_diagnostics(hist,lut):
    """Summaries of nonheld internal float16 bins; no fine labels are read."""
    counts=hist[hist>0].astype(np.float64)
    bins=np.flatnonzero(hist>0).astype(np.uint16)
    raw=bins.view(np.float16).astype(np.float64)
    mapped=lut[bins].astype(np.float64)
    if not counts.size:
        return dict(raw_boundary_mean=None,mapped_boundary_mean=None,
                    raw_positive_fraction=None,mapped_positive_fraction=None,
                    mean_absolute_mapping_shift=None)
    if not np.isfinite(raw).all() or np.any((raw<0)|(raw>1)):
        raise ValueError('Histogram contains invalid response bins')
    total=counts.sum()
    return dict(raw_boundary_mean=float(np.dot(counts,1-raw)/total),
                mapped_boundary_mean=float(np.dot(counts,1-mapped)/total),
                raw_positive_fraction=float(np.dot(counts,raw<=.5)/total),
                mapped_positive_fraction=float(np.dot(counts,mapped<=.5)/total),
                mean_absolute_mapping_shift=float(np.dot(counts,np.abs(mapped-raw))/total))

def preview(root,key,round_no,prob,reference,hold,seed,internal):
    rgb=R.read(C.DATA_ROOT/C.LAYERS['img']/'train'/(key+'.png'),True)
    canvas=Image.new('RGB',(1560,580),'white');draw=ImageDraw.Draw(canvas)
    try:font=ImageFont.truetype('DejaVuSans.ttf',16)
    except OSError:font=ImageFont.load_default()
    reference=prob if reference is None else reference
    overlay=rgb.copy();overlay[hold]=np.rint(.65*overlay[hold]+.35*np.array([230,159,0])).astype(np.uint8);overlay[seed]=[0,114,178]
    panels=[Image.fromarray(np.rint(np.where(internal,x,0)*255).astype(np.uint8)).convert('RGB') for x in (reference,prob)]+[Image.fromarray(overlay)]
    for j,(im,label) in enumerate(zip(panels,['Previous accepted p','Current raw p','Orange: held | Blue: detection core'])):
        draw.text((8+520*j,28),label,font=font,fill='black');canvas.paste(im,(8+520*j,58))
    draw.text((8,5),key,font=font,fill='black')
    path=Path(root)/'guard_preview'/(key+f'_r{round_no:03d}.png');path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)

def rules():return dict(threshold=.5,bonus=C.RIDGE_BONUS,penalty=C.NONRIDGE_PENALTY,rejected_penalty=C.REJECTED_RIDGE_PENALTY)

def validate_probability_contract():
    if C.STRUCTURE_ENABLED or C.LUT_ENABLED or C.REPEAT_CONFIDENCE_ENABLED or C.NO_REPEAT_BONUS:
        raise ValueError('Expected structure, LUT, and structure-dependent repeat updates disabled')
    if rules()!=dict(threshold=.5,bonus=0.,penalty=0.,rejected_penalty=0.):
        raise ValueError('Probability-only targets require q>=.5 and zero geometric adjustments')

def geometry_settings():
    return dict(enabled=False,admission='identity q>=0.5 inside U; independent local hold retained',
                ridge_extraction=False,contact=False,ABC=False,anchor_initialization=False,
                anchor_drift=False,anchor_history=False,geometric_target_adjustment=False,
                structure_dependent_repeat_confidence=False)

def confidence_settings():
    from repeat_confidence import validate
    settings=dict(enabled=C.REPEAT_CONFIDENCE_ENABLED,eta=C.REPEAT_CONFIDENCE_ETA,
        stop=C.REPEAT_CONFIDENCE_STOP,outline_margin=C.REPEAT_CONFIDENCE_OUTLINE_MARGIN,
        tolerance=C.REPEAT_CONFIDENCE_TOLERANCE)
    validate(settings)
    return settings

def build(keys,teacher,state_path,previous,reference,root,device,round_no,signature,smoke=False):
    validate_probability_contract()
    state_path=Path(state_path);root=Path(root)
    if state_path.exists():
        reader=Reader(state_path);info=reader.info
        if info['signature']!=signature or info['keys']!=keys:raise RuntimeError('Cached state inputs differ')
        for name,field in [('guard_candidates','candidate_records'),('geometry_stats','geometry_records')]:
            data=info[field]
            R.upsert_csv(root/f'{name}_r{round_no:03d}.csv',data,list(data[0]),['key'])
        reader.close();print(f'[reuse state r{round_no:03d}]',flush=True);return info
    original=previous is None
    if original:
        net=R.shared('cloud_small_teacher','model.py').build_model(C.RESNET50_W,hidden=64,verbose=False)
    else:net=R.model('cpu')
    ck=R.load_model(net,teacher);teacher_epoch=int(ck['epoch']);del ck;net.to(device).eval()
    settings=dict(search=C.ANCHOR_SEARCH_RADIUS,extra=C.EXCLUSION_EXTRA_RADIUS,band=C.ANCHOR_BAND_RADIUS,
        no_repeat=C.NO_REPEAT_BONUS,guard=C.GUARD_ENABLED,repeat_confidence=confidence_settings())
    prior=Reader(previous) if previous else None
    if prior and prior.keys!=keys:raise RuntimeError('Predecessor keys differ')
    if reference is None and not original:raise RuntimeError('Missing fixed SMALL10 reference')
    hist=np.zeros(65536,np.int64);frozen_tiles=[];rows=[];geometry_rows=[];stats={};worker_seconds=0.;started=time.time();previews=0
    writer=Writer(state_path)
    pool=None
    try:
        workers=1 if smoke else C.GEOMETRY_WORKERS
        if workers>1:
            pool=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=G.initialize,initargs=(str(previous) if previous else None,settings))
        else:G.initialize(str(previous) if previous else None,settings)
        with torch.inference_mode():
            for batch in D.loader(D.Tiles(keys),smoke,inference=True,workers=C.ANCHOR_LOAD_WORKERS):
                ps=torch.sigmoid(net(batch['img'].to(device)))[:,0].float().cpu().numpy()
                if not np.isfinite(ps).all():raise RuntimeError('Non-finite teacher probabilities')
                jobs=[];args=[]
                for j,i in enumerate(batch['idx'].tolist()):
                    a=(i,ps[j],batch['valid'][j].numpy(),batch['msk'][j].numpy(),batch['mk3'][j].numpy());args.append(a)
                    if pool:jobs.append(pool.submit(G.process,a))
                results=[j.result() for j in jobs] if pool else [G.process(a) for a in args]
                for j,res in enumerate(results):
                    i=res['i'];key=keys[i];g=res['g'];hold=res['hold'];saved=res['saved']
                    inside=batch['valid'][j].numpy()&(batch['mk3'][j].numpy()==1)
                    writer.tile(i,res['raw'],g,hold,saved)
                    eligible=inside&~hold;hist+=np.bincount(res['raw'][eligible].view(np.uint16),minlength=65536)
                    if saved is not None:frozen_tiles.append(i)
                    row=dict(key=key,candidate_regions=len(res['candidates']),seed_pixels=int(res['seed'].sum()),held_pixels=int(hold.sum()),
                        retained_positive_pixels=int(saved['soft_mask'].sum()) if saved else 0,candidates=json.dumps(res['candidates']))
                    rows.append(row);geometry_rows.append(dict(key=key,**g['stats']))
                    for k,v in g['stats'].items():stats[k]=stats.get(k,0)+int(v)
                    worker_seconds+=res['seconds']
                    if hold.any() and (previews<C.GUARD_PREVIEW_LIMIT or key in C.PROBE_KEYS):
                        preview(root,key,round_no,ps[j],res['reference'],hold,res['seed'],inside);previews+=1
                done=results[-1]['i']+1
                if done%64<len(results) or done==len(keys):
                    print(f'[state r{round_no:03d}] {done}/{len(keys)} {time.time()-started:.1f}s',flush=True)
        lut=distribution_lut(original,hist,reference,prior)
        summary=dict(round=round_no,teacher_arch='small_h4_full5227' if original else C.STUDENT_ARCH,teacher_epoch=teacher_epoch,
            scanned_tiles=len(keys),candidate_regions=sum(r['candidate_regions'] for r in rows),candidate_tiles=len(frozen_tiles),
            seed_pixels=sum(r['seed_pixels'] for r in rows),held_pixels=sum(r['held_pixels'] for r in rows),
            retained_positive_pixels=sum(r['retained_positive_pixels'] for r in rows),
            retained_negative_pixels=sum(r['held_pixels']-r['retained_positive_pixels'] for r in rows),
            quantile_eligible_pixels=int(hist.sum()),baseline_without_predecessor=original,
            guard_enabled=C.GUARD_ENABLED,lut_enabled=C.LUT_ENABLED,structure_enabled=False,
            mapping_applied=C.LUT_ENABLED and not original,
            **mapping_diagnostics(hist,lut),
            wall_seconds=time.time()-started,worker_seconds=worker_seconds)
        info=dict(complete=True,signature=signature,round=round_no,keys=keys,shape=[C.IMG_SIZE,C.IMG_SIZE],
                  frozen_tiles=frozen_tiles,rules=rules(),repeat_confidence=confidence_settings(),summary=summary,geometry=stats,
                  candidate_records=rows,geometry_records=geometry_rows)
        writer.commit(info,lut,hist)
        for name,data in [('guard_candidates',rows),('geometry_stats',geometry_rows)]:
            R.upsert_csv(root/f'{name}_r{round_no:03d}.csv',data,list(data[0]),['key'])
        R.atomic_json(root/f'guard_stats_r{round_no:03d}.json',summary)
        if original:
            path=root.parent/'reference.npy'
            with path.with_suffix('.npy.tmp').open('wb') as f:np.save(f,hist,allow_pickle=False)
            path.with_suffix('.npy.tmp').replace(path)
        print(f'[guard r{round_no:03d}] regions={summary["candidate_regions"]} tiles={summary["candidate_tiles"]} held_px={summary["held_pixels"]}',flush=True)
        return dict(version=1,**info)
    except BaseException:
        # A committed state is left intact; only this build's uncommitted scratch is removed.
        writer.abort();raise
    finally:
        if pool:pool.shutdown(wait=True,cancel_futures=True)
        if prior:prior.close()
        if G.PREVIOUS is not None:G.PREVIOUS.close();G.PREVIOUS=None
        del net
        if device.type=='cuda':torch.cuda.empty_cache()
