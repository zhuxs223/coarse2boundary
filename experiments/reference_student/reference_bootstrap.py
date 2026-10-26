"""Read-only baseline import; regenerate pruned r1 supervision, then freeze a reference."""
import contextlib,gc,json,shutil,time
from pathlib import Path
import numpy as np
import torch
import cfg_standard as C
import run_pair as R
import cloud_data as D
import cloud_supervision as S
from state_store import Reader
from storage_budget import reserve,state_bound


@contextlib.contextmanager
def safe_scalars():
    scalar=np.float64(0).__reduce__()[0]
    with torch.serialization.safe_globals([(scalar,'numpy._core.multiarray.scalar'),
            (scalar,'numpy.core.multiarray.scalar'),np.dtype,type(np.dtype(np.float64)),type(np.dtype(np.float32))]):
        yield


def load(path):
    with safe_scalars():return torch.load(path,map_location='cpu',weights_only=True)


def source_checkpoint():return C.BASELINE_ROOT/'round_001/final_r001.pth'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_source(current,smoke):
    """Validate scientific inputs before importing an existing first round."""
    if C.BASELINE_ROOT.resolve()==C.output().resolve():raise ValueError('Baseline and output must differ')
    ckpath=source_checkpoint();marker=C.BASELINE_ROOT/'round_001/done_r001.json'
    for path in (ckpath,marker,C.BASELINE_ROOT/'plan.json',C.BASELINE_ROOT/'reference.npy'):
        if not path.is_file():raise FileNotFoundError(f'Existing baseline input required: {path}')
    old=read(C.BASELINE_ROOT/'plan.json')['signature'];done=read(marker)
    if not done.get('complete') or done['stage']!=C.stage(0):raise ValueError('Baseline r1 is incomplete or has a different stage')
    digest=R.sha(ckpath)
    if digest!=done['checkpoint_sha256']:raise ValueError('Baseline r1 checkpoint hash differs from completion marker')
    for key in ('teacher_sha256','resnet_sha256','train','val','cohort','smoke','split_sha256','data_root','reference_code'):
        if json.loads(json.dumps(current[key]))!=old[key]:raise ValueError(f'Baseline scientific input differs: {key}')
    expected=json.loads(json.dumps(current['config']))
    for key,value in expected.items():
        if key.startswith('reference_'):continue
        previous=old['config'].get(key)
        if key=='lut_enabled' and previous is None:previous=True
        if key=='distribution_mapping' and previous is None:previous='fixed-reference-quantile'
        if value!=previous:raise ValueError(f'Baseline training configuration differs: {key}')
    # Runner bookkeeping differs in a same-policy LUT-control smoke run. All
    # model/data/loss/geometry/history implementations must match the fork source.
    provenance=read(C.HERE/'ablation_source_provenance.json')['baseline_python_sha256']
    runners={'run_all.py','run_windows.py','cloud_supervision.py','inspect_env.py'}
    for key,value in old['code'].items():
        if key not in runners and provenance.get(key)!=value:raise ValueError(f'Baseline core source differs: {key}')
    ck=load(ckpath)
    if ck['epoch']!=C.EPOCHS_PER_ROUND or ck['round']!=1 or ck['plan_signature']!=old:
        raise ValueError('Source checkpoint is not the original baseline r1')
    if not all(key in ck for key in ('model','optimizer','scaler','model_config','diagnostic_history','metric_history')):
        raise ValueError('Source must contain model, optimizer, scaler and original histories')
    cfg=ck['model_config']
    if cfg.get('architecture')!='smp.Unet' or cfg.get('decoder_interpolation')!='bilinear' or cfg.get('reference_model_sha256')!=R.sha(C.MODEL_DIR/'model_unet.py'):
        raise ValueError('Source model architecture/interpolation differs')
    del ck;gc.collect()
    return dict(path=str(ckpath),sha256=digest,epoch=C.EPOCHS_PER_ROUND,
        optimizer_and_scaler='imported unchanged',source_reference_small_sha256=R.sha(C.BASELINE_ROOT/'reference.npy'))


def atomic_array(path,array):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as stream:np.save(stream,array,allow_pickle=False)
    tmp.replace(path)


def copy_exact(source,target):
    source=Path(source);target=Path(target)
    if target.exists():
        if R.sha(source)!=R.sha(target):raise ValueError(f'Existing imported file differs: {target}')
        return
    temporary=target.with_name(target.name+'.tmp')
    shutil.copyfile(source,temporary);temporary.replace(target)


def histogram_stats(hist):
    occupied=np.flatnonzero(hist).astype(np.uint16)
    values=occupied.view(np.float16).astype(np.float64);counts=hist[occupied].astype(np.float64)
    total=int(hist.sum())
    if not total:raise ValueError('Empty reference histogram')
    return dict(pixels=total,positive_fraction=float(counts[values<=.5].sum()/total),
        boundary_mean=float(np.dot(counts,1-values)/total))


def build_student_reference(keys,checkpoint,device,smoke):
    """Full valid training U, raw sigmoid; no GT, geometry, guard or old LUT."""
    net=R.model('cpu');ck=R.load_model(net,checkpoint);del ck
    net.to(device).eval();hist=np.zeros(65536,np.int64);started=time.time()
    with torch.inference_mode():
        for batch in D.loader(D.Tiles(keys),smoke,inference=True):
            ps=torch.sigmoid(net(batch['img'].to(device)))[:,0].float().cpu().numpy()
            if not np.isfinite(ps).all():raise ValueError('Invalid student reference scores')
            for j in range(len(ps)):
                inside=batch['valid'][j].numpy()&(batch['mk3'][j].numpy()==1)
                raw=(1-ps[j][inside]).astype(np.float16)
                hist+=np.bincount(raw.view(np.uint16),minlength=65536)
            done=int(batch['idx'][-1])+1
            if done%128<len(ps) or done==len(keys):print(f'[fixed r1 reference] {done}/{len(keys)} {time.time()-started:.1f}s',flush=True)
    del net;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    return hist


def supervision_check(reader,keys):
    """Verify reconstructed r1 targets against original pre-training diagnostics."""
    totals=dict(inner_px=0,soft_px=0,teacher_candidate_px=0,negative_ignore_px=0)
    target_sum=weight_sum=0.
    for i,batch in enumerate(D.Tiles(keys)):
        inside=batch['valid'].numpy()&(batch['mk3'].numpy()==1)
        diagnostics={};w,s,t,tc=reader.finalized(i,inside,diagnostics=diagnostics)
        totals['inner_px']+=int(inside.sum());totals['soft_px']+=int(s.sum())
        totals['teacher_candidate_px']+=int(tc.sum());totals['negative_ignore_px']+=int(diagnostics['negative_ignore_mask'].sum())
        target_sum+=float(t[s].sum(dtype=np.float64));weight_sum+=float(w[inside].sum(dtype=np.float64))
        if i%512==0:print(f'[r1 supervision audit] {i}/{len(keys)}',flush=True)
    totals['soft_target_mean']=target_sum/max(totals['soft_px'],1)
    totals['mean_inner_negative_weight']=weight_sum/max(totals['inner_px'],1)
    return totals


def prepare(keys,valkeys,signature,device,smoke=False):
    import run_all as A
    marker=A.marker(0);audit_path=C.output()/'reference_bootstrap.json'
    if marker.exists():
        if not audit_path.exists():raise ValueError('Imported first round lacks bootstrap audit')
        audit=read(audit_path)
        if audit['first_student_sha256']!=signature['first_student']['sha256'] or R.sha(A.checkpoint(0))!=audit['first_student_sha256']:
            raise ValueError('Imported first student changed')
        if R.sha(C.output()/'reference.npy')!=audit['selected_reference_sha256']:raise ValueError('Fixed reference changed')
        A.done(0,signature);print('[reference bootstrap] reuse imported r1 and frozen reference',flush=True);return
    root=A.roundroot(0);root.mkdir(parents=True,exist_ok=True)
    source=C.BASELINE_ROOT/'round_001';state=A.statepath(0)
    source_state=source/'state_r001.npz'
    required=(0 if state.exists() else state_bound(len(keys),(C.IMG_SIZE,C.IMG_SIZE)))+512*1024**2
    reserve(C.output(),required,C.MAX_OUTPUT_GIB,C.MIN_FREE_GIB,protected=(state,),keep_rounds=C.STATE_KEEP_ROUNDS)
    copied=source_state.is_file()
    if copied:
        expected=read(source/'done_r001.json')['state_sha256']
        if R.sha(source_state)!=expected:raise ValueError('Baseline state hash differs')
        copy_exact(source_state,state)
    elif not state.exists():
        print('[reference bootstrap] baseline r1 state pruned; reconstruct labels from the same SMALL, no r1 training',flush=True)
        state_sig=dict(plan=signature,stage=C.stage(0),teacher_sha256=R.sha(C.INITIAL_SMALL_TEACHER),predecessor_state_sha256=None)
        S.build(keys,C.INITIAL_SMALL_TEACHER,state,None,None,root,device,1,state_sig,smoke)
    reader=Reader(state)
    if reader.keys!=keys or reader.info['round']!=1:raise ValueError('Wrong initial state')
    if copied and reader.info['signature']['plan']!=read(C.BASELINE_ROOT/'plan.json')['signature']:
        raise ValueError('Copied state scientific signature differs')
    small=reader.archive()['raw_histogram'].copy()
    original_small=np.load(C.BASELINE_ROOT/'reference.npy',allow_pickle=False)
    if small.shape!=(65536,) or small.sum()!=original_small.sum():raise ValueError('Initial histogram domain differs')
    hist_tv=float(np.abs(small.astype(np.float64)-original_small).sum()/(2*small.sum()))
    # Do not silently continue with substantially changed teacher inference.
    if hist_tv>1e-3:raise ValueError(f'Reconstructed SMALL histogram differs (TV={hist_tv:.6f}); inspect runtime/data')
    ck=load(source_checkpoint());original_diag=ck['diagnostic_history'][0]
    totals=supervision_check(reader,keys)
    discrepancies={}
    for key,value in totals.items():
        expected=float(original_diag[key]);difference=abs(value-expected)/max(abs(expected),1)
        discrepancies[key]=dict(original=expected,reconstructed=value,relative_difference=difference)
        if difference>1e-3:raise ValueError(f'Reconstructed r1 supervision differs: {key}, {discrepancies[key]}')
    info=reader.info;reader.close()
    copy_exact(source_checkpoint(),A.checkpoint(0))
    # Exact checkpoint bytes preserve model, AdamW and scaler; the reconstructed
    # state's archive hash can differ from the historical source archive hash.
    for filename,records in [('metrics_r001.csv',ck['metric_history']),('diagnostics_r001.csv',ck['diagnostic_history'])]:
        R.upsert_csv(root/filename,records,list(records[0]),['epoch','split','reg','domain'] if filename.startswith('metrics') else ['global_epoch'])
    imported_source_state_sha=ck['state_sha256'];del ck;gc.collect()
    A.persist_guard(0,info)
    for directory in ('probes','timeline'):
        original=source/directory
        if directory=='probes' and original.is_dir():
            for file in original.rglob('*.png'):
                target=root/directory/file.relative_to(original);target.parent.mkdir(parents=True,exist_ok=True);copy_exact(file,target)
    atomic_array(C.output()/'reference_small.npy',small)
    student_file=C.output()/'reference_student_r1.npy'
    cache_info=C.output()/'reference_student_r1.json'
    expected_cache=dict(first_student_sha256=signature['first_student']['sha256'],keys=keys,
                        reference_domain='all valid mk3==1',quantization='float16(1-raw sigmoid)',model_config='baseline bilinear U-Net')
    if student_file.exists() and cache_info.exists():
        saved=read(cache_info)
        if saved['inputs']!=expected_cache or saved['sha256']!=R.sha(student_file):raise ValueError('Student reference cache differs')
        student=np.load(student_file,allow_pickle=False)
    else:
        student=build_student_reference(keys,A.checkpoint(0),device,smoke)
        if student.shape!=small.shape or student.sum()!=small.sum():raise ValueError('Student/SMALL reference domains differ')
        atomic_array(student_file,student)
        R.atomic_json(cache_info,dict(inputs=expected_cache,sha256=R.sha(student_file),stats=histogram_stats(student)))
    selected=student if C.REFERENCE_MODE=='student_r1' else small
    atomic_array(C.output()/'reference.npy',selected)
    audit=dict(reference_policy=C.REFERENCE_MODE,first_student_sha256=signature['first_student']['sha256'],
        model_optimizer_scaler_imported_byte_identically=True,first_round_retrained=False,
        initial_state_origin='copied original NPZ' if copied else 'reconstructed from identical SMALL',
        initial_histogram_TV=hist_tv,initial_supervision_check=discrepancies,
        initial_state_sha256=R.sha(state),source_checkpoint_state_sha256=imported_source_state_sha,
        small_reference=histogram_stats(small),student_r1_reference=histogram_stats(student),
        selected_reference_sha256=R.sha(C.output()/'reference.npy'),selected_reference_frozen_from_round=2,
        inference_reference_uses_fine_GT=False,reference_update='none after bootstrap')
    R.atomic_json(audit_path,audit)
    R.atomic_json(marker,dict(stage=C.stage(0),plan_signature=signature,
        checkpoint_sha256=R.sha(A.checkpoint(0)),state_name=state.name,state_sha256=R.sha(state),
        imported_baseline_r1=True,complete=True))
    print('[reference bootstrap]',json.dumps(audit),flush=True)
