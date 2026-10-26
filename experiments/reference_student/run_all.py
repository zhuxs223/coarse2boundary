"""Cloud entry point: round-final checkpoints only, restart interrupted rounds."""
import runtime_setup
import argparse,csv,gc,json,math,os,time
from pathlib import Path
import numpy as np
import torch
import cfg_standard as C
import run_pair as R
import cloud_data as D
import cloud_supervision as S
import weighted_loss as W
from abc_gate import PARAMETERS as ABC_PARAMETERS
from target_rules import NEGATIVE_IGNORE_LOW,NEGATIVE_IGNORE_HIGH
from state_store import Reader
from storage_budget import reserve,state_bound
from dataset_protocol import load_targets,smoke_targets

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def label(index):return f'r{index+1:03d}'
def roundroot(index):return C.output()/f'round_{index+1:03d}'
def statepath(index):return roundroot(index)/f'state_{label(index)}.npz'
def checkpoint(index):return roundroot(index)/f'final_{label(index)}.pth'
def marker(index):return roundroot(index)/f'done_{label(index)}.json'

def configuration():
    return dict(protocol='fullteacher-student-abc-full5227-v1',student_arch=C.STUDENT_ARCH,
        data_protocol=C.DATA_PROTOCOL,metric_split=C.METRIC_SPLIT,
        evaluate_every_epochs=C.EVALUATE_EVERY_EPOCHS,model_selection='fixed-round-budget',
        epochs_per_round=C.EPOCHS_PER_ROUND,lut_enabled=C.LUT_ENABLED,
        distribution_mapping='fixed-reference-quantile' if C.LUT_ENABLED else 'float16-identity',
        reference_policy=C.REFERENCE_MODE,reference_domain='all valid training-U pixels; no fine GT',
        reference_frozen_from_round=2,
        batch=C.BATCH_SIZE,infer_batch=C.INFER_BATCH,
        lr=C.LR,weight_decay=C.WEIGHT_DECAY,seed=C.SEED,amp=C.AMP,
        outside_weights=C.OUTSIDE_WEIGHTS,first_soft=C.FIRST_SOFT_WEIGHT,soft=C.SOFT_WEIGHT,pseudo=C.PSEUDO_WEIGHT,
        search=C.ANCHOR_SEARCH_RADIUS,band=C.ANCHOR_BAND_RADIUS,extra=C.EXCLUSION_EXTRA_RADIUS,
        rules=S.rules(),guard=C.GUARD_ENABLED,guard_settings=__import__('gray_region_guard').SETTINGS,
        no_repeat=C.NO_REPEAT_BONUS,repeat_confidence=S.confidence_settings(),
        geometry=ABC_PARAMETERS,negative_ignore=[float(NEGATIVE_IGNORE_LOW),float(NEGATIVE_IGNORE_HIGH)],
        interpolation=C.DECODER_INTERPOLATION,init='ImageNet ResNet50 encoder + seeded random decoder',
        layers=C.LAYERS,trim=C.EDGE_TRIM,shape=C.IMG_SIZE,mean=C.IMAGENET_MEAN,std=C.IMAGENET_STD)

def plan(smoke=False):
    train,val,groups,cohort=load_targets(C.SPLIT_JSON,C.EXPECTED_TARGET_TILES)
    if C.DATA_PROTOCOL!=cohort['protocol']:raise ValueError('Configured data protocol differs')
    if C.EVALUATE_EVERY_EPOCHS<1:raise ValueError('Evaluation interval must be positive')
    if smoke:
        train,val=smoke_targets(groups)
    if not train or not val or len(set(train))!=len(train) or len(set(val))!=len(val):raise ValueError('Invalid split')
    for key in dict.fromkeys(train+val):
        if '..' in key.split('/') or key.startswith('/') or '\\' in key:raise ValueError('Unsafe dataset key')
        for layer in C.LAYERS.values():
            if not (C.DATA_ROOT/layer/'train'/(key+'.png')).is_file():raise FileNotFoundError(f'{layer}: {key}')
    C.output().mkdir(parents=True,exist_ok=True)
    code={p.name:R.sha(p) for p in C.HERE.glob('*.py') if p.name!='cfg_standard.py' and not p.name.startswith('test_')}
    reference_code={p.name:R.sha(p) for p in C.MODEL_DIR.glob('*.py')}
    sig=dict(config=configuration(),teacher_sha256=R.sha(C.INITIAL_SMALL_TEACHER),resnet_sha256=R.sha(C.RESNET50_W),
        train=train,val=val,cohort=cohort,smoke=smoke,split_sha256=R.sha(C.SPLIT_JSON),
        data_root=str(C.DATA_ROOT),code=code,reference_code=reference_code)
    import reference_bootstrap as B
    sig['first_student']=B.validate_source(sig,smoke)
    # Compare normalized JSON types; tuples serialize as lists.
    sig=json.loads(json.dumps(sig))
    p=C.output()/'plan.json'
    if p.exists() and read(p)['signature']!=sig:raise RuntimeError('Scientific inputs/code changed: use a NEW RUN_NAME')
    R.atomic_json(p,dict(signature=sig,rounds_requested=C.ROUNDS))
    return train,val,sig

def precheck(smoke=False):
    from teacher_protocol import validate_teacher
    teacher_info=validate_teacher()
    train,val,sig=plan(smoke)
    ck=torch.load(C.INITIAL_SMALL_TEACHER,map_location='cpu',weights_only=True)
    if ck.get('epoch')!=10:raise ValueError('SMALL teacher must be ep10')
    net=R.shared('precheck_small','model.py').build_model(C.RESNET50_W,hidden=64,verbose=False)
    net.load_state_dict(ck['model'],strict=True);del net,ck
    R.seed(C.SEED);net=R.model('cpu');cfg=R.model_config(net)
    if cfg['decoder_interpolation']!=C.DECODER_INTERPOLATION or any(isinstance(m,torch.nn.ConvTranspose2d) for m in net.modules()):raise ValueError('Wrong decoder')
    expected_params=2145857 if C.STUDENT_ARCH=='h8' else 32521105
    if sum(p.numel() for p in net.parameters())!=expected_params:raise ValueError('Unexpected student architecture')
    assert 'bnd' not in D.Tiles(train[:1])[0]
    params=sum(p.numel() for p in net.parameters());del net;gc.collect()
    steps=math.ceil(len(train)/C.BATCH_SIZE)*C.EPOCHS_PER_ROUND
    info=dict(train=len(train),evaluation=len(val),metric_split=C.METRIC_SPLIT,
        train_eval_overlap=len(set(train)&set(val)),cohort=sig['cohort'],
        student_start_epoch=C.EPOCHS_PER_ROUND,teacher_epoch=10,teacher=teacher_info,student_arch=C.STUDENT_ARCH,student_parameters=params,rounds=C.ROUNDS,
        batch=C.BATCH_SIZE,epochs_per_round=C.EPOCHS_PER_ROUND,optimizer_steps_per_round=steps,
        optimizer_steps_per_epoch=math.ceil(len(train)/C.BATCH_SIZE),
        optimizer_steps_total=steps*C.ROUNDS,new_optimizer_steps=steps*max(0,C.ROUNDS-1),
        first_student=sig['first_student'],reference_policy=C.REFERENCE_MODE,evaluate_every_epochs=C.EVALUATE_EVERY_EPOCHS,
        state_upper_GiB=state_bound(len(train),(C.IMG_SIZE,C.IMG_SIZE))/1024**3,
        output=str(C.output()),lut_enabled=C.LUT_ENABLED,
        distribution_mapping='fixed-reference-quantile' if C.LUT_ENABLED else 'float16-identity',
        pseudo_weight=C.PSEUDO_WEIGHT,repeat_confidence=S.confidence_settings(),
        geometry=ABC_PARAMETERS,negative_ignore=[float(NEGATIVE_IGNORE_LOW),float(NEGATIVE_IGNORE_HIGH)],
        runtime=runtime_setup.SETTINGS,model_config=cfg)
    R.atomic_json(C.output()/'precheck.json',info);print('[precheck]',json.dumps(info),flush=True)
    return train,val,sig

def done(index,sig):
    path=marker(index)
    if not path.exists():return False
    info=read(path)
    if info['plan_signature']!=sig or info['stage']!=C.stage(index):raise RuntimeError('Completed round config changed')
    p=checkpoint(index)
    if not p.exists() or R.sha(p)!=info['checkpoint_sha256']:raise RuntimeError('Completed checkpoint missing/changed')
    print(f'[reuse completed {label(index)}]',flush=True);return True

def persist_guard(index,info):
    root=roundroot(index);s=info['summary']
    R.atomic_json(root/f'guard_stats_{label(index)}.json',s)
    R.upsert_csv(C.output()/'guard_summary.csv',[s],list(s),['round'])

def probes(net,keys,index,epoch,device,smoke):
    net.eval();chosen=keys if smoke else [k for k in C.PROBE_KEYS if k in keys]
    if not chosen:return
    arrays={};root=roundroot(index)/'probes'
    import cv2
    with torch.inference_mode():
        for batch in D.loader(D.Tiles(chosen),smoke,inference=True):
            p=torch.sigmoid(net(batch['img'].to(device)))[:,0].float().cpu().numpy()
            for j,i in enumerate(batch['idx'].tolist()):
                key=chosen[i];valid=batch['valid'][j].numpy();mk=batch['mk3'][j].numpy();mask=batch['msk'][j].numpy()
                q=np.where(valid&(mask|(mk==2)),p[j],0);arrays[f'prob_{i:03d}']=q
                dest=root/(key+f'_{label(index)}_ep{epoch:03d}.png');dest.parent.mkdir(parents=True,exist_ok=True)
                if not cv2.imwrite(str(dest),np.rint(q*255).clip(0,255).astype(np.uint8)):raise OSError(dest)
    arrays['keys']=np.asarray(chosen)
    with (roundroot(index)/f'probe_probs_{label(index)}.npz').open('wb') as f:np.savez_compressed(f,**arrays)

def complete(index,ck,info,sig):
    root=roundroot(index);r=label(index)
    R.upsert_csv(root/f'metrics_{r}.csv',ck['metric_history'],['round','epoch_in_round']+R.A.CSV_COLUMNS,['epoch','split','reg','domain'])
    R.upsert_csv(root/f'diagnostics_{r}.csv',ck['diagnostic_history'],list(ck['diagnostic_history'][0]),['global_epoch'])
    persist_guard(index,info)
    R.atomic_json(marker(index),dict(stage=C.stage(index),plan_signature=sig,checkpoint_sha256=R.sha(checkpoint(index)),
        state_name=statepath(index).name,state_sha256=ck['state_sha256'],complete=True))

def train_round(index,trainkeys,valkeys,sig,device,smoke=False):
    stage=C.stage(index);root=roundroot(index);root.mkdir(parents=True,exist_ok=True);r=label(index)
    for target in (statepath(index).with_name(statepath(index).name+'.building'),checkpoint(index).with_name(checkpoint(index).name+'.tmp')):
        if target.exists():
            if target.is_symlink() or target.parent.resolve()!=root.resolve():raise RuntimeError('Unsafe unfinished-state cleanup')
            target.unlink();print('[removed uncommitted scratch]',target.name,flush=True)
    previous=statepath(index-1) if index else None;teacher=checkpoint(index-1) if index else C.INITIAL_SMALL_TEACHER
    state_sig=dict(plan=sig,stage=stage,teacher_sha256=R.sha(teacher),
        predecessor_state_sha256=R.sha(previous) if previous else None)
    refpath=C.output()/'reference.npy'
    reference=np.load(refpath,allow_pickle=False) if refpath.exists() else None
    if index and reference is None:raise RuntimeError('Missing fixed reference')
    required=(0 if statepath(index).exists() else state_bound(len(trainkeys),(C.IMG_SIZE,C.IMG_SIZE)))+512*1024**2
    reserve(C.output(),required,C.MAX_OUTPUT_GIB,C.MIN_FREE_GIB,protected=(previous,statepath(index)),keep_rounds=C.STATE_KEEP_ROUNDS)
    info=S.build(trainkeys,teacher,statepath(index),previous,reference,root,device,index+1,state_sig,smoke)
    reader=Reader(statepath(index))
    if index==0:
        if not refpath.exists():
            with refpath.open('wb') as f:np.save(f,reader.archive()['raw_histogram'],allow_pickle=False)
        elif not np.array_equal(np.load(refpath,allow_pickle=False),reader.archive()['raw_histogram']):raise RuntimeError('Fixed reference changed')
    reader.close();persist_guard(index,info)
    state_sha=R.sha(statepath(index));final=checkpoint(index)
    if final.exists():
        ck=torch.load(final,map_location='cpu',weights_only=True)
        if ck['epoch']!=stage['end'] or ck['state_sha256']!=state_sha or ck['plan_signature']!=sig:raise RuntimeError('Invalid uncommitted final')
        complete(index,ck,info,sig);print(f'[recover committed final {r}]',flush=True);return
    # Interrupted rounds have no mid-round checkpoint. Rebuild initialization and logs.
    for name in (f'metrics_{r}.csv',f'diagnostics_{r}.csv'):
        path=root/name
        if path.exists():path.unlink()
    R.seed(C.SEED);C.START_EPOCH=stage['start'];net=R.model(device)
    opt=torch.optim.AdamW(net.parameters(),lr=C.LR,weight_decay=C.WEIGHT_DECAY)
    scaler=torch.amp.GradScaler('cuda',enabled=C.AMP and device.type=='cuda')
    if index:
        ck=R.load_model(net,teacher)
        if ck['epoch']!=stage['start']:raise RuntimeError('Predecessor epoch differs')
        opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler']);del ck
    # No student10 load: fresh ImageNet encoder and random decoder in round 1.
    R.loader=D.loader
    def loss(logits,batch):return W.losses(logits,batch,stage['alpha'],C.PSEUDO_WEIGHT,C.FIRST_SOFT_WEIGHT if index==0 else C.SOFT_WEIGHT,C.OUTLINE_TARGET)
    R.losses=loss
    ds=D.Tiles(trainkeys,statepath(index));val=D.Tiles(valkeys,evaluation=True);history=[];diagnostics=[]
    for ep in range(1,stage['epochs']+1):
        diag=R.train_epoch(net,opt,scaler,ds,device,smoke,ep);epoch=stage['start']+ep
        if diag['soft_target_below_half_px']:raise RuntimeError('Targets below .5')
        score=epoch%C.EVALUATE_EVERY_EPOCHS==0 or ep==stage['epochs']
        metrics=([dict(round=index+1,epoch_in_round=ep,**x) for x in
            R.A.rows_for_csv(epoch,C.METRIC_SPLIT,R.evaluate(net,val,device,smoke))] if score else [])
        history+=metrics;diag=dict(round=index+1,epoch_in_round=ep,global_epoch=epoch,outside_weight=stage['alpha'],**diag);diagnostics.append(diag)
        if metrics:R.upsert_csv(root/f'metrics_{r}.csv',metrics,['round','epoch_in_round']+R.A.CSV_COLUMNS,['epoch','split','reg','domain'])
        R.upsert_csv(root/f'diagnostics_{r}.csv',[diag],list(diag),['global_epoch'])
        if metrics:
            main=next(x for x in metrics if x['reg']=='ALL' and x['domain']=='agri')
            print(f'[{r} student ep{epoch} {C.METRIC_SPLIT}] P={main["P_seg"]:.4f} R={main["R_seg"]:.4f} IoU={main["IoU_seg"]:.4f}',flush=True)
        else:print(f'[{r} student ep{epoch}] full-target metrics deferred to scoring epoch',flush=True)
    probes(net,trainkeys,index,stage['end'],device,smoke)
    ck=dict(model=net.state_dict(),model_config=R.model_config(net),optimizer=opt.state_dict(),scaler=scaler.state_dict(),
        epoch=stage['end'],round=index+1,metric_history=history,diagnostic_history=diagnostics,state_sha256=state_sha,plan_signature=sig)
    reserve(C.output(),512*1024**2,C.MAX_OUTPUT_GIB,C.MIN_FREE_GIB,protected=(previous,statepath(index)),keep_rounds=C.STATE_KEEP_ROUNDS)
    R.atomic_save(final,ck)
    complete(index,ck,info,sig)
    if ds._state:ds._state.close()
    del ck,net,opt,scaler,ds;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    reserve(C.output(),0,C.MAX_OUTPUT_GIB,C.MIN_FREE_GIB,protected=(statepath(index),),keep_rounds=C.STATE_KEEP_ROUNDS)

def report():
    rows=[]
    for path in sorted(C.output().glob('round_*/metrics_r*.csv')):
        with path.open(newline='',encoding='utf-8') as f:rows+=list(csv.DictReader(f))
    if rows:R.upsert_csv(C.output()/'summary.csv',rows,list(rows[0]),['epoch','split','reg','domain'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,domain in zip(axes,['agri','inner_apx']):
        selected=[x for x in rows if x['reg']=='ALL' and x['domain']==domain]
        for field,color in [('P_seg','#0072B2'),('R_seg','#E69F00'),('IoU_seg','#8B5FBF')]:
            ax.plot([int(x['epoch']) for x in selected],[float(x[field]) for x in selected],label=field,color=color)
        ax.set(title=f'{C.METRIC_SPLIT} (in-sample) / {domain}',xlabel='Student epoch',ylim=(0,1));ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(C.output()/'metric_trend.png',dpi=140);plt.close(fig)

def acquire_lock(root):
    path=root/'.run.lock'
    if path.exists():
        owner=read(path)
        if process_alive(int(owner['pid'])):
            raise RuntimeError('Output already has a live writer. Do not run two launchers into one RUN_NAME.')
        path.unlink()
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:json.dump(dict(pid=os.getpid(),time=time.time()),f)
    return path

def process_alive(pid):
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=(wintypes.DWORD,wintypes.BOOL,wintypes.DWORD);kernel.OpenProcess.restype=wintypes.HANDLE
        handle=kernel.OpenProcess(0x1000,False,pid)
        if not handle:return ctypes.get_last_error()==5  # access denied may be a live process
        code=wintypes.DWORD()
        kernel.GetExitCodeProcess.argtypes=(wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD))
        kernel.CloseHandle.argtypes=(wintypes.HANDLE,)
        try:return bool(kernel.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value==259
        finally:kernel.CloseHandle(handle)
    try:os.kill(pid,0)
    except ProcessLookupError:return False
    except PermissionError:return True
    return True

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--precheck',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda');a=p.parse_args()
    runtime_setup.configure()
    capacity=runtime_setup.cpu_capacity(C.CPU_LIMIT)
    C.NUM_WORKERS=min(C.NUM_WORKERS,max(0,capacity-2))
    C.ANCHOR_LOAD_WORKERS=min(C.ANCHOR_LOAD_WORKERS,max(0,capacity-2))
    C.GEOMETRY_WORKERS=min(C.GEOMETRY_WORKERS,max(1,capacity-C.ANCHOR_LOAD_WORKERS-2))
    print(f'[CPU] cap={capacity} train_workers={C.NUM_WORKERS} geometry_workers={C.GEOMETRY_WORKERS} anchor_load_workers={C.ANCHOR_LOAD_WORKERS}',flush=True)
    if a.smoke:
        C.RUN_NAME='_smoke_'+C.RUN_NAME+'_'+C.SMOKE_RUN_ID;C.ROUNDS=3;C.EPOCHS_PER_ROUND=2;C.BATCH_SIZE=C.INFER_BATCH=2
        C.NUM_WORKERS=C.ANCHOR_LOAD_WORKERS=0;C.GEOMETRY_WORKERS=1;C.OUTSIDE_WEIGHTS=[1.,.5,0.]
    if a.device=='cpu' and not (a.smoke or a.precheck):p.error('CPU mode is for quick validation only')
    if not a.precheck and a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('ROCm/CUDA GPU unavailable')
    C.output().mkdir(parents=True,exist_ok=True);lock=acquire_lock(C.output())
    try:
        train,val,sig=precheck(a.smoke)
        if a.precheck:return
        device=torch.device(a.device)
        import reference_bootstrap as B
        B.prepare(train,val,sig,device,a.smoke)
        for index in range(C.ROUNDS):
            if done(index,sig):continue
            print(f'=== {label(index)} student {C.stage(index)["start"]}->{C.stage(index)["end"]} ===',flush=True)
            train_round(index,train,val,sig,device,a.smoke);report()
        print('Completed:',C.output(),flush=True)
    finally:
        if lock.exists():lock.unlink()

if __name__=='__main__':main()
