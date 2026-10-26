"""Independent nnPU-Oracle adaptation; atomic epoch commits and fixed ep200."""
import runtime_setup
import argparse
import gc
import json
import math
import re
import shutil
import time
import cv2
import numpy as np
import torch
import cfg_standard as C
import data as D
import model_student as M
import supervision as S
import utils as U
from dataset_protocol import load_targets, smoke_targets
from prepare_priors import load_priors, prepare, DOMAIN
from sampling import PositiveBatchSampler
from reference import assess as A


def configuration():
    names=('EPOCHS','PU_WEIGHT','LR_DROP_EPOCH','LR_AFTER_DROP','BATCH_SIZE','INFER_BATCH',
        'NUM_WORKERS','AMP','LR','WEIGHT_DECAY','SEED','IMG_SIZE','EDGE_TRIM','EVALUATE_EVERY',
        'SAVE_EVERY','DECODER_INTERPOLATION','LAYERS','IMAGENET_MEAN','IMAGENET_STD','AUGMENT','PROBE_KEYS')
    return dict(protocol='nnpu-oracle-adapted-full5227-v1',settings={n:getattr(C,n) for n in names},
        optimizer='Adam; all trainable parameters',scheduler='1e-4 ep1-50; 1e-5 ep51-200',
        objective='source BCE(P,N) + 1 * nnPU(P,U); beta=0 gamma=1',
        prior_domain=DOMAIN,extra_information='Oracle true per-image prevalence in U',
        empty_positive='borrow batch pooled P; deterministic swaps ensure a P reference; no repeated samples',
        selection='fixed final epoch; GT metrics for retrospective monitoring only',
        no_small=True,no_geometry=True,no_ema=True,no_temporal_prior_estimation=True)


def plan(keys,smoke):
    _,_,_,cohort=load_targets(C.SPLIT_JSON)
    for key in keys:
        for layer in C.LAYERS.values():
            path=C.DATA_ROOT/layer/'train'/(key+'.png')
            if not path.is_file():
                raise FileNotFoundError(path)
    if U.sha(C.RESNET50_W)!='8c28bce1adf71c9dbb928e91724832100e67dc6ff68af1a3a79f53cc39833bbf':
        raise ValueError('ImageNet ResNet50 SHA256 differs')
    code={p.relative_to(C.HERE).as_posix():U.sha(p) for p in sorted(C.HERE.rglob('*.py'))
          if not p.name.startswith('test_') and '__pycache__' not in p.parts}
    signature=json.loads(json.dumps(dict(config=configuration(),keys=keys,cohort=cohort,smoke=smoke,
        split_sha256=U.sha(C.SPLIT_JSON),resnet_sha256=U.sha(C.RESNET50_W),
        priors_sha256=U.sha(C.priors_path()) if C.priors_path().exists() else None,
        data_root=str(C.DATA_ROOT.resolve()),code=code)))
    path=C.output()/'plan.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))['signature']!=signature:
        raise RuntimeError('Scientific code/config/inputs changed; use an independent output directory')
    # A precheck before Oracle preparation must not freeze a missing prior file.
    if signature['priors_sha256'] is not None:
        U.json_write(path,dict(signature=signature))
    return signature


def precheck(keys,priors):
    U.seed(C.SEED)
    net=M.build_model(str(C.RESNET50_W),verbose=False)
    config=M.model_config(net)
    if config['decoder_interpolation']!='bilinear' or any(isinstance(m,torch.nn.ConvTranspose2d) for m in net.modules()):
        raise RuntimeError('Unexpected student architecture')
    item=D.Tiles(keys[:1],priors=priors)[0]
    if 'bnd' in item:
        raise RuntimeError('Training path unexpectedly read fine GT')
    if priors is not None:
        # Build batches once before training so insufficient positive references fail early.
        batches=list(PositiveBatchSampler(keys,priors,C.BATCH_SIZE,C.SEED+1))
        if sorted(i for b in batches for i in b)!=list(range(len(keys))):
            raise RuntimeError('PU sampler coverage differs')
    info=dict(train=len(keys),evaluation=len(keys),overlap=len(keys),epochs=C.EPOCHS,
        batch=C.BATCH_SIZE,optimizer_steps_per_epoch=math.ceil(len(keys)/C.BATCH_SIZE),
        optimizer_steps_total=math.ceil(len(keys)/C.BATCH_SIZE)*C.EPOCHS,
        method='nnPU-Oracle adaptation',fine_gt_use='Oracle scalar preparation + evaluation only',
        prior_file=str(C.priors_path()),priors_ready=priors is not None,
        priors_sha256=U.sha(C.priors_path()) if priors is not None else None,
        pu_weight=C.PU_WEIGHT,source_background_weight=1,negative_correction=dict(beta=0,gamma=1),
        warm_epochs=0,optimizer='Adam',lr=C.LR,lr_drop_after_epoch=C.LR_DROP_EPOCH,
        lr_after_drop=C.LR_AFTER_DROP,weight_decay=C.WEIGHT_DECAY,
        small_teacher=False,geometry=False,ema=False,train_workers=C.NUM_WORKERS,
        student_parameters=sum(p.numel() for p in net.parameters()),model_config=config,
        cuda=torch.cuda.is_available(),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        output=str(C.output()),runtime=runtime_setup.SETTINGS)
    U.json_write(C.output()/'precheck.json',info)
    print('[precheck]',json.dumps(info),flush=True)
    del net,item
    gc.collect()


def train_epoch(net,opt,scaler,keys,priors,epoch,device):
    U.seed(C.SEED+epoch)
    lr=C.LR if epoch<=C.LR_DROP_EPOCH else C.LR_AFTER_DROP
    for group in opt.param_groups:
        group['lr']=lr
    ds=D.Tiles(keys,epoch=epoch,training=True,priors=priors)
    sampler=PositiveBatchSampler(keys,priors,C.BATCH_SIZE,C.SEED+epoch)
    net.train()
    start=time.perf_counter()
    fields=('loss','source','pu','rpos','rneg','corrected','fallback_tiles','U_pixels','P_pixels','eligible_tiles')
    totals=torch.zeros(len(fields),device=device,dtype=torch.float64)
    steps=updates=0
    for batch in D.loader(ds,batch_sampler=sampler):
        # No fine maps, only RGB/coarse masks/valid and scalar pi enter this path.
        if 'bnd' in batch:
            raise RuntimeError('Fine GT leaked into training batch')
        batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda',enabled=C.AMP and device.type=='cuda'):
            loss,stats=S.losses(net(batch['img'])[:,0],batch)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        oldscale=scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        updates+=int(scaler.get_scale()>=oldscale)
        values=dict(loss=loss,**stats)
        totals+=torch.stack([values[f].detach().double() for f in fields])
        steps+=1
        if steps%100==0:
            print(f'[train ep{epoch:03d}] {steps}/{len(sampler)} {time.perf_counter()-start:.1f}s',flush=True)
    row=dict(epoch=epoch,lr=lr,steps=steps,optimizer_updates=updates,amp_skipped_steps=steps-updates)
    for name,value in zip(fields,totals.cpu().tolist()):
        row[name]=int(value) if name in fields[6:] else value/steps
    row['corrected_batches']=round(row.pop('corrected')*steps)
    row['correction_fraction']=row['corrected_batches']/steps
    row['train_seconds']=time.perf_counter()-start
    return row


def probes(net,keys,epoch,device):
    selected=[k for k in C.PROBE_KEYS if k in set(keys)]
    net.eval()
    with torch.inference_mode():
        for batch in D.loader(D.Tiles(selected),inference=True,workers=0):
            prob=torch.sigmoid(net(batch['img'].to(device)).float())[:,0].cpu().numpy()
            for j,idx in enumerate(batch['idx'].tolist()):
                path=C.output()/'timeline'/f'epoch_{epoch:03d}'/selected[idx]
                path.parent.mkdir(parents=True,exist_ok=True)
                np.save(str(path)+'_raw.npy',prob[j],allow_pickle=False)
                if not cv2.imwrite(str(path)+'_raw.png',np.rint(prob[j]*255).clip(0,255).astype(np.uint8)):
                    raise OSError(path)


def evaluate(net,keys,epoch,device):
    start=time.perf_counter()
    meters={(g,d):A.Meter() for g in A.GROUPS for d in A.DOMAINS}
    counts=dict(epoch=epoch,U_pixels=0,U_gt=0,U_predicted=0,U_TP=0)
    net.eval()
    with torch.inference_mode():
        for batch in D.loader(D.Tiles(keys,evaluation=True),inference=True):
            prob=torch.sigmoid(net(batch['img'].to(device,non_blocking=True)).float())[:,0].cpu().numpy()
            for j,idx in enumerate(batch['idx'].tolist()):
                gt,mk3,valid=batch['bnd'][j].numpy(),batch['mk3'][j].numpy(),batch['valid'][j].numpy()
                doms=A.build_domains(C.IMG_SIZE,C.IMG_SIZE,batch['msk'][j].numpy(),gt,
                    batch['nod'][j].numpy(),mk3,C.EDGE_TRIM)
                region={'Central China':'CC','Southwest China':'SW'}[keys[idx].split('/')[0]]
                A.evaluate_batch(meters,prob[j],gt,doms,.5,region)
                u=valid & (mk3==1)
                pred=prob[j]>=.5
                for name,mask in (('U_pixels',u),('U_gt',u&gt),('U_predicted',u&pred),('U_TP',u&pred&gt)):
                    counts[name]+=int(mask.sum())
    counts.update(U_precision=A.safe_div(counts['U_TP'],counts['U_predicted']),
        U_recall=A.safe_div(counts['U_TP'],counts['U_gt']),
        U_base_rate=A.safe_div(counts['U_gt'],counts['U_pixels']),
        U_predicted_fraction=A.safe_div(counts['U_predicted'],counts['U_pixels']),
        eval_seconds=time.perf_counter()-start)
    rows=A.rows_for_csv(epoch,'target_full',meters)
    main=next(r for r in rows if r['reg']=='ALL' and r['domain']=='agri')
    print(f'[pu ep{epoch:03d} target_full] P={main["P_seg"]:.4f} R={main["R_seg"]:.4f} IoU={main["IoU_seg"]:.4f}',flush=True)
    return rows,counts


def write_logs(ck):
    U.csv_write(C.output()/'diagnostics.csv',ck['diagnostics'])
    U.csv_write(C.output()/'metrics.csv',ck['metrics'],A.CSV_COLUMNS)
    U.csv_write(C.output()/'internal_quality.csv',ck['quality'])


def snapshot(ck):
    path=C.output()/'checkpoints'/f'ep{ck["epoch"]:03d}.pth'
    if not path.exists():
        U.torch_write(path,dict(model=ck['model'],model_config=ck['model_config'],epoch=ck['epoch'],
            optimizer_steps=ck['optimizer_steps'],signature=ck['signature'],
            purpose='inference only; resume from latest.pth'))


def publish(ck):
    write_logs(ck)
    if ck['epoch']%C.SAVE_EVERY==0 or ck['epoch']==C.EPOCHS:
        snapshot(ck)
    U.json_write(C.output()/'progress.json',dict(epoch=ck['epoch'],epochs=C.EPOCHS,
        optimizer_steps=ck['optimizer_steps'],complete=ck['epoch']==C.EPOCHS))


def fit(keys,priors,signature,args):
    if not torch.cuda.is_available() and not args.smoke:
        raise RuntimeError('Full training requires CUDA')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    U.seed(C.SEED)
    net=M.build_model(str(C.RESNET50_W),verbose=False).to(device)
    opt=torch.optim.Adam(net.parameters(),lr=C.LR,weight_decay=C.WEIGHT_DECAY)
    scaler=torch.amp.GradScaler('cuda',enabled=C.AMP and device.type=='cuda')
    completed=steps=0
    diagnostics,metrics,quality=[],[],[]
    latest=C.output()/'latest.pth'
    if latest.exists():
        ck=torch.load(latest,map_location='cpu',weights_only=True)
        if ck['signature']!=signature:
            raise RuntimeError('Resume signature differs')
        structural=('architecture','encoder','encoder_stride','decoder_interpolation','decoder_blocks','align_corners','reference_model_sha256')
        if any(ck['model_config'][k]!=M.model_config(net)[k] for k in structural):
            raise RuntimeError('Resume model architecture differs')
        net.load_state_dict(ck['model'],strict=True)
        opt.load_state_dict(ck['optimizer'])
        scaler.load_state_dict(ck['scaler'])
        completed,steps=ck['epoch'],ck['optimizer_steps']
        diagnostics,metrics,quality=ck['diagnostics'],ck['metrics'],ck['quality']
        publish(ck)
        del ck
        print(f'[resume] completed epoch={completed}, optimizer_steps={steps}; Oracle hash verified',flush=True)
    for epoch in range(completed+1,C.EPOCHS+1):
        if shutil.disk_usage(C.output()).free<C.MIN_FREE_GIB*1024**3+600*1024**2:
            raise RuntimeError('Insufficient disk space for checkpoint + reserve')
        print(f'=== pu epoch {epoch:03d}/{C.EPOCHS} ===',flush=True)
        row=train_epoch(net,opt,scaler,keys,priors,epoch,device)
        steps+=row['optimizer_updates']
        probes(net,keys,epoch,device)
        row['evaluation_seconds']=0.
        if epoch%C.EVALUATE_EVERY==0 or epoch==C.EPOCHS:
            new_metrics,q=evaluate(net,keys,epoch,device)
            metrics+=new_metrics
            quality.append(q)
            row['evaluation_seconds']=q['eval_seconds']
        diagnostics.append(row)
        ck=dict(model=net.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),epoch=epoch,
            model_config=M.model_config(net),optimizer_steps=steps,signature=signature,
            diagnostics=diagnostics,metrics=metrics,quality=quality)
        if args.smoke_fail_before_commit==epoch:
            raise RuntimeError('Intentional smoke interruption before epoch commit')
        U.torch_write(latest,ck)
        publish(ck)
        print(f'[epoch {epoch:03d}] loss={row["loss"]:.5f} source={row["source"]:.5f} PU={row["pu"]:.5f} '
            f'Rpos={row["rpos"]:.5f} Rneg={row["rneg"]:.5f}; correction={row["correction_fraction"]:.1%}; '
            f'fallback_tiles={row["fallback_tiles"]}; train={row["train_seconds"]:.1f}s',flush=True)
        del ck
        if args.stop_after_epoch==epoch:
            print('[stop] epoch boundary; repeat the same command to resume',flush=True)
            break
    print('[complete]' if completed==C.EPOCHS or epoch==C.EPOCHS else '[paused at epoch boundary]',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precheck',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--smoke-id',default='v1')
    parser.add_argument('--smoke-workers',type=int,default=0)
    parser.add_argument('--smoke-epochs',type=int,default=2)
    parser.add_argument('--smoke-fail-before-commit',type=int,default=0)
    parser.add_argument('--stop-after-epoch',type=int,default=0)
    args=parser.parse_args()
    runtime_setup.configure()
    keys,_,groups,_=load_targets(C.SPLIT_JSON)
    if args.smoke:
        if not re.fullmatch(r'[a-zA-Z0-9_-]+',args.smoke_id) or args.smoke_epochs<1:
            raise ValueError('Invalid smoke id/epochs')
        keys,_=smoke_targets(groups)
        C.RUN_NAME='_smoke_unet_pu_'+args.smoke_id
        C.EPOCHS=args.smoke_epochs
        C.BATCH_SIZE=C.INFER_BATCH=2
        C.EVALUATE_EVERY=C.SAVE_EVERY=1
        C.NUM_WORKERS=args.smoke_workers
        C.MIN_FREE_GIB=1
        C.PROBE_KEYS=keys
        torch.use_deterministic_algorithms(True)
    elif args.smoke_fail_before_commit or args.smoke_workers:
        parser.error('Fault injection/worker override requires --smoke')
    C.NUM_WORKERS=min(C.NUM_WORKERS,max(0,runtime_setup.cpu_capacity(C.CPU_LIMIT)-2))
    with U.run_lock(C.output()):
        if not C.priors_path().exists() and not args.precheck:
            if (C.output()/'latest.pth').exists():
                raise RuntimeError('Committed Oracle priors missing; restore original prior file')
            print('[Oracle] preparing true per-tile U prevalence before training; extra supervision declared',flush=True)
            prepare(keys,C.priors_path())
        priors=load_priors(keys,C.priors_path()) if C.priors_path().exists() else None
        signature=plan(keys,args.smoke)
        precheck(keys,priors)
        if not args.precheck:
            fit(keys,priors,signature,args)


if __name__=='__main__':
    main()
