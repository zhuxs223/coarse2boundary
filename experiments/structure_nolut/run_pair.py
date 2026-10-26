"""Shared original training/evaluation semantics, independent UNet runner."""
import runtime_setup
import csv,hashlib,importlib.util,json,random,sys,time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import cfg_standard as C
from target_rules import supervision_targets,ignore_ambiguous_negatives


def shared(name, file):
    spec = importlib.util.spec_from_file_location(name, C.MODEL_DIR/file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def atomic_save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def seed(n):
    random.seed(n); np.random.seed(n); torch.manual_seed(n)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(n)


def read(path, rgb=False):
    a = cv2.imread(str(path), cv2.IMREAD_COLOR if rgb else cv2.IMREAD_GRAYSCALE)
    if a is None: raise FileNotFoundError(path)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB) if rgb else a


class Tiles(Dataset):
    """Training/probing NEVER opens boundary_labels; GT access is evaluation-only."""
    def __init__(self, keys, pseudo_dir=None, evaluation=False, uniform_weight=None, target_lut=None):
        self.keys = list(keys); self.pseudo_dir = pseudo_dir; self.evaluation = evaluation
        self.target_lut=target_lut # Serialized with Dataset for Windows workers.
        self.root = C.DATA_ROOT; self.layers = C.LAYERS
        self.mean = np.array(C.IMAGENET_MEAN, np.float32)
        self.std = np.array(C.IMAGENET_STD, np.float32)
        self.mode=C.MODE;self.exponent=C.EXPONENT;self.uniform_weight=uniform_weight
        self.target_mode='refresh';self.geometry_dir=C.GEOMETRY_DIR;self.bonus_dir=C.BONUS_DIR;self.maintained_dir=C.MAINTAINED_DIR
        self.no_repeat_bonus=C.NO_REPEAT_BONUS
        self.frozen_dir=C.FROZEN_DIR
        self.frozen_keys=set(C.FROZEN_KEYS)
        self.ignore_threshold=C.IGNORE_THRESHOLD
        self.ridge_bonus=C.RIDGE_BONUS; self.nonridge_penalty=C.NONRIDGE_PENALTY
        self.rejected_penalty=C.REJECTED_RIDGE_PENALTY

    def __len__(self): return len(self.keys)

    def __getitem__(self, i):
        key = self.keys[i]
        def path(layer): return self.root/self.layers[layer]/'train'/(key+'.png')
        rgb = read(path('img'), True); msk = read(path('msk')) > 0; mk3 = read(path('mk3'))
        if rgb.shape != (C.IMG_SIZE, C.IMG_SIZE, 3) or mk3.shape != msk.shape:
            raise ValueError(f'Invalid image/mask shape: {key}')
        if not np.isin(mk3, [0,1,2]).all(): raise ValueError(f'Invalid mk3 codes: {key}')
        nod = (rgb.max(axis=2) == 0) & ~msk
        valid = A.interior_region(C.IMG_SIZE, C.IMG_SIZE, C.EDGE_TRIM) & ~nod
        img = ((rgb.astype(np.float32)/255-self.mean)/self.std).transpose(2,0,1)
        out = dict(img=torch.from_numpy(np.ascontiguousarray(img)),
                   mk3=torch.from_numpy(mk3), msk=torch.from_numpy(msk),
                   nod=torch.from_numpy(nod), valid=torch.from_numpy(valid), idx=i)
        if self.pseudo_dir is not None:
            layer = np.load(self.pseudo_dir/(key+'.npy'), allow_pickle=False)
            if layer.shape != mk3.shape or not np.isfinite(layer).all() or np.any((layer<0)|(layer>1)):
                raise ValueError(f'Invalid frozen weight layer: {key}')
            if np.any((layer>0)&~((mk3==1)&valid)):
                raise ValueError(f'Pseudo label outside internal valid domain: {key}')
            if self.target_lut is not None:
                layer=self.target_lut[layer.astype(np.float16).view(np.uint16)]
            internal=(mk3==1)&valid
            support=np.unpackbits(np.load(self.geometry_dir/(key+'.npy'),allow_pickle=False),count=mk3.size).reshape(mk3.shape).astype(bool)
            allowed=np.unpackbits(np.load(self.bonus_dir/(key+'.npy'),allow_pickle=False),count=mk3.size).reshape(mk3.shape).astype(bool)
            maintained=np.unpackbits(np.load(self.maintained_dir/(key+'.npy'),allow_pickle=False),count=mk3.size).reshape(mk3.shape).astype(bool)
            if not self.no_repeat_bonus:maintained[:]=False
            if np.any(allowed&~support):raise ValueError('Bonus eligibility outside original ridge support')
            weights,soft_mask,target=supervision_targets(layer,internal,support,allowed=allowed,
                threshold=self.ignore_threshold,bonus=self.ridge_bonus,penalty=self.nonridge_penalty,rejected_penalty=self.rejected_penalty,maintained=maintained)
            teacher_candidate=internal&(layer.astype(np.float32)<=1-self.ignore_threshold)
            frozen_mask=np.zeros(mk3.shape,bool)
            if key in self.frozen_keys:
                from supervision_hold import load_frozen, apply_frozen
                frozen=load_frozen(self.frozen_dir/(key+'.npz'),mk3.shape)
                apply_frozen(weights,soft_mask,target,teacher_candidate,frozen)
                frozen_mask.ravel()[frozen['indices']]=True
                if np.any(frozen_mask&~internal):raise ValueError('Frozen supervision outside internal domain')
            ignored=ignore_ambiguous_negatives(weights,soft_mask,layer,internal)
            out['weight']=torch.from_numpy(weights)
            out['soft_mask']=torch.from_numpy(soft_mask)
            out['soft_target']=torch.from_numpy(target)
            out['negative_ignore_mask']=torch.from_numpy(ignored)
            out['frozen_mask']=torch.from_numpy(frozen_mask)
            out['teacher_candidate_mask']=torch.from_numpy(teacher_candidate)
            out['revoked_mask']=torch.from_numpy(teacher_candidate&~soft_mask)
            out['geometry_support']=torch.from_numpy(support&internal)
            out['bonus_allowed']=torch.from_numpy(allowed&internal)
            out['maintained_ridge']=torch.from_numpy(maintained&internal)
        if self.evaluation: out['bnd'] = torch.from_numpy(read(path('bnd')) > 0)
        return out


def loader(ds, smoke, shuffle=False, inference=False, epoch=0):
    return DataLoader(ds, batch_size=2 if smoke else (C.INFER_BATCH if inference else C.BATCH_SIZE),
                      shuffle=shuffle, num_workers=0 if smoke else C.NUM_WORKERS,
                      pin_memory=torch.cuda.is_available(), drop_last=False,
                      generator=torch.Generator().manual_seed(C.SEED+epoch))


def model(device):
    return M.build_model(str(C.RESNET50_W), verbose=False).to(device)


def load_model(net, path):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    expected = M.model_config(net)
    if expected is not None:
        saved = ck.get('model_config')
        structural=('architecture','encoder','encoder_stride','decoder_interpolation','decoder_blocks','align_corners','reference_model_sha256')
        if saved is None:
            raise RuntimeError('Student checkpoint lacks interpolation metadata')
        if any(saved.get(k)!=expected.get(k) for k in structural):
            raise RuntimeError('Checkpoint model_config differs from runtime decoder')
    net.load_state_dict(ck['model'], strict=True)
    return ck


def model_config(net):
    return M.model_config(net)


def losses(logits, batch):
    valid, mk3 = batch['valid'], batch['mk3']
    source = valid & (mk3!=1)
    internal = valid & (mk3==1)
    denominator = internal.sum().clamp(min=1)
    source_bce = F.binary_cross_entropy_with_logits(logits.float(), (mk3==2).float()*C.OUTLINE_TARGET, reduction='none')
    negative_bce = F.binary_cross_entropy_with_logits(logits.float(), torch.zeros_like(logits,dtype=torch.float32), reduction='none')
    weights = batch['weight'].detach()
    target = batch['soft_target'].detach().float()
    soft_mask = batch['soft_mask'].detach().bool() & internal
    soft_bce = F.binary_cross_entropy_with_logits(logits.float(), target, reduction='none')
    ls = (source_bce*source).sum()/source.sum().clamp(min=1)
    ln = (negative_bce*weights*internal).sum()/denominator
    lsoft = (soft_bce*soft_mask).sum()/denominator
    return ls+C.PSEUDO_WEIGHT*ln+C.SOFT_WEIGHT*lsoft, ls, ln, lsoft, internal, soft_mask


def evaluate(net, ds, device, smoke):
    net.eval(); meters = {(g,d):A.Meter() for g in A.GROUPS for d in A.DOMAINS}
    with torch.no_grad():
        for batch in loader(ds, smoke, inference=True):
            prob = torch.sigmoid(net(batch['img'].to(device)))[:,0].float().cpu().numpy()
            for j,idx in enumerate(batch['idx'].tolist()):
                doms = A.build_domains(C.IMG_SIZE,C.IMG_SIZE,batch['msk'][j].numpy(),
                                      batch['bnd'][j].numpy(),batch['nod'][j].numpy(),
                                      batch['mk3'][j].numpy(),C.EDGE_TRIM)
                region = ds.keys[idx].split('/')[0]
                group = {'Central China':'CC','Southwest China':'SW'}[region]
                A.evaluate_batch(meters,prob[j],batch['bnd'][j].numpy(),doms,C.THR,group)
    return meters


def timeline(net, keys, dest, device, smoke):
    net.eval(); ds=Tiles(keys)
    with torch.no_grad():
        for batch in loader(ds,smoke,inference=True):
            p=torch.sigmoid(net(batch['img'].to(device)))[:,0].float().cpu().numpy()
            for j,idx in enumerate(batch['idx'].tolist()):
                path=dest/(keys[idx]+'.png');path.parent.mkdir(parents=True,exist_ok=True)
                if not cv2.imwrite(str(path),np.rint(p[j]*255).clip(0,255).astype(np.uint8)):
                    raise OSError(path)
                np.save(path.with_suffix('.npy'),p[j],allow_pickle=False)


def upsert_csv(path, rows, fields, identity):
    old=[]
    if path.exists():
        with path.open(newline='',encoding='utf-8') as f:old=list(csv.DictReader(f))
    incoming={tuple(str(r[k]) for k in identity) for r in rows}
    old=[r for r in old if tuple(str(r[k]) for k in identity) not in incoming]
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(old+rows)
    tmp.replace(path)


def commit_logs(root, ck):
    upsert_csv(root/'metrics.csv',ck['metric_rows'],['round','epoch_in_round']+A.CSV_COLUMNS,
               ['epoch','split','reg','domain'])
    row=ck['diagnostic']
    upsert_csv(root/'diagnostics.csv',[row],list(row),['global_epoch'])


def train_epoch(net,opt,scaler,ds,device,smoke,global_epoch):
    from repeat_confidence import STAT_FIELDS
    seed(C.SEED+C.START_EPOCH+global_epoch);net.train();start=time.time()
    sums=dict(loss=0.,source_loss=0.,negative_loss=0.,soft_loss=0.,
              inner_px=0,inner_ge_half_px=0,soft_px=0,negative_px=0,
              soft_ge_half_px=0,soft_target_sum=0.,soft_student_sum=0.,
              soft_abs_error_sum=0.,negative_weight_sum=0.,steps=0)
    batches=loader(ds,smoke,shuffle=True,epoch=C.START_EPOCH+global_epoch)
    repeat_stats=np.zeros(len(STAT_FIELDS),np.float64)
    repeat_prediction=torch.zeros(2,device=device,dtype=torch.float64)
    for step,batch in enumerate(batches,1):
        # Reduce tiny target statistics on CPU before moving the image batch.
        stats=batch.pop('repeat_confidence_stats',None)
        if stats is not None:repeat_stats+=stats.numpy().sum(axis=0)
        batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda',enabled=C.AMP and device.type=='cuda'):
            logits=net(batch['img'])[:,0]
            loss,ls,ln,lsoft,inner,soft=losses(logits,batch)
        if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
        scaler.scale(loss).backward();scaler.step(opt);scaler.update()
        for name,value in [('loss',loss),('source_loss',ls),('negative_loss',ln),('soft_loss',lsoft)]:
            sums[name]+=float(value.detach())
        weights=batch['weight'].detach();q=batch['soft_target'].detach()
        p=torch.sigmoid(logits.detach().float())
        if 'repeat_confidence_mask' in batch:
            selected=batch['repeat_confidence_mask'].bool()
            repeat_prediction+=torch.stack([(p*selected).sum(),(selected&(p>=.9)).sum()])
        outline=batch['valid']&(batch['mk3']==2)
        sums['outline_px']=sums.get('outline_px',0)+int(outline.sum())
        sums['outline_student_sum']=sums.get('outline_student_sum',0.)+float(p[outline].sum())
        sums['outline_ge_half_px']=sums.get('outline_ge_half_px',0)+int((outline&(p>=.5)).sum())
        sums['inner_px']+=int(inner.sum());sums['inner_ge_half_px']+=int((inner&(p>=.5)).sum())
        sums['soft_px']+=int(soft.sum())
        sums['frozen_px']=sums.get('frozen_px',0)+int(batch['frozen_mask'].sum())
        sums['frozen_soft_px']=sums.get('frozen_soft_px',0)+int((batch['frozen_mask']&soft).sum())
        sums['teacher_candidate_px']=sums.get('teacher_candidate_px',0)+int(batch['teacher_candidate_mask'].sum())
        sums['revoked_teacher_candidate_px']=sums.get('revoked_teacher_candidate_px',0)+int(batch['revoked_mask'].sum())
        ridge=soft&batch['geometry_support'].bool();nonridge=soft&~batch['geometry_support'].bool()
        maintained=ridge&batch['maintained_ridge'].bool()
        boosted=ridge&batch['bonus_allowed'].bool()&~maintained;held=ridge&~batch['bonus_allowed'].bool()
        sums['maintained_ridge_soft_px']=sums.get('maintained_ridge_soft_px',0)+int(maintained.sum())
        sums['maintained_ridge_target_sum']=sums.get('maintained_ridge_target_sum',0.)+float(q[maintained].sum())
        sums['maintained_ridge_student_sum']=sums.get('maintained_ridge_student_sum',0.)+float(p[maintained].sum())
        sums['boosted_ridge_soft_px']=sums.get('boosted_ridge_soft_px',0)+int(boosted.sum())
        sums['rejected_ridge_soft_px']=sums.get('rejected_ridge_soft_px',0)+int(held.sum())
        rejected_candidate=batch['teacher_candidate_mask'].bool()&batch['geometry_support'].bool()&~batch['bonus_allowed'].bool()
        sums['rejected_ridge_candidate_px']=sums.get('rejected_ridge_candidate_px',0)+int(rejected_candidate.sum())
        sums['revoked_rejected_ridge_px']=sums.get('revoked_rejected_ridge_px',0)+int((rejected_candidate&~soft).sum())
        sums['rejected_ridge_target_sum']=sums.get('rejected_ridge_target_sum',0.)+float(q[held].sum())
        for label,mask in [('ridge',ridge),('nonridge',nonridge)]:
            sums[label+'_soft_px']=sums.get(label+'_soft_px',0)+int(mask.sum())
            sums[label+'_target_sum']=sums.get(label+'_target_sum',0.)+float(q[mask].sum())
            sums[label+'_student_sum']=sums.get(label+'_student_sum',0.)+float(p[mask].sum())
        sums['soft_target_below_half_px']=sums.get('soft_target_below_half_px',0)+int((soft&(q<.5)).sum());sums['negative_px']+=int((inner&~soft).sum())
        ignored=batch['negative_ignore_mask'].bool()
        if torch.any(ignored&(~inner|soft|(weights!=0))):raise RuntimeError('Ignored negatives still contribute to loss')
        sums['negative_ignore_px']=sums.get('negative_ignore_px',0)+int(ignored.sum())
        sums['soft_ge_half_px']+=int((soft&(p>=.5)).sum())
        sums['soft_target_sum']+=float((q*soft).sum())
        sums['soft_student_sum']+=float((p*soft).sum())
        sums['soft_abs_error_sum']+=float(((p-q).abs()*soft).sum())
        sums['negative_weight_sum']+=float((weights*inner).sum());sums['steps']+=1
        if step%C.PROGRESS_EVERY==0 or step==len(batches):
            print(f'  train {step}/{len(batches)} loss={sums["loss"]/step:.4f} lr={opt.param_groups[0]["lr"]:g}',flush=True)
    n=sums.pop('steps')
    for k in ['loss','source_loss','negative_loss','soft_loss']:sums[k]/=max(n,1)
    sums['inner_ge_half_fraction']=sums['inner_ge_half_px']/max(sums['inner_px'],1)
    sums['soft_fraction']=sums['soft_px']/max(sums['inner_px'],1)
    sums['negative_ignore_fraction']=sums['negative_ignore_px']/max(sums['inner_px'],1)
    sums['teacher_candidate_fraction']=sums['teacher_candidate_px']/max(sums['inner_px'],1)
    sums['revoked_fraction_of_teacher_candidates']=sums['revoked_teacher_candidate_px']/max(sums['teacher_candidate_px'],1)
    sums['soft_ge_half_fraction']=sums['soft_ge_half_px']/max(sums['soft_px'],1)
    sums['mean_inner_negative_weight']=sums.pop('negative_weight_sum')/max(sums['inner_px'],1)
    for k in ['target','student','abs_error']:
        sums[f'soft_{k}_mean']=sums.pop(f'soft_{k}_sum')/max(sums['soft_px'],1)
    sums['negative_loss_contribution']=C.PSEUDO_WEIGHT*sums['negative_loss']
    sums['soft_loss_contribution']=C.SOFT_WEIGHT*sums['soft_loss']
    sums['outline_target']=C.OUTLINE_TARGET
    sums['outline_student_mean']=sums.pop('outline_student_sum')/max(sums['outline_px'],1)
    sums['outline_ge_half_fraction']=sums['outline_ge_half_px']/max(sums['outline_px'],1)
    for label in ['ridge','nonridge']:
        for field in ['target','student']:
            sums[label+'_'+field+'_mean']=sums.pop(label+'_'+field+'_sum')/max(sums[label+'_soft_px'],1)
    sums['rejected_ridge_target_mean']=sums.pop('rejected_ridge_target_sum')/max(sums['rejected_ridge_soft_px'],1)
    sums['maintained_ridge_target_mean']=sums.pop('maintained_ridge_target_sum')/max(sums['maintained_ridge_soft_px'],1)
    sums['maintained_ridge_student_mean']=sums.pop('maintained_ridge_student_sum')/max(sums['maintained_ridge_soft_px'],1)
    sums['maintained_fraction_of_qualified_soft']=sums['maintained_ridge_soft_px']/max(sums['maintained_ridge_soft_px']+sums['boosted_ridge_soft_px'],1)
    sums['continued_epoch']=C.START_EPOCH+global_epoch
    d=dict(zip(STAT_FIELDS,repeat_stats));den=max(d['eligible_px'],1)
    student_sum,student_ge09=repeat_prediction.tolist()
    for name in ('eligible_px','raised_px','extra_px','already_high_px'):
        sums['repeat_confidence_'+name]=int(d[name])
    for name,source in [('original_target_mean','original_target_sum'),('raw_teacher_mean','raw_teacher_sum'),
                        ('target_mean','final_target_sum'),('increase_mean','increase_sum')]:
        sums['repeat_confidence_'+name]=d[source]/den
    for name,source in [('target_ge_09_fraction','target_ge_09_px'),('target_ge_099_fraction','target_ge_099_px'),
                        ('target_one_fraction','target_one_px')]:
        sums['repeat_confidence_'+name]=d[source]/den
    sums['repeat_confidence_student_mean']=student_sum/den
    sums['repeat_confidence_student_ge_09_fraction']=student_ge09/den
    print(f'[repeat ep{C.START_EPOCH+global_epoch}] pixels={int(d["eligible_px"])} '
          f'target_mean={d["final_target_sum"]/den:.4f} student_mean={student_sum/den:.4f} '
          f'target_ge0.9={d["target_ge_09_px"]/den:.3f} '
          f'target_ge0.99={d["target_ge_099_px"]/den:.3f} '
          f'already_high={int(d["already_high_px"])}',flush=True)
    sums['seconds']=time.time()-start;sums['steps']=n
    return sums


def quantile_lut(reference,current):
    """Map cached 1-q values by midpoint rank; ascending base is descending q."""
    if reference.sum()<=0 or current.sum()<=0:raise ValueError('Empty histogram')
    support=np.flatnonzero(reference)
    ref_cdf=np.cumsum(reference[support],dtype=np.float64)/reference.sum()
    midpoint=(np.cumsum(current,dtype=np.float64)-current/2)/current.sum()
    ix=np.searchsorted(ref_cdf,midpoint,side='left').clip(0,len(support)-1)
    return support[ix].astype(np.uint16).view(np.float16)


A=shared("unet_ridge_assess", "assess.py")
import model_student as M
