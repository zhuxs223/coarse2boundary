"""Real-image three-arm E2E with an explicit constant teacher fixture.

The fixture is not the workstation r020 model and produces no scientific results.
Checks pairing, label isolation, pseudo conversion, metric selection, test and resume.
"""
import argparse
import os
import gc
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import torch

import run_all as R


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--data-root')
    parser.add_argument('--reuse',action='store_true',help='Reuse completed fixture GPU runs while extending validation')
    args=parser.parse_args()
    root=Path(args.output).resolve()
    if args.reuse:
        cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
        split=json.loads((root/'splits.json').read_text(encoding='utf-8'))
    else:
        if root.exists():raise FileExistsError('Use a fresh smoke output: '+str(root))
        root.mkdir(parents=True)
        cfg=json.loads((R.HERE/'config.json').read_text(encoding='utf-8'))
        full=json.loads((R.HERE.parent/'meta/splits_v4_sd.json').read_text(encoding='utf-8'))
        split={name:[next(k for k in full[name] if k.startswith(region))
                     for region in ('Central China/','Southwest China/')]
               for name in ('train_sd','val_sd','test_sd')}
        split['dropped_buffer']=[]
        R.atomic_json(root/'splits.json',split)
        cfg.update(data_root=args.data_root or cfg['data_root'],split_file=str(root/'splits.json'),
                   resnet_weights=os.environ.get('EXP810_RESNET', str((R.HERE/cfg['resnet_weights']).resolve())),
                   pseudo_checkpoint=str(root/'teacher_fixture.pth'),output_root=str(root/'run'),
                   epochs=2,batch_size=2,infer_batch_size=2,workers=0,preview_every=2)
        # Zero boundary-logit head => p=sigmoid(-2)<0.5; pseudo parcel must equal coarse mask.
        R.seed_all(cfg['seed'])
        net=R.model(cfg['resnet_weights'])
        with torch.no_grad():
            net.segmentation_head[0].weight.zero_();net.segmentation_head[0].bias.fill_(-2)
        R.atomic_torch(cfg['pseudo_checkpoint'],dict(model={'net.'+k:v for k,v in net.state_dict().items()},
                       model_config=R.model_config(),epoch=200,round=20,fixture=True))
        del net
        gc.collect()
        R.atomic_json(root/'config.json',cfg)
    # Metric sanity: boundary-only parcel IoU is undefined; the neighborhood is not.
    coarse=np.zeros((25,25),bool);coarse[2:23,2:23]=True
    bnd=np.zeros_like(coarse);bnd[2:23,12]=True
    parcel=coarse&~bnd
    band=(R.ndi.distance_transform_edt(~bnd)<=3)&coarse
    assert R.cell(parcel,parcel,band)[1:3].sum()==0
    assert R.cell(coarse,parcel,band)[1]>0
    assert R.cell(np.zeros_like(parcel),parcel,band)[2]>0
    assert R.cell(coarse&~coarse,bnd,coarse)[2]>0
    # Coarse/image-only loaders must not touch either fine-label layer.
    actual_layer=R.layer
    def no_fine(root_,name,key):
        assert name not in ('parcel_labels','boundary_labels'),name
        return actual_layer(root_,name,key)
    with patch.object(R,'layer',side_effect=no_fine):
        assert 'boundary' not in R.Tiles(cfg,split['train_sd'],arm='coarse_mask')[0]
        assert 'target' not in R.Tiles(cfg,split['train_sd'])[0]
    cmd=[sys.executable,'-X','utf8',str(R.HERE/'run_all.py'),'--config',str(root/'config.json')]
    subprocess.run(cmd,check=True)
    result_root=Path(cfg['output_root'])
    with patch.object(R,'layer',side_effect=no_fine):
        pseudo=R.Tiles(cfg,split['train_sd'],arm='pseudo_parcel')[0]
        coarse_sample=R.Tiles(cfg,split['train_sd'],arm='coarse_mask')[0]
        assert torch.equal(pseudo['target'],coarse_sample['target'])
    bests={arm:torch.load(result_root/arm/'best.pth',map_location='cpu',weights_only=True) for arm in R.ARMS}
    assert len({v['initial_state_sha256'] for v in bests.values()})==1
    for arm,best in bests.items():
        last=torch.load(result_root/arm/'last.pth',map_location='cpu',weights_only=True)
        assert last['epoch']==2 and all(h['steps']==1 for h in last['history'])
        assert best['selected_val_IoU']==max(h['val_selection_IoU'] for h in last['history'])
        assert all(row['split']=='val' for row in best['selected_val_metrics'])
        test=json.loads((result_root/arm/'test_metrics.json').read_text())
        assert test['best_epoch']==best['epoch'] and all(row['split']=='test' for row in test['metrics'])
        del last
    # CUDA bilinear backward/AMP may differ numerically even with paired data and seeds.
    squared_diff=squared_reference=max_abs=0.0
    for key,value in bests['pseudo_parcel']['model'].items():
        other=bests['coarse_mask']['model'][key]
        if value.is_floating_point():
            diff=(value-other).float()
            max_abs=max(max_abs,float(diff.abs().max()))
            squared_diff+=float(diff.square().double().sum())
            squared_reference+=float(other.float().square().double().sum())
        else:
            assert torch.equal(value,other)
    relative_l2=(squared_diff/max(squared_reference,1e-30))**.5
    score_difference=abs(bests['pseudo_parcel']['selected_val_IoU']-bests['coarse_mask']['selected_val_IoU'])
    assert relative_l2<1e-3 and score_difference<1e-3
    gpu_variation=dict(relative_l2=relative_l2,max_abs_parameter_difference=max_abs,val_score_difference=score_difference)
    del bests
    gc.collect()
    before={arm:R.sha(result_root/arm/'best.pth') for arm in R.ARMS}
    subprocess.run(cmd,check=True)
    after={arm:R.sha(result_root/arm/'best.pth') for arm in R.ARMS}
    assert before==after,'Completed resume changed best checkpoint identity'
    # Interrupt after the authoritative epoch commit but before best/log writes.
    # Resume with Windows spawn workers; final state must equal uninterrupted training.
    continuous=dict(cfg,output_root=str(root/'cpu_continuous'),workers=0,amp=False)
    resumed=dict(cfg,output_root=str(root/'partial_resume'),workers=2,amp=False)
    Path(continuous['output_root']).mkdir()
    Path(resumed['output_root']).mkdir()
    device=torch.device('cpu')
    torch.use_deterministic_algorithms(True)
    R.cv2.setNumThreads(1);torch.set_num_threads(8)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    continuous_info=R.preflight(continuous,split,device)
    R.train_arm('fine_parcel',continuous,split,continuous_info,device)
    info=R.preflight(resumed,split,device)
    try:
        R.train_arm('fine_parcel',resumed,split,info,device,_interrupt_after_commit=1)
        raise AssertionError('Failure injection did not stop training')
    except R.ValidationInterruption:
        pass
    assert not (Path(resumed['output_root'])/'fine_parcel/best.pth').exists()
    gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    R.train_arm('fine_parcel',resumed,split,info,device)
    a=torch.load(Path(continuous['output_root'])/'fine_parcel/last.pth',map_location='cpu',weights_only=True)
    b=torch.load(Path(resumed['output_root'])/'fine_parcel/last.pth',map_location='cpu',weights_only=True)
    def equal_state(x,y):
        if isinstance(x,torch.Tensor):return torch.equal(x,y)
        if isinstance(x,dict):return x.keys()==y.keys() and all(equal_state(x[k],y[k]) for k in x)
        if isinstance(x,(list,tuple)):return len(x)==len(y) and all(equal_state(xx,yy) for xx,yy in zip(x,y))
        return x==y
    for field in ('model','optimizer','scaler','best'):
        if field=='best':
            assert equal_state(a[field]['model'],b[field]['model'])
            assert a[field]['epoch']==b[field]['epoch']
        else:
            assert equal_state(a[field],b[field]),field
    assert [h['metrics'] for h in a['history']]==[h['metrics'] for h in b['history']]
    R.atomic_json(root/'validation.json',dict(passed=True,images_per_split=2,epochs_per_arm=2,
        fixture='constant p<0.5 teacher, not the actual workstation model',gpu_numerical_variation=gpu_variation,
        checks=['spatial split','three independent paired initializations','no fine labels in pseudo/coarse training',
                'pseudo mask conversion','nondegenerate selection domain','val-only best selection',
                'test using best weights after all arms','paired pseudo/coarse labels and initialization exact; GPU parameter variation below relative-L2 1e-3','completed-run reuse',
                'interruption between checkpoint and logs','CPU deterministic two-worker spawn resume exactly matches uninterrupted model/optimizer/scaler/metrics']))
    print('SMOKE VALIDATION PASSED',flush=True)


if __name__=='__main__':main()
