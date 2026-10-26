"""User-launched H/8 encoder/decoder extension, paired with original SMALL."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import tempfile
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

import model_h8 as S
import model_reference as M
import run_pair as P


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def state_sha(state):
    h=hashlib.sha256()
    for k,v in state.items():
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def build_model(reference):
    P.seed_all()
    model=S.SmallH8(P.WEIGHTS)
    ref=json.loads((reference/'manifest.json').read_text(encoding='utf-8'))
    original_sha=state_sha(S.original_state_dict(model))
    assert original_sha==ref['initial_state_sha256'], 'Original SMALL initialization differs'
    assert ref['h2_skip'] and ref['n_train']==5227 and ref['epochs']==10
    assert ref['params']==557505
    assert torch.count_nonzero(model.head.conv_in[0].weight[:,256:])==0
    return model,ref,original_sha


def make_optimizer(model):
    return torch.optim.AdamW([
        dict(params=[p for p in model.parameters() if p.ndim!=1],weight_decay=.01),
        dict(params=[p for p in model.parameters() if p.ndim==1],weight_decay=0)],lr=1e-4)


def precheck(args,keys,groups):
    model,ref,original_sha=build_model(args.reference)
    weights=torch.load(P.WEIGHTS,map_location='cpu',weights_only=True)
    assert all(torch.equal(v,weights['layer2.'+k]) for k,v in model.layer2.state_dict().items())
    # Check the hand-built stage against torchvision's actual ResNet50 stage.
    from torchvision.models import resnet50
    official=resnet50(weights=None).layer2
    official.load_state_dict(model.layer2.state_dict(),strict=True)
    official.eval()
    model.layer2.eval()
    with torch.inference_mode():
        feature=torch.randn(1,256,16,16)
        torch.testing.assert_close(official(feature),model.layer2(feature),rtol=0,atol=0)
    del official,feature,weights
    P.seed_all()
    baseline=P.Small(True)
    assert P.state_sha(baseline)==original_sha
    device=torch.device('cuda')
    torch.cuda.reset_peak_memory_stats()
    model.to(device).eval()
    baseline.to(device).eval()
    batch=next(iter(P.loader(P.Samples(keys[:12]),0,True)))
    assert 'bnd' not in batch and batch['img'].shape==(12,3,512,512)
    with torch.inference_mode():
        x=batch['img'][:2].to(device)
        extended=model(x)
        original=baseline(x)
        initial_mae=float((extended-original).abs().mean())
        initial_max=float((extended-original).abs().max())
        # The first H/4 convolution has a different channel count and CUDA
        # may select a different accumulation algorithm. Allow FP32 roundoff.
        torch.testing.assert_close(extended,original,rtol=1e-4,atol=2e-5)
    del baseline,x,extended,original
    optimizer=make_optimizer(model)
    scaler=torch.amp.GradScaler('cuda')
    model.train()
    shapes={'h8':[],'h4_concat':[],'h2_concat':[]}
    hooks=[]
    for name,module in (('h8',model.head.h8_in),('h4_concat',model.head.conv_in),
                        ('h2_concat',model.head.refine1)):
        hooks.append(module.register_forward_pre_hook(
            lambda _,inp,name=name: shapes[name].append(tuple(inp[0].shape))))
    steps=[]
    for step in range(1,3):
        start=time.time()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            logits=model(batch['img'].to(device))
            loss=P.loss_fn(logits,batch,device)
        assert logits.shape==(12,1,512,512) and torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grads=dict(h8_fusion=float(model.head.conv_in[0].weight.grad[:,256:].float().norm()),
                   h8_head=float(model.head.h8_in[0].weight.grad.float().norm()),
                   layer2=float(model.layer2[0].conv1.weight.grad.float().norm()))
        assert all(np.isfinite(v) for v in grads.values())
        assert grads['h8_fusion']>0 and model.conv1.weight.grad.abs().sum()>0
        if step==2:
            assert grads['layer2']>0 and grads['h8_head']>0, 'H/8 branch failed to learn'
        scaler.step(optimizer)
        scaler.update()
        assert model.head.conv_in[0].weight[:,256:].abs().sum()>0
        steps.append(dict(step=step,loss=float(loss.detach()),grad_norms=grads,
                          seconds=time.time()-start))
        print(f'[precheck step{step}] loss={float(loss.detach()):.6f} '
              f'fusion_grad={grads["h8_fusion"]:.8f} layer2_grad={grads["layer2"]:.8f}',flush=True)
    for hook in hooks:
        hook.remove()
    assert shapes==dict(h8=[(12,512,64,64)]*2,h4_concat=[(12,320,128,128)]*2,
                        h2_concat=[(12,128,256,256)]*2),shapes
    P.evaluate(model,P.loader(P.Samples(keys[:12],True),0),device,groups,label='precheck h8 scoring')
    with tempfile.TemporaryDirectory(prefix='small_h8_precheck_') as temp:
        path=Path(temp)/'model.pth'
        torch.save(model.state_dict(),path)
        fresh=S.SmallH8(P.WEIGHTS)
        fresh.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
        assert P.state_sha(fresh)==P.state_sha(model)
    result=dict(status='passed',training_batch=12,scoring_batch=12,image_size=512,
                original_initial_weights_identical=True,initial_logit_MAE=initial_mae,
                initial_logit_max_difference=initial_max,initial_tolerance=dict(rtol=1e-4,atol=2e-5),
                layer2_imagenet_loaded=True,layer2_matches_torchvision=True,
                params=sum(p.numel() for p in model.parameters()),baseline_params=ref['params'],
                feature_shapes=shapes,steps=steps,
                peak_gpu_GiB=torch.cuda.max_memory_allocated()/1024**3,
                training_dataset_has_fine_gt=False,checkpoint_reload_verified=True)
    P.dump(P.BASE/'h8_precheck.json',result)
    print(f'[precheck] PASS: params={result["params"]}; original initialization matched; '
          f'initial_logit_MAE={initial_mae:.9g}; two batch12 updates; H/8 gradients verified; '
          f'scoring/reload passed; peak_GPU={result["peak_gpu_GiB"]:.2f} GiB',flush=True)


def compare(out,reference,probes):
    analysis=out/'analysis'
    analysis.mkdir(exist_ok=True)
    records=[]
    for name,path in (('baseline_h4',reference),('h8',out)):
        records += [dict(arm=name,**r) for r in read_csv(path/'metrics.csv') if r['split']=='full5227']
    P.append_csv(analysis/'comparison.csv',records)
    lines=['# SMALL H/4 与 H/8 对照','',
           '新增ImageNet预训练ResNet50 layer2（H/8）及对应解码阶段；保留H/4、H/2跳接。',
           'H/8卷积后nearest上采样至H/4，与原H/4 concat；后续decoder沿用原结构。',
           '原有权重初始化、数据、弱监督损失和训练配置一致；不含attention。',
           '这是H/8编码/解码扩展对照，不是只改变编码器深度的单因素实验。',
           '阈值固定0.5；精标仅评分；全5227为固定目标恢复诊断，不是留出泛化评估。','',
           '|全5227域|H/4 P|R|IoU|H/8 P|R|IoU|IoU差值(pp)|',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for domain in ('agri','training_U','valid'):
        a=next(r for r in records if r['arm']=='baseline_h4' and r['group']=='ALL' and r['domain']==domain)
        b=next(r for r in records if r['arm']=='h8' and r['group']=='ALL' and r['domain']==domain)
        lines.append(f'|{domain}|{float(a["P_seg"]):.4f}|{float(a["R_seg"]):.4f}|'
                     f'{float(a["IoU_seg"]):.4f}|{float(b["P_seg"]):.4f}|'
                     f'{float(b["R_seg"]):.4f}|{float(b["IoU_seg"]):.4f}|'
                     f'{100*(float(b["IoU_seg"])-float(a["IoU_seg"])):+.3f}|')
    selected=sorted(random.Random(20261012).sample(sorted(probes),6))
    ds=P.Samples(selected,True)
    gallery=Image.new('RGB',(4*256,6*(256+48)),'white')
    draw=ImageDraw.Draw(gallery)
    font_path=Path('C:/Windows/Fonts/msyh.ttc')
    font=ImageFont.truetype(str(font_path),13) if font_path.is_file() else ImageFont.load_default()
    for i,key in enumerate(selected):
        basename=key.replace('/','__')
        pn=np.load(reference/'probes'/'ep10'/(basename+'.npy'))
        ph=np.load(out/'probes'/'ep10'/(basename+'.npy'))
        sample=ds[i]
        arrays=(ds.read('images',key,True),(sample['bnd'].numpy()*255).astype(np.uint8),
                ((pn>=.5)*255).astype(np.uint8),((ph>=.5)*255).astype(np.uint8))
        y=i*(256+48)
        draw.text((4,y),key,fill='black',font=font)
        for col,(label,array) in enumerate(zip(('RGB','Fine GT','Original H/4 >=0.5','H/8 >=0.5'),arrays)):
            draw.text((col*256+4,y+23),label,fill='black',font=font)
            gallery.paste(Image.fromarray(array).convert('RGB').resize((256,256),Image.Resampling.NEAREST),
                          (col*256,y+48))
    gallery.save(analysis/'random6_binary_comparison.png')
    lines += ['', '固定26张探针中按seed20261012随机选6张展示阈值0.5的二值图。',
              '单种子、固定ep10，不按精标选择模型；未追加低概率分布分析。','']
    (analysis/'结果摘要.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines[:14]),flush=True)


def train(args,keys,val_keys,groups,probes):
    if args.out.exists():
        raise FileExistsError(f'Refusing to overwrite existing experiment: {args.out}')
    model,ref,original_sha=build_model(args.reference)
    assert keys==ref['train_keys'] and probes==ref['probes']
    reference_epochs=read_csv(args.reference/'train.csv')
    assert len(reference_epochs)==10
    args.out.mkdir(parents=True)
    out=args.out
    device=torch.device('cuda')
    torch.cuda.reset_peak_memory_stats()
    model.to(device)
    optimizer=make_optimizer(model)
    scaler=torch.amp.GradScaler('cuda')
    train_dl=P.loader(P.Samples(keys),args.workers,True)
    val_dl=P.loader(P.Samples(val_keys,True),args.workers)
    full_dl=P.loader(P.Samples(keys,True),args.workers)
    probe_dl=P.loader(P.Samples(probes,True),0)
    manifest=dict(ref)
    manifest.update(arm='full_h8_ep10',initial_state_sha256=P.state_sha(model),
                    original_initial_state_sha256=original_sha,
                    layer2_initial_state_sha256=state_sha(model.layer2.state_dict()),
                    params=sum(p.numel() for p in model.parameters()),baseline_params=ref['params'],
                    decoder_upsampling='nearest',reference=str(args.reference),off_definition=None,
                    h8=dict(encoder='ImageNet ResNet50 stem+layer1+layer2',
                            layer2_blocks=4,layer2_output_channels=512,
                            decoder='H8 conv512->64->64, nearest to H4, concat original H4 (256)',
                            h4_fusion_channels=320,h4_fusion_output_channels=64,
                            h4_fusion_new_input_weights_init='zero',h4_skip=True,h2_skip=True,
                            attention=False,
                            interpretation='encoder plus corresponding decoder expansion'),
                    source_code_sha256={str(p):P.sha(p) for p in
                         (Path(__file__),Path(S.__file__),Path(P.__file__),Path(M.__file__))},
                    input_sha256={str(p):P.sha(p) for p in (P.FULL_SPLIT,P.OLD_SPLIT,P.WEIGHTS)},
                    torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name())
    P.dump(out/'manifest.json',manifest)
    attempts=updates=0
    start=time.time()
    P.dump(out/'status.json',dict(status='running',epoch=0))
    print(f'[h8] full5227, 10ep, batch12; params={manifest["params"]}; '
          'H/4+H/2 skips; nearest; no attention; matched original initialization',flush=True)
    for epoch in range(1,11):
        model.train()
        epoch_start=time.time()
        total_loss=0.
        order_hash=hashlib.sha256()
        for step,batch in enumerate(train_dl,1):
            order_hash.update(np.asarray(batch['idx'],dtype=np.int64).tobytes())
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                loss=P.loss_fn(model(batch['img'].to(device,non_blocking=True)),batch,device)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss ep{epoch} step{step}')
            scaler.scale(loss).backward()
            before=scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            updates+=int(scaler.get_scale()>=before)
            attempts+=1
            total_loss+=float(loss.detach())
            if step==1 or step%100==0 or step==len(train_dl):
                print(f'[h8 ep{epoch:02d}] step={step}/{len(train_dl)} '
                      f'loss={total_loss/step:.6f} elapsed_s={time.time()-epoch_start:.1f}',flush=True)
        assert order_hash.hexdigest()==reference_epochs[epoch-1]['sample_order_sha256']
        checkpoint=dict(epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),
                        scaler=scaler.state_dict(),cfg_exp='full_h8_ep10',
                        model_config=dict(architecture='SMALL_H8',h2_skip=True,h4_skip=True,
                                          decoder_upsampling='nearest',h8=manifest['h8']),
                        sup=dict(pos=2,ignore=1,bg_weight=1),train_attempts=attempts,
                        optimizer_updates=updates,manifest=manifest)
        tmp=out/f'ep{epoch:02d}.pth.tmp'
        torch.save(checkpoint,tmp)
        tmp.replace(out/f'ep{epoch:02d}.pth')
        rows,_=P.evaluate(model,val_dl,device,groups,label=f'h8 ep{epoch} old_val414')
        P.append_csv(out/'metrics.csv',[dict(epoch=epoch,split='old_val414',**r) for r in rows])
        P.evaluate(model,probe_dl,device,groups,probes,out/'probes'/f'ep{epoch:02d}',
                   label=f'h8 ep{epoch} probes')
        P.append_csv(out/'train.csv',[dict(epoch=epoch,loss=total_loss/len(train_dl),attempts=attempts,
                     optimizer_updates=updates,sample_order_sha256=order_hash.hexdigest(),
                     seconds=time.time()-epoch_start)])
        P.dump(out/'status.json',dict(status='running',epoch=epoch,attempts=attempts,
                                      optimizer_updates=updates))
    rows,images=P.evaluate(model,full_dl,device,groups,label='h8 ep10 full5227')
    P.append_csv(out/'metrics.csv',[dict(epoch=10,split='full5227',**r) for r in rows])
    P.append_csv(out/'full5227_per_image.csv',images)
    P.dump(out/'comparison_checks.json',dict(original_initial_weights_identical=True,
           layer2_imagenet_loaded=True,h4_skip_retained=True,h2_skip_retained=True,nearest_retained=True,
           epoch_sample_orders_identical=True,attempts=attempts,optimizer_updates=updates,
           reference=str(args.reference),changed='H/8 ResNet layer2 plus matching decoder stage'))
    compare(out,args.reference,probes)
    P.dump(out/'status.json',dict(status='complete',epoch=10,attempts=attempts,optimizer_updates=updates,
           runtime_s=time.time()-start,peak_gpu_GiB=torch.cuda.max_memory_allocated()/1024**3))
    print('[h8] COMPLETE; original initial weights and all 10 sample orders matched.',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--precheck',action='store_true')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--out',type=Path,default=P.BASE/'out'/'h8_ep10')
    parser.add_argument('--reference',type=Path,default=P.BASE/'out'/'pair_ep10'/'full_h2_on_ep10')
    args=parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    torch.backends.cudnn.benchmark=True
    keys,val_keys,groups,probes=P.data_info()
    if args.precheck:
        precheck(args,keys,groups)
        return
    try:
        train(args,keys,val_keys,groups,probes)
    except BaseException as exc:
        if not isinstance(exc,FileExistsError) and (args.out/'manifest.json').is_file():
            status_path=args.out/'status.json'
            prior=json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
            prior.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                         error=f'{type(exc).__name__}: {exc}')
            P.dump(status_path,prior)
        raise


if __name__=='__main__':
    main()
