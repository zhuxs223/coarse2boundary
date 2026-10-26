"""Foreground Windows entry; right-click Run or python run_windows.py."""
from pathlib import Path
import argparse
import datetime
import os
import subprocess
import sys
import cfg_standard as C


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precheck',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--root',type=Path,default=C.BASE)
    parser.add_argument('--teacher',type=Path)
    parser.add_argument('--epochs',type=int,default=C.EPOCHS)
    parser.add_argument('--smoke-id',default='v1')
    parser.add_argument('--smoke-epochs',type=int,default=4)
    parser.add_argument('--smoke-warm',type=int,default=2)
    parser.add_argument('--smoke-workers',type=int,default=0)
    args=parser.parse_args()
    C.BASE=args.root.resolve(); C.RESULTS_ROOT=C.BASE/'out'
    if args.smoke:C.RUN_NAME='_smoke_unet_s2l_small_prior_'+args.smoke_id
    prefix='precheck' if args.precheck else 'smoke' if args.smoke else 'train'
    logroot=C.output()/'logs'
    logroot.mkdir(parents=True,exist_ok=True)
    path=logroot/(prefix+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.log')
    env=os.environ.copy()
    env.update(PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',EXP810_ROOT=str(C.BASE))
    if args.teacher:env['EXP810_SMALL_TEACHER']=str(args.teacher.resolve())
    command=[sys.executable,'-u',str(C.HERE/'run.py'),'--epochs',str(args.epochs)]
    if args.precheck:command.append('--precheck')
    if args.smoke:command += ['--smoke','--smoke-id',args.smoke_id,'--smoke-epochs',str(args.smoke_epochs),
                              '--smoke-warm',str(args.smoke_warm),'--smoke-workers',str(args.smoke_workers)]
    print(f'Python: {sys.executable}\nRoot: {C.BASE}\nLog: {path}',flush=True)
    with path.open('w',encoding='utf-8',buffering=1) as log:
        log.write(f'Python: {sys.executable}\nRoot: {C.BASE}\nCommand: {command!r}\n')
        child=subprocess.Popen(command,cwd=C.HERE,env=env,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
        try:
            for line in child.stdout:
                # File write precedes console rendering, so the log remains usable if the IDE console fails.
                log.write(line)
                log.flush()
                try:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                except (OSError,UnicodeError):
                    pass
            code=child.wait()
        except KeyboardInterrupt:
            print('\nStopping; the last completed epoch remains resumable.',flush=True)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                if os.name=='nt':subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],check=False)
                else:child.terminate();child.wait()
            code=130
        finally:
            child.stdout.close()
        log.write(f'\nExit code: {code}\n')
    print(f'Exit code: {code}; log: {path}',flush=True)
    return code


if __name__=='__main__':
    raise SystemExit(main())
