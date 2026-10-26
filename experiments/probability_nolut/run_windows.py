"""Foreground Windows launcher, same interpreter, merged live UTF-8 file log."""
from pathlib import Path
import argparse,datetime,os,subprocess,sys
import cfg_standard as C


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--precheck',action='store_true')
    mode.add_argument('--smoke',action='store_true')
    parser.add_argument('--rounds',type=int,default=10,help='10 rounds = 100 student epochs; use --rounds 20 for 200 epochs')
    args=parser.parse_args()
    if not 1<=args.rounds<=30:parser.error('--rounds must be between 1 and 30')
    if args.smoke:C.RUN_NAME='_smoke_'+C.RUN_NAME+'_'+C.SMOKE_RUN_ID
    flags=['--precheck'] if args.precheck else ['--smoke'] if args.smoke else []
    prefix='precheck' if args.precheck else 'smoke' if args.smoke else 'train'
    logs=C.output()/'logs';logs.mkdir(parents=True,exist_ok=True)
    path=logs/(prefix+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.log')
    env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',EXP810_ROOT=str(C.BASE),
        EXP810_STUDENT='unet',EXP810_ABLATION_ROUNDS=str(args.rounds))
    command=[sys.executable,'-u',str(C.HERE/'acceleration/run_fast.py'),*flags]
    print(f'Python: {sys.executable}\nRoot: {C.BASE}\nLog: {path}',flush=True)
    with path.open('w',encoding='utf-8',buffering=1) as log:
        log.write(f'Python: {sys.executable}\nRoot: {C.BASE}\nCommand: {command!r}\n')
        child=subprocess.Popen(command,cwd=C.HERE,env=env,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
        try:
            for line in child.stdout:
                log.write(line);log.flush();sys.stdout.write(line);sys.stdout.flush()
            code=child.wait()
        except KeyboardInterrupt:
            # The console delivers Ctrl+C to both parent and child. Let the
            # runner release its lock and worker pools before forcing shutdown.
            print('\nStopping this run...',flush=True)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                if os.name=='nt':subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],check=False)
                else:child.terminate();child.wait()
            code=130
        finally:child.stdout.close()
        log.write(f'\nExit code: {code}\n')
    print(f'Exit code: {code}; log: {path}',flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
