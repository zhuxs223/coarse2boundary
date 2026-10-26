"""Applied BEFORE torch/numpy imports, including spawned CPU workers."""
import os
import warnings
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
SETTINGS=dict(MIOPEN_FIND_MODE='FAST',MIOPEN_FIND_ENFORCE='NONE',
    PYTORCH_TUNABLEOP_ENABLED='0',PYTORCH_TUNABLEOP_TUNING='0',
    PYTORCH_TUNABLEOP_RECORD_UNTUNED='0',MIOPEN_LOG_LEVEL='3',
    MIOPEN_ENABLE_LOGGING='0',MIOPEN_ENABLE_LOGGING_CMD='0',
    OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
for name,value in SETTINGS.items():os.environ[name]=value
warnings.filterwarnings('ignore',message=r'.*(MIOpen|TunableOp).*',category=UserWarning)

def cpu_capacity(cap=8):
    from pathlib import Path
    n=min(cap,len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1)
    try:
        q,p=Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if q!='max':n=min(n,max(1,int(q)//int(p)))
    except (OSError,ValueError):
        try:
            q=int(Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read_text());p=int(Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read_text())
            if q>0:n=min(n,max(1,q//p))
        except (OSError,ValueError):pass
    return max(1,n)

def configure():
    import torch,cv2
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False
    if torch.version.hip and hasattr(torch.cuda,'tunable'):
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.tuning_enable(False)
