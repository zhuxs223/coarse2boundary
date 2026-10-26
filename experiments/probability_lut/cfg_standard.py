"""Probability-only admission ablation; fixed teacher, LUT and loss settings."""
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent

def resolve_project_root(here,override=None):
    """Support direct or nested ZIP extraction; explicit root still takes priority."""
    if override:
        return Path(override).resolve()
    for candidate in Path(here).resolve().parents:
        if (candidate/'meta/full_teacher_provenance.json').is_file():
            return candidate
    return Path(here).resolve().parent

BASE=resolve_project_root(HERE,os.environ.get('EXP810_ROOT'))
MODEL_DIR=HERE/'reference'
DATA_ROOT=Path(os.environ.get('EXP810_DATA',str(BASE/'data/GTPBD_enhenced_png')))
SPLIT_JSON=Path(os.environ.get('EXP810_SPLIT',str(BASE/'meta/splits_full5227.json')))
INITIAL_SMALL_TEACHER=Path(os.environ.get('EXP810_TEACHER',str(BASE/'meta/small_full5227_h4_ep10.pth')))
RESNET50_W=Path(os.environ.get('EXP810_RESNET',str(BASE/'weights/resnet50-0676ba61.pth')))
RESULTS_ROOT=BASE/'out'
STUDENT_ARCH='unet'
LUT_MODE='on'
LUT_ENABLED=True
STRUCTURE_ENABLED=False
RUN_NAME='unet_fullteacher_probability'
SMOKE_RUN_ID='probability_ablation_v1'
ROUNDS=int(os.environ.get('EXP810_ABLATION_ROUNDS','10'))
if not 1<=ROUNDS<=30:raise ValueError('Rounds must be between 1 and 30')
EPOCHS_PER_ROUND=10
DATA_PROTOCOL='fixed-target-full5227-v1'
EXPECTED_TARGET_TILES=5227
EVALUATE_EVERY_EPOCHS=10  # Full target GT monitoring only; always score round end.
METRIC_SPLIT='target_full'  # All scored images also participate in weak training.
BATCH_SIZE=8
INFER_BATCH=16
NUM_WORKERS=int(os.environ.get('EXP810_WORKERS','6'))
GEOMETRY_WORKERS=16
ANCHOR_LOAD_WORKERS=2
CPU_LIMIT=30  # Ryzen 9950X workstation; re-capped on every launch.
MAX_OUTPUT_GIB=80.0
MIN_FREE_GIB=10.0
STATE_KEEP_ROUNDS=None  # Keep independent historical NPZs while budget allows.
AMP=True
LR=1e-4
WEIGHT_DECAY=.01
SEED=42
OUTSIDE_WEIGHTS=[1.,1.,.75,.5,.25,0.]
DECODER_INTERPOLATION='nearest' if STUDENT_ARCH=='h8' else 'bilinear'
NO_REPEAT_BONUS=False  # No geometry-dependent maintained/repeated classification.
REPEAT_CONFIDENCE_ENABLED=False
REPEAT_CONFIDENCE_ETA=.2
REPEAT_CONFIDENCE_STOP=.95
REPEAT_CONFIDENCE_OUTLINE_MARGIN=5
REPEAT_CONFIDENCE_TOLERANCE=1
GUARD_ENABLED=True
GUARD_PREVIEW_LIMIT=8
PSEUDO_WEIGHT=.125  # Same current fullteacher baseline internal negative-loss multiplier.
FIRST_SOFT_WEIGHT=1.
SOFT_WEIGHT=1.
ANCHOR_SEARCH_RADIUS=2
ANCHOR_BAND_RADIUS=1
EXCLUSION_EXTRA_RADIUS=3
RIDGE_BONUS=0.
NONRIDGE_PENALTY=0.
REJECTED_RIDGE_PENALTY=0.
LAYERS=dict(img='images',msk='mask_labels',mk3='mask_labels_3px',bnd='boundary_labels')
TRAIN_SPLIT='train_full'
VAL_SPLIT='target_full'
IMG_SIZE=512
EDGE_TRIM=1
IMAGENET_MEAN=(.485,.456,.406)
IMAGENET_STD=(.229,.224,.225)
MODE='spatial'
EXPONENT=2
IGNORE_THRESHOLD=.5
OUTLINE_TARGET=1.
THR=.5
START_EPOCH=0
PROGRESS_EVERY=50
GEOMETRY_DIR=BONUS_DIR=MAINTAINED_DIR=FROZEN_DIR=None
FROZEN_KEYS=[]
PROBE_KEYS=['Central China/baidi_0_5_11520_13312',
            'Central China/baidi_0_5_9984_11520',
            'Central China/baidi_0_5_9472_14592',
            'Central China/baidi_0_5_14336_14336',
            'Southwest China/128_768_21248']

def output():return RESULTS_ROOT/RUN_NAME
def stage(index):
    return dict(index=index,round=index+1,start=index*EPOCHS_PER_ROUND,
                end=(index+1)*EPOCHS_PER_ROUND,epochs=EPOCHS_PER_ROUND,
                alpha=OUTSIDE_WEIGHTS[min(index,len(OUTSIDE_WEIGHTS)-1)])
