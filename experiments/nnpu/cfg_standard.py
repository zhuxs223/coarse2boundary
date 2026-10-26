"""Fixed-target nnPU-Oracle adaptation; paths selected through the public release configuration."""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = Path(os.environ.get('EXP810_ROOT', str(HERE.parents[1]))).resolve()
DATA_ROOT = Path(os.environ.get('EXP810_DATA', str(BASE / 'data/GTPBD_enhenced_png')))
SPLIT_JSON = Path(os.environ.get('EXP810_SPLIT', str(BASE / 'meta/splits_full5227.json')))
RESNET50_W = Path(os.environ.get('EXP810_RESNET', str(BASE / 'weights/resnet50-0676ba61.pth')))
RESULTS_ROOT = BASE / 'out'
RUN_NAME = 'unet_pu'
EPOCHS=int(os.environ.get('EXP810_EPOCHS','100'))
PU_WEIGHT = 1.0
LR_DROP_EPOCH = 50
LR_AFTER_DROP = 1e-5
BATCH_SIZE = 8
INFER_BATCH = 8
NUM_WORKERS=int(os.environ.get('EXP810_WORKERS','6'))
CPU_LIMIT = 30
AMP = True
LR = 1e-4
WEIGHT_DECAY = .01
SEED = 42
IMG_SIZE = 512
EDGE_TRIM = 1
EVALUATE_EVERY = 10
SAVE_EVERY = 10
MIN_FREE_GIB = 5
DECODER_INTERPOLATION = 'bilinear'
MODEL_DIR = HERE / 'reference'
LAYERS = dict(img='images', msk='mask_labels', mk3='mask_labels_3px', bnd='boundary_labels')
IMAGENET_MEAN = (.485, .456, .406)
IMAGENET_STD = (.229, .224, .225)
AUGMENT = dict(horizontal_flip=.5, vertical_flip=.5, rotate90=True,
               brightness_contrast_probability=.5, brightness_limit=.1, contrast_limit=.1)
PROBE_KEYS = ['Central China/baidi_0_5_11520_13312',
              'Central China/baidi_0_5_9984_11520',
              'Central China/baidi_0_5_9472_14592',
              'Central China/baidi_0_5_14336_14336',
              'Southwest China/128_768_21248']


def priors_path():
    return output() / 'oracle_priors.json'


def output():
    return RESULTS_ROOT / RUN_NAME
