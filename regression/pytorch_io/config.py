from yacs.config import CfgNode as CN


_C = CN(new_allowed=True)

_C.GENERAL = CN(new_allowed=True)
_C.GENERAL.RESUME = True
_C.GENERAL.TIME_TO_RUN = 3600000
_C.GENERAL.SUMMARY_STEPS = 100
_C.GENERAL.CHECKPOINT_STEPS = 20000
_C.GENERAL.CHECKPOINT_DIR = "checkpoints"
_C.GENERAL.SUMMARY_DIR = "tensorboard"
_C.GENERAL.NUM_GPUS = 1
_C.GENERAL.NUM_WORKERS = 4
_C.GENERAL.ALLOW_CUDA = True
_C.GENERAL.PIN_MEMORY = False
_C.GENERAL.DISTRIBUTED = False
_C.GENERAL.LOCAL_RANK = 0
_C.GENERAL.USE_SYNCBN = False
_C.GENERAL.WORLD_SIZE = 1

_C.TRAIN = CN(new_allowed=True)
_C.TRAIN.NUM_EPOCHS = 100
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.SHUFFLE = True
_C.TRAIN.WARMUP = False
_C.TRAIN.NORMALIZE_PER_IMAGE = False
_C.TRAIN.CLIP_GRAD = False
_C.TRAIN.CLIP_GRAD_VALUE = 1.0
_C.LOSS_WEIGHTS = CN(new_allowed=True)

_C.DATASETS = CN(new_allowed=True)

_C.MODEL = CN(new_allowed=True)

_C.EXTRA = CN(new_allowed=True)

def default_config():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
