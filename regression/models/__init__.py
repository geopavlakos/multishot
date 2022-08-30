from .smpl_wrapper import SMPL
from .base_model import BaseModel
from .temporal_model import TemporalModel

def create_model(cfg):
    return BaseModel(cfg)

def create_temporal_model(cfg):
    return TemporalModel(cfg)
