from .smpl_wrapper import SMPL
from .base_model import BaseModel

def create_model(cfg):
    return BaseModel(cfg)
