import torch
import pytorch_io
import projects

parser = pytorch_io.Project.get_parser()
args = parser.parse_args()
cfg = pytorch_io.Project.get_experiment_file(args)

project_type = pytorch_io.Project.registry[cfg.PROJECT]
callable_func = project_type().get_runnable(cfg)
local_rank = cfg.GENERAL.LOCAL_RANK
world_size = cfg.GENERAL.WORLD_SIZE
callable_func()
