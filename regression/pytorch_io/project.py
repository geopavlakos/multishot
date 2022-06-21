import argparse
import os
import sys

from yacs.config import CfgNode as CN
from .config import default_config

class ProjectRegistration(type):
    def __init__(cls, name, bases, nmspc):
        super().__init__(name, bases, nmspc)
        if not hasattr(cls, 'registry'):
            cls.registry = dict()
        cls.registry[name] = cls

    # Metamethods, called on class objects:
    def __iter__(cls):
        return iter(cls.registry)

    def __str__(cls):
        return str(cls.registry)

class Project(metaclass=ProjectRegistration):
    def get_runnable(self, config):
        raise NotImplementedError("Sub-classes must implement get_runnable")

    @staticmethod
    def get_experiment_file(args):
        base_experiment_path = os.path.join(args.saved_experiment_dir, args.name)
        experiment_config_file_path = os.path.join(base_experiment_path, "config.yaml")
        can_reload = args.reload and os.path.exists(experiment_config_file_path)

        if can_reload:
            with open(experiment_config_file_path, 'r') as f:
                cfg = CN.load_cfg(f)
                cfg.freeze()
        else:
            cfg = default_config()
            cfg.merge_from_file(args.new_experiment_config)
            cfg.GENERAL.CHECKPOINT_DIR = os.path.join(base_experiment_path, cfg.GENERAL.CHECKPOINT_DIR)
            cfg.GENERAL.SUMMARY_DIR = os.path.join(base_experiment_path, cfg.GENERAL.SUMMARY_DIR)
            cfg.GENERAL.LOCAL_RANK = args.local_rank
            cfg.GENERAL.WORLD_SIZE = args.world_size
            os.makedirs(base_experiment_path, exist_ok=True)
            cfg.freeze()

            with open(experiment_config_file_path, 'w') as f:
                cfg.dump(stream=f)


        return cfg

    @staticmethod
    def get_parser(parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()
        req = parser.add_argument_group("Required")
        req.add_argument("--name", type=str, required=True)
        req.add_argument("--reload", action="store_true")
        req.add_argument(
                "--new_experiment_config",
                type=str,
                help="The base experiment configuration",
                required=True,
                )
        req.add_argument(
                "--saved_experiment_dir",
                type=str,
                default="logs",
                help="The base directory of all saved experiments",
                required=True,
                )
        dist = parser.add_argument_group("Distributed")
        dist.add_argument('--local_rank', type=int, default=0)
        dist.add_argument('--world_size', type=int, default=1)
        return parser
