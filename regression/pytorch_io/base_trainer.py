import sys
import signal
import time
from collections.abc import Iterable

import torch
from torch.utils.data import DataLoader
import numpy as np

# import tqdm and configure
from tqdm import tqdm
tqdm.monitor_interval = 0

# Grab SummaryWriter from either pytorch or tensorboardX
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from .saver import CheckpointSaver
from .data_loader import CheckpointDataLoader
from .hook_manager import HookManager

def lr_lambda_warmup(step):
    interval = step
    print('This is the step', step)
    print(step / 1000)
    if interval < 1000:
        return step / 1000
    else:
        return 1

class BaseTrainer(HookManager):
    """ BaseTrainer class to be inherited

    This class includes hooks to run arbitrary functions at useful times throughout the training process.

    init_fn is a good place to define:
    - models_dict - a dictionary that contains all models used
    - optimizers_dict - a dictionary that contains all optimizers used
    - train_ds - the dataset to use for training

    However these just need to be defined BEFORE running train()
    """

    def __init__(self, cfg):
        super().__init__()


        self.cfg = cfg
        self.external_training_stop = None

        self.device = torch.device('cuda' if torch.cuda.is_available() and cfg.GENERAL.ALLOW_CUDA else 'cpu')
        self.device_id = self.cfg.GENERAL.LOCAL_RANK
        self.local_rank = self.cfg.GENERAL.LOCAL_RANK
        self.should_log = not self.cfg.GENERAL.DISTRIBUTED or self.local_rank == 0

        # At startup load the checkpoint and setup the learning rate schedulers
        # override this function to define your model, optimizers etc.
        self.add_start_hook(self.init_fn)
        self.add_start_hook(self.load_checkpoint)

        # Add hooks to create and iterate the data loader
        self.add_start_hook(self.create_data_loader)
        # self.add_pre_epoch_hook(self.next_dataset_perm)

        # Add a hook to safely stop after time_to_run seconds
        self.add_cron_hooks(self.safe_stop, cfg.GENERAL.TIME_TO_RUN)

        # Add signal handler to flag for a safe stop
        signal.signal(signal.SIGTERM, self.safe_stop)

        # Add hooks for what should be run every certain number of steps
        # If the number of steps is <=0 this will just not add
        self.add_step_cron_hooks(self._train_summaries, cfg.GENERAL.SUMMARY_STEPS, True)

        # if "lr_decay" in self.options:
        if cfg.TRAIN.WARMUP:
            self.add_start_hook(self.lr_schedulers_setup)
            self.add_step_cron_hooks(self.lr_schedulers_step, 1, True)
        #     # After each epoch step the learning rates forward
        #     self.add_post_epoch_hook(self.lr_schedulers_step)

        # Create a CheckpointSaver and add hooks for various times to save
        self.saver = CheckpointSaver(save_dir=cfg.GENERAL.CHECKPOINT_DIR, should_log=self.should_log)
        self.add_step_cron_hooks(self.save_checkpoint, cfg.GENERAL.CHECKPOINT_STEPS, True)
        self.add_finalize_hook(self.save_checkpoint)

        self._train_state = {}

        self.epoch_count = 0
        self.num_epochs = cfg.TRAIN.NUM_EPOCHS
        self._train_state['step_count'] = 0
        self.checkpoint = None
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.num_workers = cfg.GENERAL.NUM_WORKERS
        self.pin_memory = cfg.GENERAL.PIN_MEMORY
        self.shuffle_train = cfg.TRAIN.SHUFFLE
        self.resume = cfg.GENERAL.RESUME

        # tensorboardX SummaryWriter for use in train_summaries
        if self.should_log:
            self.summary_writer = SummaryWriter(cfg.GENERAL.SUMMARY_DIR)

    def create_data_loader(self):
        # Create the dataloader for use during training
        self._train_state["data_loader"] = DataLoader(self.train_ds,
                    batch_size = self.batch_size,
                    num_workers = self.num_workers,
                    pin_memory = self.pin_memory,
                    shuffle = self.shuffle_train,
                )

    def next_dataset_perm(self):
        # setup the next epoch inside of train_data_loader
        # this will include the next dataset permutation
        self._train_state['data_loader'].next_epoch(self.checkpoint)

    def safe_stop(self, exit_code=0):
        tqdm.write('Requesting stop with exit code [%d] after current batch' % exit_code)
        self.external_training_stop = exit_code

    def save_checkpoint(self):
        for k in self.models_dict.keys():
            if k in self.optimizers_dict:
                raise KeyError("self.models_dict and self.optimziers_dict cannot have the same keys")

        stateful_objects = {**self.models_dict, **self.optimizers_dict}
        if hasattr(self, 'amp'):
            stateful_objects['amp'] = self.amp
        if self.should_log:
            self.saver.save_checkpoint(stateful_objects,
                   {"epoch": self._train_state['epoch'],
                    # "batch_idx": self._train_state['step']+1,
                    "batch_size": self.batch_size,
                    # "dataset_perm": self._train_state['data_loader'].get_dataset_perm(),
                    "total_step_count": self._train_state['step_count']})

    def load_checkpoint(self):
        self.models_dict = {k:v.to(self.device)
                for k,v in self.models_dict.items()}

        # Load the latest checkpoints
        self.checkpoint = None
        stateful_objects = {**self.models_dict, **self.optimizers_dict}
        if 'PRETRAINED_CHECKPOINT' in self.cfg.MODEL:
            checkpoint_file = self.cfg.MODEL.PRETRAINED_CHECKPOINT
        else:
            checkpoint_file = None
        if hasattr(self, 'amp'):
            stateful_objects['amp'] = self.amp
        if self.resume and checkpoint_file is not None:
            self.checkpoint = self.saver.load_checkpoint(stateful_objects, device=f'cuda:{self.device_id}', checkpoint_file=checkpoint_file)
        if self.resume and self.saver.exists_checkpoint():
            self.checkpoint = self.saver.load_checkpoint(stateful_objects, device=f'cuda:{self.device_id}')

        # Reload epoch and step count if a checkpoint was loaded
        if self.checkpoint is not None:
            if 'epoch' in self.checkpoint and 'total_step_count' in self.checkpoint:
                self.epoch_count = self.checkpoint['epoch']
                self._train_state['step_count'] = self.checkpoint['total_step_count']

    def lr_schedulers_setup(self):
        self.lr_schedulers = {k: torch.optim.lr_scheduler.LambdaLR(v, lr_lambda_warmup, last_epoch=self._train_state['step_count']-1)\
                              for k,v in self.optimizers_dict.items()}

    def lr_schedulers_step(self):
        for opt in self.optimizers_dict:
            self.lr_schedulers[opt].step()

    # It works for target=device or target=dtype
    # I replaced Iterable with list because this would iterate over strings as well
    def recursive_to(self, x, target):
        if isinstance(x, dict):
            return {k: self.recursive_to(v, target) for k, v in x.items()}
        elif isinstance(x, torch.Tensor):
            return x.to(target)
        elif isinstance(x, list):
            return [self.recursive_to(i, target) for i in x]
        else:
            return x

    def train(self):
        # Create the dataloader that will generate the data
        # permutation for each epoch
        np.random.seed()
        torch.manual_seed(np.random.randint(2**32) + self.local_rank)
        self._run_start_hooks()

        for self._train_state['epoch'] in tqdm(range(self.epoch_count, self.num_epochs),
                total=self.num_epochs, initial=self.epoch_count):

            self._run_pre_epoch_hooks()

            # If we have received any reason to stop, stop
            if self.external_training_stop is not None:
                break

            for step, batch in enumerate(tqdm(self._train_state['data_loader'], desc='Epoch '+str(self._train_state['epoch']),
                                              total=len(self.train_ds) // self.batch_size,
                                              initial=0)):
                self._train_state['step'] = step

                # Cron hooks are run first since time checks normally involve exiting or similar
                self._run_cron_hooks()

                # If we have received any reason to stop, stop
                if self.external_training_stop is not None:
                    break

                # Run the network
                self._train_state['batch'] = self.recursive_to(batch, self.device)
                self._train_state['out'] = self.train_step(self._train_state['batch'])

                # Iterate the step
                self._train_state['step_count'] += 1

                # Run the step_cron if it is time
                if self._train_state['step_count'] % self.step_cron_gcd == 0:
                    self._run_step_cron_hooks(self._train_state['step_count'])

            self._run_post_epoch_hooks()

            # Once again for the post hooks
            if self.external_training_stop is not None:
                break

            # load a checkpoint only on startup, for the next epochs
            # just iterate over the dataset as usual
            self.checkpoint=None

        self._run_finalize_hooks()

    def get_lr(self):
        return next(iter(self.lr_schedulers.values())).get_lr()[0]

    def init_fn(self):
        raise NotImplementedError('You need to provide an init_fn method')

    def train_step(self, input_batch):
        raise NotImplementedError('You need to provide a train_step method')

    def train_summaries(self, input_batch, model_output):
        raise NotImplementedError('You need to provide a train_summaries method')

    def _train_summaries(self):
        self.train_summaries(self._train_state['batch'],  self._train_state['out'])

    def test(self):
        raise NotImplementedError('You need to provide a test method')
