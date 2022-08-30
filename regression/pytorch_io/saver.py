from __future__ import division
import os
import datetime

import torch

class CheckpointSaver(object):
    def __init__(self, save_dir, should_log):
        self.save_dir = os.path.abspath(save_dir)
        if not os.path.exists(self.save_dir) and should_log:
            os.makedirs(self.save_dir)
        self._get_latest_checkpoint()

    # check if a checkpoint exists in the current directory
    def exists_checkpoint(self, checkpoint_file=None):
        if checkpoint_file is None:
            return False if self.latest_checkpoint is None else True
        else:
            return os.path.isfile(checkpoint_file)

    # save checkpoint
    def save_checkpoint(self, stateful_objects, state):
        """
        Saves the current state that is handed to it for future use.

        Arguments:
        - stateful_objects - All objects from that network that implement state_dict()
        - state - All objects that should be saved as is (i.e. epoch, batch_idx, batch_size, dataset_perm, total_step_count, etc.)
        """
        timestamp = datetime.datetime.now()
        checkpoint_filename = os.path.abspath(os.path.join(self.save_dir, timestamp.strftime('%Y_%m_%d-%H_%M_%S') + '.pt'))
        checkpoint = {}

        for so in stateful_objects:
            checkpoint[so] = stateful_objects[so].state_dict()

        for s in state:
            checkpoint[s] = state[s]

        print('Saving checkpoint file [' + checkpoint_filename + ']')
        torch.save(checkpoint, checkpoint_filename)

    # load a checkpoint
    def load_checkpoint(self, stateful_objects, checkpoint_file=None, device=None):
        """
        Loads a checkpoint into stateful_objects and passes all leftover keys back to the caller.

        Arguments:
        - stateful_objects - All objects that need to be loaded again.
        - checkpoint_file - The checkpoint file to use, if None grab the latest_checkpoint file.
        """

        if checkpoint_file is None:
            print('Loading latest checkpoint [' + self.latest_checkpoint + ']')
            checkpoint_file = self.latest_checkpoint
        else:
            print('Loading checkpoint [' + checkpoint_file + ']')

        checkpoint = torch.load(checkpoint_file, map_location=device)
        checkpoint_keys = set(checkpoint.keys())

        for so in stateful_objects:
            if so in checkpoint:
            #if so in checkpoint and so == 'model':
                #stateful_objects[so].load_state_dict(checkpoint[so], strict=False)
                try:
                    stateful_objects[so].load_state_dict(checkpoint[so], strict=False)
                except TypeError:
                    stateful_objects[so].load_state_dict(checkpoint[so])
                    checkpoint_keys.remove(so)

        extra_state = {k: checkpoint[k] for k in checkpoint_keys}

        return extra_state

    def _get_latest_checkpoint(self):
        """
        Checks for existance of checkpoints inside self.save_dir. Stores and returns the latest checkpoint as specified by the filename.

        Returns: The path of the latest checkpoint
        """

        checkpoint_list = []
        for dirpath, dirnames, filenames in os.walk(self.save_dir):
            for filename in filenames:
                if filename.endswith('.pt'):
                    checkpoint_list.append(os.path.abspath(os.path.join(dirpath, filename)))
        checkpoint_list = sorted(checkpoint_list)
        self.latest_checkpoint =  None if (len(checkpoint_list) is 0) else checkpoint_list[-1]
        return self.latest_checkpoint
