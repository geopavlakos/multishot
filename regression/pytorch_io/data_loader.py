import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import Sampler

class CheckpointSampler(Sampler):
    """ Handles the loading of a checkpoint inside of a sampler

    Subclasses need to implement next_dataset_perm
    """
    def __init__(self, data_source):
        self.data_source = data_source
        self.is_batch_sampler = False

    def next_epoch(self, checkpoint):
        if checkpoint is not None and checkpoint['dataset_perm'] is not None:
            self.dataset_perm = checkpoint['dataset_perm']
            if self.is_batch_sampler:
                self.perm = self.dataset_perm[checkpoint['batch_idx']:]
            else:
                self.perm = self.dataset_perm[checkpoint['batch_size']*checkpoint['batch_idx']:]
        else:
            self.dataset_perm = self.next_dataset_perm()
            self.perm = self.dataset_perm

    def __iter__(self):
        return iter(self.perm)
    
    def __len__(self):
        return len(self.perm)

class RandomSampler(CheckpointSampler):
    def __init__(self, data_source):
        super().__init__(data_source)

    def next_dataset_perm(self):
        return torch.randperm(len(self.data_source)).tolist()

class SequentialSampler(CheckpointSampler):

    def __init__(self, data_source):
        super().__init__()

    def next_dataset_perm(self):
        return list(range(len(self.data_source)))


class WeightedRandomSampler(CheckpointSampler):
    """
    Samples from a data_source with weighted probabilities for each element.
    Weights do not need to sum to 1. 
    Typical use case is when you have multiple datasets, the weights for each dataset are
    set to 1/len(ds). This ensures even sampling amongst datasets with different lengths.
    weights - tensor with numel=len(data_source)
    
    """
    def __init__(self, data_source, weights):
        super(WeightedRandomSampler, self).__init__(data_source)
        self.data_source = data_source
        self.weights = weights

    def next_dataset_perm(self):
        return torch.multinomial(self.weights, len(self.data_source), replacement=True).tolist()

class CheckpointDataLoader(DataLoader):
    def __init__(self, dataset, checkpoint=None, batch_size=1,
                 shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, pin_memory=False, drop_last=True,
                 worker_init_fn=None, timeout=0,
                 collate_fn=default_collate):

        if sampler is None and batch_sampler is None:
            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)

        if batch_sampler is not None:
            sampler = None
            drop_last = False
            batch_size = 1
            shuffle = False

        if checkpoint is not None:
            self.checkpoint_batch_idx = checkpoint['batch_idx']
        else:
            self.checkpoint_batch_idx = 0

        super().__init__(dataset, sampler=sampler,
                shuffle=False, batch_size=batch_size,
                drop_last=drop_last,
                pin_memory=pin_memory,
                timeout=timeout,
                num_workers=num_workers,
                worker_init_fn=worker_init_fn,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn)

    def get_dataset_perm(self):
        if isinstance(self.sampler, CheckpointSampler):
            return self.sampler.dataset_perm
        elif isinstance(self.batch_sampler, CheckpointSampler):
            return self.batch_sampler.dataset_perm
        else:
            return None

    def next_epoch(self, checkpoint):
        if checkpoint is not None:
            self.checkpoint_batch_idx = checkpoint['batch_idx']
        else:
            self.checkpoint_batch_idx = 0

        if isinstance(self.sampler, CheckpointSampler) or hasattr(self.sampler, "next_epoch"):
            return self.sampler.next_epoch(checkpoint)
        elif isinstance(self.batch_sampler, CheckpointSampler) or hasattr(self.batch_sampler, "next_epoch"):
            return self.batch_sampler.next_epoch(checkpoint)
        else:
            raise ValueError("Tried next_epoch, but neither\
                              sampler nor batch_sampler inherit from\
                              CheckpointSampler or have implemented next_epoch")
