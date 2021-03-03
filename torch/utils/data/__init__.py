import torch._C
from .sampler import Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler, WeightedRandomSampler, BatchSampler
from .dataset import (Dataset, IterableDataset, TensorDataset, ConcatDataset, ChainDataset,
                      Subset, random_split)
from .dataset import IterableDataset as IterDataPipe
from .dataset import functional_datapipe
from .distributed import DistributedSampler
from .dataloader import DataLoader, _DatasetKind, get_worker_info
from . import datapipes
from typing import Any

__all__ = ['Sampler', 'SequentialSampler', 'RandomSampler',
           'SubsetRandomSampler', 'WeightedRandomSampler', 'BatchSampler',
           'DistributedSampler', 'Dataset', 'IterableDataset', 'TensorDataset',
           'ConcatDataset', 'ChainDataset', 'Subset', 'random_split',
           'DataLoader', '_DatasetKind', 'get_worker_info',
           'IterDataPipe', 'functional_datapipe', 'set_deterministic']



class set_deterministic(object):
    prev: bool

    def __init__(self, mode: bool) -> None:
        self.prev = torch._C._get_deterministic_dataloader()
        torch._C._set_deterministic_dataloader(mode)

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch._C._set_deterministic_dataloader(self.prev)
