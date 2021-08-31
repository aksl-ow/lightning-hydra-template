import torch
from torch.utils.data.dataloader import default_collate
from typing import List, Optional, Any, Tuple
import pytorch_lightning as pl
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    StratifiedKFold,
    RepeatedStratifiedKFold,
    KFold,
    RepeatedKFold,
)

from classic_algos.utils.data import SlideFeaturesDataset


def pad_collate_fn(
    batch: List[Tuple[torch.Tensor, Any]],
    batch_first: bool = True,
    max_len: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pads together sequences of arbitrary lengths
    Adds a mask of the padding to the samples that can later be used
    to ignore padding in activation functions.
    Expected to be used in combination of a torch.utils.datasets.DataLoader.
    Expect the sequences to be padded to be the first one in the sample tuples.
    Others members will be batched using default_collate
    Parameters
    ----------
    batch: List[Tuple[torch.Tensor, Any]]
    batch_first: bool = True
        Either return (B, N_TILES, F) or (N_TILES, B, F)
    max_len: int
    Returns
    -------
    padded_sequences, masks, Any: Tuple[torch.Tensor, torch.Tensor, Any]
        - if batch_first: Tuple[(B, N_TILES, F), (B, N_TILES, 1), ...]
        - else: Tuple[(N_TILES, B, F), (N_TILES, B, 1), ...]
        with N_TILES = max_len if max_len is not None
        or N_TILES = max length of the training samples.
    """
    # Expect the sequences to be the first one in the sample tuples
    sequences = []
    others = []
    for sample in batch:
        sequences.append(sample[0])
        others.append(sample[1:])

    if max_len is None:
        max_len = max([s.size(0) for s in sequences])

    trailing_dims = sequences[0].size()[1:]

    if batch_first:
        padded_dims = (len(sequences), max_len) + trailing_dims
        masks_dims = (len(sequences), max_len, 1)
    else:
        padded_dims = (max_len, len(sequences)) + trailing_dims
        masks_dims = (max_len, len(sequences), 1)

    padded_sequences = sequences[0].data.new(*padded_dims).fill_(0.0)
    masks = torch.ones(*masks_dims, dtype=torch.bool)

    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            padded_sequences[i, :length, ...] = tensor[:max_len, ...]
            masks[i, :length, ...] = False
        else:
            padded_sequences[:length, i, ...] = tensor[:max_len, ...]
            masks[:length, i, ...] = False

    # Batching other members of the tuple using default_collate
    others = default_collate(others)

    return (padded_sequences, masks, *others)

class CVSplit:

    def __init__(
            self,
            n_splits: int,
            n_repeats: Optional[int] = None,
            stratified: bool = False,
            random_state: int = None,
    ):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.stratified = stratified
        self.random_state = random_state
    
    def _get_fold_generator(self, pl_dataset: pl.LightningDataModule):
        if self.n_repeats == 1:
            fold_generator = StratifiedKFold(n_splits=self.n_splits, random_state=self.random_state)
            if pl_dataset.task == "regression":
                fold_generator = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        else:
            fold_generator = RepeatedStratifiedKFold(
                n_repeats=self.n_repeats, n_splits=self.n_splits, random_state=self.random_state
            )
            if pl_dataset.task == "regression":
                fold_generator = RepeatedKFold(
                    n_repeats=self.n_repeats, n_splits=self.n_splits, random_state=self.random_state
                )
        return fold_generator
    
    def __call__(self, pl_dataset: pl.LightningDataModule, y=None, groups=None):
        fold_generator = self._get_fold_generator(pl_dataset)
        if self.stratified:
            if y is None:
                raise ValueError("if stratified is set to True, y must be provided")

        X = np.arange(len(pl_dataset.dataset_train))
        for idx_train, idx_valid in fold_generator.split(X, y=y, groups=groups):
            dataset_train = torch.utils.data.Subset(pl_dataset.dataset_train, idx_train)
            dataset_valid = torch.utils.data.Subset(pl_dataset.dataset_train, idx_valid)
            yield dataset_train, dataset_valid
