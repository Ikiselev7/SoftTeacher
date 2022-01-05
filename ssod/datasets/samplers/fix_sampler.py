from __future__ import division

import copy
import itertools
import math

import numpy as np
import torch
from mmcv.runner import get_dist_info
from mmdet.datasets import ConcatDataset
from torch.utils.data import Sampler, WeightedRandomSampler

from ssod.utils import get_root_logger

from ..builder import SAMPLERS


def chain(iter_obj):
    tmp = []
    for s in iter_obj:
        tmp.extend(s)
    return tmp


@SAMPLERS.register_module()
class DistributedGroupFixRatioSampler(Sampler):
    def __init__(
            self,
            dataset: ConcatDataset,
            samples_per_gpu=1,
            num_replicas=None,
            rank=None,
            sample_ratio=None,
            by_prob=True,
            at_least_one=False,
            seed=0,
            max_iters=None,
    ):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        assert isinstance(
            dataset, ConcatDataset
        ), "The dataset must contains multiple sub datasets"
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        if at_least_one:
            assert self.samples_per_gpu >= len(self.dataset.datasets)
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed if seed is not None else 0

        self.sample_ratio = (
            sample_ratio
            if sample_ratio is not None
            else [1] * len(self.dataset.datasets)
        )
        assert len(self.sample_ratio) == len(self.dataset.datasets)
        self.by_prob = by_prob
        self.at_least_one = at_least_one
        self.base_indices = [
            self._get_sub_seq(
                d, offset=self.dataset.cumulative_sizes[i - 1] if i > 0 else 0
            )
            for i, d in enumerate(self.dataset.datasets)
        ]
        self.set_num = len(self.base_indices)

        group_num_per_set = [len(self.base_indices[i]) for i in range(self.set_num)]
        if not all([num == max(group_num_per_set) for num in group_num_per_set]):
            self.logger.warn(
                "The number of groups in each set is not same. Ignoring the group flag...."
            )
            self.base_indices = [
                np.concatenate(indices) for indices in self.base_indices
            ]
            self.group_num = 1
        else:
            self.group_num = len(self.base_indices[0])
        self.max_iters = max_iters
        self._compute_samples()

    def __iter__(self):
        self._compute_samples()
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        indices = copy.deepcopy(self.base_indices)
        cumulated_indices = []

        for i in range(self.group_num):

            _indices = [
                itertools.cycle(
                    indice[i][
                        list(torch.randperm(len(indice[i]), generator=g).numpy())
                    ].tolist()
                )
                for indice in indices
            ]

            for _ in range(self.iter_per_group[i]):
                size_per_bucket = self._sample(g)

                samples_per_batch = [
                    [next(_indices[j]) for _ in range(size_per_bucket[j])]
                    for j in range(self.set_num)
                ]
                # print(samples_per_batch)
                # shuffle across process
                if self.at_least_one:
                    for s in samples_per_batch:
                        assert (
                                len(s) >= self.num_replicas
                        ), "When `at_least_one` set to `True`, size of each set must be larger than world_size."
                    base = chain(
                        [
                            np.asarray(s[: self.num_replicas])[
                                list(
                                    torch.randperm(
                                        self.num_replicas, generator=g
                                    ).numpy()
                                )
                            ].tolist()
                            for s in samples_per_batch
                        ]
                    )

                    extra = np.array(
                        chain(
                            [
                                s[self.num_replicas :]
                                for s in samples_per_batch
                                if len(s) > self.num_replicas
                            ]
                        )
                    )
                    if len(extra) > 0:
                        extra = extra[
                            list(torch.randperm(len(extra), generator=g,).numpy())
                        ]
                    extra = extra.tolist()
                    samples_per_batch = base + extra
                else:
                    samples_per_batch = np.array(chain(samples_per_batch))[
                        list(
                            torch.randperm(
                                self.samples_per_gpu * self.num_replicas, generator=g
                            ).numpy()
                        )
                    ].tolist()
                cumulated_indices.append(samples_per_batch)
        cumulated_indices = (
            np.asarray(cumulated_indices)[
                list(torch.randperm(len(cumulated_indices), generator=g).numpy())
            ]
                .reshape((-1,))
                .tolist()
        )
        assert len(cumulated_indices) == len(self) * self.num_replicas
        # subsample
        cumulated_indices = cumulated_indices[
                            self.rank : self.rank + self.num_replicas * len(self) : self.num_replicas
                            ]
        assert len(cumulated_indices) == len(self)
        return iter(cumulated_indices)

    def _sample(self, generator=None):
        total_batch_size = self.num_replicas * self.samples_per_gpu
        # normalize
        sample_prob = [s / sum(self.sample_ratio) for s in self.sample_ratio]
        if (not self.by_prob) or (generator is None):
            size_per_bucket = [int(total_batch_size * p) for p in sample_prob]
            size_per_bucket[-1] = total_batch_size - sum(size_per_bucket[:-1])
        else:
            if self.at_least_one:
                extra_size = total_batch_size - self.num_replicas * self.set_num
                if extra_size > 0:
                    sample_seq = list(
                        WeightedRandomSampler(
                            sample_prob,
                            extra_size,
                            replacement=True,
                            generator=generator,
                        )
                    )
                else:
                    sample_seq = []
                for i in range(self.set_num):
                    sample_seq = sample_seq + [i for _ in range(self.num_replicas)]
                _, size_per_bucket = np.unique(sample_seq, return_counts=True)
            else:
                sample_seq = list(
                    WeightedRandomSampler(
                        sample_prob,
                        total_batch_size,
                        replacement=True,
                        generator=generator,
                    )
                )
                _, size_per_bucket = np.unique(sample_seq, return_counts=True)

        return size_per_bucket

    def _get_sub_seq(self, dataset, offset=0):
        flag = dataset.flag
        group_sizes = np.bincount(flag)
        indices = []
        for i, size in enumerate(group_sizes):
            if size > 0:
                indice = np.where(flag == i)[0]
                indice = indice + offset
                assert len(indice) == size
                indices.append(indice)
        return indices

    def __len__(self):
        return self.max_iters * self.samples_per_gpu

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _compute_samples(self):
        # estimate the length of epoch
        size_per_set = self._sample()
        iter_per_set = [
            [math.ceil(len(indice) / size) if size > 0 else 0 for indice in indices]
            for indices, size in zip(self.base_indices, size_per_set)
        ]
        self.iter_per_group = [max(iters) for iters in zip(*iter_per_set)]
        base_iters = sum(self.iter_per_group)
        if self.max_iters is None:
            self.max_iters = base_iters
        else:
            self.iter_per_group = [
                int(i * self.max_iters / base_iters) for i in self.iter_per_group
            ]
            self.iter_per_group[-1] = self.max_iters - sum(self.iter_per_group[:-1])

        self.logger = get_root_logger()
        self.logger.info(
            f"Sampling {len(self.base_indices)} datasets by ratio {self.sample_ratio}..."
        )
        self.logger.info(
            "Size for each dataset: {}".format(
                " : ".join(
                    [
                        f"{len(self.dataset.datasets[i])}"
                        for i in range(len(self.dataset.datasets))
                    ]
                )
            )
        )
        for i in range(len(self.base_indices)):
            self.logger.info(
                "Split dataset {} by flag. Group Sizes: {}".format(
                    i, [len(x) for x in self.base_indices[i]]
                )
            )
        self.logger.info("Iterations in one epoch : {}".format(self.max_iters))
        # split max_iters for each group
        self.logger.info(
            "Iterations in one epoch for each group: {}".format(self.iter_per_group)
        )
