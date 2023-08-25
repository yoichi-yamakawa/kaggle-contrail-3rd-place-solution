import copy
import math
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import Sampler


class HappyWhaleSamplerV1(DistributedSampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - df (pd.DataFrame): train_df
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, dataset, shuffle, drop_last=True):
        super().__init__(dataset, shuffle=shuffle, drop_last=drop_last)
        assert drop_last == True
        self.df = dataset.df
        self.config = self.dataset.config
        self.batch_size = self.config.train_batch_size
        self.num_instances = self.config.sampler_config.num_instances  # for each individual id
        self.num_iids_per_batch = self.batch_size // self.num_instances
        self.index_dic = self.get_indices_for_iids()
        self.iids = list(self.index_dic.keys())

        self.final_idx = self.get_sampler_indices()
        self.len_final_idxs = len(self.final_idx)
        self.n_pairs = self.len_final_idxs // self.num_instances

        self.num_samples = self.n_pairs // self.num_replicas * self.num_instances
        self.total_size = self.num_samples * self.num_replicas

    def get_indices_for_iids(self):
        index_dic = defaultdict(list)
        for index, iid in enumerate(self.df.individual_id.values):
            index_dic[iid].append(index)

        return index_dic

    def get_sampler_indices(self):
        batch_idxs_dict = defaultdict(list)

        for iid in self.iids:
            idxs = copy.deepcopy(self.index_dic[iid])

            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)

            elif len(idxs) % self.num_instances != 0:
                num_add = self.num_instances - len(idxs) % self.num_instances
                add_idxs = np.random.choice(idxs, size=num_add, replace=True)
                idxs.extend(add_idxs)

            random.shuffle(idxs)
            batch_idxs = []

            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[iid].append(batch_idxs)
                    batch_idxs = []

        avai_iids = copy.deepcopy(self.iids)
        final_idxs = []

        while len(avai_iids) >= self.num_iids_per_batch:
            selected_iids = random.sample(avai_iids, self.num_iids_per_batch)
            for iid in selected_iids:
                batch_idxs = batch_idxs_dict[iid].pop(0)
                final_idxs.extend(batch_idxs)

                if len(batch_idxs_dict[iid]) == 0:
                    avai_iids.remove(iid)

        return final_idxs

    def __iter__(self):
        indices = self.final_idx[: self.total_size]
        instance_indices = list(range(self.total_size // self.num_instances))
        instance_indices = instance_indices[self.rank : len(instance_indices) : self.num_replicas]  # [0, 4, 8, ....]
        select_indices = []

        for k in instance_indices:
            select_indices.extend(indices[k * self.num_instances : (k + 1) * self.num_instances])
        #         pd.to_pickle(select_indices, f"/mnt/vol21/yo1mtrv/kaggle/p20-happywhale/notebook/{self.rank}_indices.pkl")

        return iter(select_indices)

    def __len__(self):
        return self.num_samples
