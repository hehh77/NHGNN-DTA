from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data as DATA
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import os

from torch.utils.data import DataLoader, Dataset

import re

class Pre_DTADataset(torch.utils.data.Dataset):
    def __init__(self,path,drug_vocab,target_vocab,smiles_emb , target_emb,smiles_len , target_len,seq_len = 220,tar_len = 1000):
        self.tar_len = tar_len
        self.target_vocab = target_vocab
        self.seq_len = seq_len
        self.drug_vocab = drug_vocab
        self.smiles = []
        self.targets = []
        self.label = []
        self.sm_len = []
        self.tar_len = []
        
        
        df = pd.read_csv(path)
        for i in tqdm(range(len(df))):
            sm = df.loc[i,'compound_iso_smiles']
            target = df.loc[i,'target_key']
            seq = df.loc[i,'target_sequence']
            label = df.loc[i,'affinity']
            sm_emb = smiles_emb[sm]
            tar_emb = target_emb[seq]
            sm_len = smiles_len[sm]
            tar_len = target_len[seq]
            self.smiles.append(sm_emb)
            self.targets.append(tar_emb)
            self.label.append(label)
            self.sm_len.append(sm_len)
            self.tar_len.append(tar_len)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, item):
        return [self.smiles[item],self.targets[item],self.label[item],self.sm_len[item] , self.tar_len[item]]



class DTADataset(InMemoryDataset):
    def __init__(self, root, path, smiles_emb, target_emb, smiles_idx, smiles_graph, target_graph, smiles_len, target_len):

        super(DTADataset, self).__init__(root)
        self.path = path
        df = pd.read_csv(path)
        self.data = []
        self.process(df, smiles_emb, target_emb, smiles_idx, smiles_graph, target_graph, smiles_len, target_len)



    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['process.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, df, smiles_emb, target_emb, smiles_idx, smiles_graph, target_graph, smiles_len, target_len):
        # data_list = []
        for i in tqdm(range(len(df))):
            sm = df.loc[i, 'compound_iso_smiles']
            target = df.loc[i, 'target_key']
            seq = df.loc[i, 'target_sequence']
            label = df.loc[i, 'affinity']
            sm_g = smiles_graph[sm]
            ta_g = target_graph[seq]
            sm_idx = smiles_idx[sm]
            tar_len = target_len[seq]

            s_off = self.off_adj(sm_g, tar_len - 2)
            com_adj = np.concatenate((ta_g, s_off), axis=0)
            total_len = tar_len - 2 + len(sm_idx)
            tem1 = np.zeros([total_len, 2])
            tem2 = np.zeros([total_len, 2])
            for i in range(total_len):
                tem1[i, 0] = total_len
                tem1[i, 1] = i
                tem2[i, 1] = total_len
                tem2[i, 0] = i
            tem1 = np.int64(tem1)
            tem2 = np.int64(tem2)
            com_adj = np.concatenate((com_adj, tem1), axis=0)
            com_adj = np.concatenate((com_adj, tem2), axis=0)
            com_adj = np.concatenate((com_adj, [[total_len, total_len]]), axis=0)


            Data = DATA(y=torch.FloatTensor([label]),
                        edge_index=torch.LongTensor(com_adj).transpose(1, 0),
                        sm=sm,
                        target=target,
                        # seq = seq
                        )
            self.data.append(Data)
        if self.pre_filter is not None:
            self.data = [data for data in self.data if self.pre_filter(self.data)]

        if self.pre_transform is not None:
            self.data = [self.pre_transform(data) for data in self.data]


    def off_adj(self, adj, size):
        adj1 = adj.copy()
        for i in range(adj1.shape[0]):
            adj1[i][0] += size
            adj1[i][1] += size
        return adj1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]