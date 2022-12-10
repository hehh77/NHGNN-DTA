
import numpy as np
import rdkit
import rdkit.Chem as Chem
import networkx as nx
from build_vocab import WordVocab
import pandas as pd

import os
import torch.nn as nn
from dataset import DTADataset
from model import *
from utils import *

from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch



#############################



class NHGNN(nn.Module):
    def __init__(self, embedding_dim, lstm_dim, hidden_dim, dropout_rate,
                 alpha, n_heads, bilstm_layers=2, protein_vocab=26,
                 smile_vocab=45, theta=0.5):
        super(NHGNN, self).__init__()
        self.is_bidirectional = True
        # drugs
        self.theta = theta
        self.dropout = nn.Dropout(dropout_rate)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.bilstm_layers = bilstm_layers
        self.n_heads = n_heads
        # self.gat = GATNet()
        self.hgin = GINConvNet()

    
        # SMILES
        self.smiles_vocab = smile_vocab
        self.smiles_embed = nn.Embedding(smile_vocab + 1, 256, padding_idx=0)

        self.is_bidirectional = True
        self.smiles_input_fc = nn.Linear(256, lstm_dim)
        self.smiles_lstm = nn.LSTM(lstm_dim, lstm_dim, self.bilstm_layers, batch_first=True,
                                  bidirectional=self.is_bidirectional, dropout=dropout_rate)
        self.ln1 = torch.nn.LayerNorm(lstm_dim * 2)
        self.out_attentions3 = LinkAttention(hidden_dim, n_heads)

        # protein
        self.protein_vocab = protein_vocab
        self.protein_embed = nn.Embedding(protein_vocab + 1, embedding_dim, padding_idx=0)
        self.is_bidirectional = True
        self.protein_input_fc = nn.Linear(embedding_dim, lstm_dim)
   
        self.protein_lstm = nn.LSTM(lstm_dim, lstm_dim, self.bilstm_layers, batch_first=True,
                                  bidirectional=self.is_bidirectional, dropout=dropout_rate)
        self.ln2 = torch.nn.LayerNorm(lstm_dim * 2)
        self.protein_head_fc = nn.Linear(lstm_dim * n_heads, lstm_dim)
        self.protein_out_fc = nn.Linear(2 * lstm_dim, hidden_dim)
        self.out_attentions2 = LinkAttention(hidden_dim, n_heads)

        # link
        self.out_attentions = LinkAttention(hidden_dim, n_heads)
        self.out_fc1 = nn.Linear(hidden_dim * 3, 256 * 8)
        self.out_fc2 = nn.Linear(256 * 8, hidden_dim * 2)
        self.out_fc3 = nn.Linear(hidden_dim * 2, 1)
        self.layer_norm = nn.LayerNorm(lstm_dim * 2)

   
    def forward(self, data, reset=False):
        batchsize = len(data.sm)
        smiles = torch.zeros(batchsize, seq_len).cuda().long()
        protein = torch.zeros(batchsize, tar_len).cuda().long()
        smiles_lengths = []
        protein_lengths = []

        for i in range(batchsize):
            sm = data.sm[i]
            seq_id = data.target[i]
            seq = target_seq[seq_id]
            smiles[i] = smiles_emb[sm]
            protein[i] = target_emb[seq]
            smiles_lengths.append(smiles_len[sm])
            protein_lengths.append(target_len[seq])

        smiles = self.smiles_embed(smiles)  # B * seq len * emb_dim


        smiles = self.smiles_input_fc(smiles)  # B * seq len * lstm_dim
        smiles, _ = self.smiles_lstm(smiles)  # B * seq len * lstm_dim*2
        smiles = self.ln1(smiles)
        # del smiles

        protein = self.protein_embed(protein)  # B * tar_len * emb_dim

        protein = self.protein_input_fc(protein)  # B * tar_len * lstm_dim

        # del protein

        protein, _ = self.protein_lstm(protein)  # B * tar_len * lstm_dim *2
        protein = self.ln2(protein)
        if reset:
            return smiles, protein


        smiles_mask = self.generate_masks(smiles, smiles_lengths, self.n_heads)  # B * head* seq len
        smiles_out, smile_attn = self.out_attentions3(smiles, smiles_mask)  # B * lstm_dim*2


        protein_mask = self.generate_masks(protein, protein_lengths, self.n_heads)  # B * head * tar_len
        protein_out, prot_attn = self.out_attentions2(protein, protein_mask)  # B * (lstm_dim *2)

        # drugs and proteins
        out_cat = torch.cat((smiles, protein), dim=1)  # B * head * lstm_dim *2
        out_masks = torch.cat((smiles_mask, protein_mask), dim=2)  # B * tar_len+seq_len * (lstm_dim *2)
        out_cat, out_attn = self.out_attentions(out_cat, out_masks)
        out = torch.cat([smiles_out, protein_out, out_cat], dim=-1)  # B * (rnn*2 *3)
        out = self.dropout(self.relu(self.out_fc1(out)))  # B * (256*8)
        out = self.dropout(self.relu(self.out_fc2(out)))  # B *  hidden_dim*2
        out = self.out_fc3(out).squeeze()

        del smiles_out, protein_out


        gout = self.hgin(data)

        return gout * self.theta + out.view(-1, 1) * (1-self.theta)

    def generate_masks(self, adj, adj_sizes, n_heads):
        out = torch.ones(adj.shape[0], adj.shape[1])
        max_size = adj.shape[1]
        if isinstance(adj_sizes, int):
            out[0, adj_sizes:max_size] = 0
        else:
            for e_id, drug_len in enumerate(adj_sizes):
                out[e_id, drug_len: max_size] = 0
        out = out.unsqueeze(1).expand(-1, n_heads, -1)
        return out.cuda(device=adj.device)



#############################



#############################
CUDA = '0'
dataset_name = 'davis'
seed = 0
reset_epoch = 40
drug_vocab = WordVocab.load_vocab('../Vocab/smiles_vocab.pkl')
target_vocab = WordVocab.load_vocab('../Vocab/protein_vocab.pkl')

tar_len = 2600
seq_len = 536
load_model_path = None



LR = 1e-4
GIN_lr = 1e-3
NUM_EPOCHS = 400


model_file_name = None


embedding_dim = 128
lstm_dim = 64
hidden_dim = 128
dropout_rate = 0.1
alpha = 0.2
n_heads = 8


#############################


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #（保证程序cuda序号与实际cuda序号对应）
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA


device = torch.device('cuda:0')
seed_torch(seed)





def target_to_graph(target_key, target_sequence, contact_dir):
    target_edge_index = []
    target_size = len(target_sequence)
    contact_file = os.path.join(contact_dir, target_key + '.npy')
    contact_map = np.load(contact_file)
    contact_map += np.matrix(np.eye(contact_map.shape[0]))
    index_row, index_col = np.where(contact_map > 0.8)
    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])
    target_edge_index = np.array(target_edge_index)
    return target_size, target_edge_index




def smiles_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
        # edge_index.append([e1, e2])
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    edge_index = np.array(edge_index)
    return c_size, edge_index



df = pd.read_csv('../Data/'+ dataset_name+ '.csv')
smiles = set(df['compound_iso_smiles'])
target = set(df['target_key'])

target_seq = {}
for i in range(len(df)):
    target_seq[df.loc[i, 'target_key']] = df.loc[i, 'target_sequence']
target_graph = {}
for k in target_seq:
    seq = target_seq[k]
    _, graph = target_to_graph(k, seq, '../Data/'+dataset_name+'/')
    target_graph[seq] = graph
smiles_graph = {}
for sm in smiles:
    _, graph = smiles_to_graph(sm)
    smiles_graph[sm] = graph



target_emb = {}
target_len = {}
for k in target_seq:
    seq = target_seq[k]
    content = []
    flag = 0
    for i in range(len(seq)):
        if flag >= len(seq):
            break
        if (flag + 1 < len(seq)):
            if target_vocab.stoi.__contains__(seq[flag:flag + 2]):
                content.append(target_vocab.stoi.get(seq[flag:flag + 2]))
                flag = flag + 2
                continue
        content.append(target_vocab.stoi.get(seq[flag], target_vocab.unk_index))
        flag = flag + 1

    if len(content) > tar_len:
        content = content[:tar_len]

    X = [target_vocab.sos_index] + content + [target_vocab.eos_index]
    target_len[seq] = len(content)
    if tar_len > len(X):
        padding = [target_vocab.pad_index] * (tar_len - len(X))
        X.extend(padding)
    target_emb[seq] = torch.tensor(X)


smiles_idx = {}
smiles_emb = {}
smiles_len = {}

for sm in smiles:
    content = []
    flag = 0
    for i in range(len(sm)):
        if flag >= len(sm):
            break
        if (flag + 1 < len(sm)):
            if drug_vocab.stoi.__contains__(sm[flag:flag + 2]):
                content.append(drug_vocab.stoi.get(sm[flag:flag + 2]))
                flag = flag + 2
                continue
        content.append(drug_vocab.stoi.get(sm[flag], drug_vocab.unk_index))
        flag = flag + 1

    if len(content) > seq_len:
        content = content[:seq_len]

    X = [drug_vocab.sos_index] + content + [drug_vocab.eos_index]
    smiles_len[sm] = len(content)
    if seq_len > len(X):
        padding = [drug_vocab.pad_index] * (seq_len - len(X))
        X.extend(padding)

    smiles_emb[sm] = torch.tensor(X)

    if not smiles_idx.__contains__(sm):
        tem = []
        for i, c in enumerate(X):
            if atom_dict.__contains__(c):
                tem.append(i)
        smiles_idx[sm] = tem

print('creat dataset...')
dataset = DTADataset(root = '../Data/'+dataset_name,path='../Data/'+dataset_name+'.csv',smiles_emb=smiles_emb , target_emb = target_emb,smiles_idx=smiles_idx,smiles_graph=smiles_graph , target_graph = target_graph,
smiles_len=smiles_len , target_len= target_len)

print('built model...')
model = NHGNN(embedding_dim = embedding_dim, lstm_dim = lstm_dim, hidden_dim = hidden_dim, dropout_rate = dropout_rate, alpha = alpha, n_heads = n_heads).to(device)

if load_model_path is not None:
        print('load_model...',load_model_path)
        save_model = torch.load(load_model_path)
        model_dict = model.state_dict()
        state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

def reset_feature(dataset , model):
    torch.cuda.empty_cache()
    batch_size = 128
    with torch.no_grad():
        model.eval()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        start = 0
        for data in tqdm(dataloader):
            sm , pro = model(data,reset = True)
            tar_len = []
            idx = []
            for i in range(min(batch_size,len(dataset) - start)):
                sm_id = dataset[start+i].sm
                pro_id = dataset[start+i].target
                pro_id = target_seq[pro_id]
                tar_len.append(target_len[pro_id])
                idx.append(smiles_idx[sm_id])
            for i in range(start , min(len(dataset) , start + batch_size)):
                dataset.data[i].x = torch.cat([pro[i-start,1:tar_len[i-start]-1] , sm[i-start,idx[i-start]] , sm[i-start,0].unsqueeze(0)]).cpu()
            start = start+batch_size

GIN_params = list(map(id, model.hgin.parameters()))
base_params = filter(lambda p: id(p) not in GIN_params,
                     model.parameters())

optimizer = torch.optim.Adam(
            [{'params': base_params},
            {'params': model.hgin.parameters(), 'lr': GIN_lr}], lr=LR)

loss_fn = nn.MSELoss()

schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=2e-4, last_epoch=-1)

best_mse = 1000
best_test_mse = 1000
best_epoch = -1
best_test_epoch = -1

torch.cuda.empty_cache()
for epoch in range(NUM_EPOCHS):
    print("No {} epoch".format(epoch))
    if epoch % reset_epoch == 0:
        print('update nodes features...')
        reset_feature(dataset , model)
        batch_size = 128
        test_size = (int)(len(dataset)*0.1)

        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset=dataset,
            lengths=[len(dataset)-(test_size*2) , test_size*2],
            generator=torch.Generator().manual_seed(0)
        )
        val_dataset ,test_dataset = torch.utils.data.random_split(
            dataset=test_dataset,
            lengths=[test_size,test_size],
            generator=torch.Generator().manual_seed(0)
        )


        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print('Train size:', len(train_loader))
        print('Test size:', len(val_loader))
        print('Test size:', len(test_loader))

    train(model , train_loader,optimizer,epoch)
    G, P = predicting(model, val_loader)
    val1 = get_mse(G, P)
    if val1 < best_mse:
        best_mse = val1
        best_epoch = epoch + 1
        if model_file_name is not None:
            torch.save(model.state_dict(), model_file_name)
        print('mse improved at epoch ', best_epoch, '; best_mse', best_mse)
    else:
        print('current mse: ',val1 ,  ' No improvement since epoch ', best_epoch, '; best_mse', best_mse)
    schedule.step()

