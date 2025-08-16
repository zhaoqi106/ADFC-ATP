import os
import torch
import random
import pickle
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
from rdkit import Chem
from model.ECFP import ECFPModel
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict
from model.encoder_gat import GATNet
from model.FusionModel_ECFP import FusionModel
from sklearn.model_selection import train_test_split
from utils.data_process import Network2TuDataset, read_data
from AttentiveFP import save_smiles_dicts, get_smiles_array
from sklearn.metrics import (roc_auc_score, f1_score, recall_score, accuracy_score,
                             precision_score, matthews_corrcoef, balanced_accuracy_score,
                             precision_recall_curve, auc)


warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def data_prep(task_names, max_atom_num=100):
    all_df_list, feature_dicts, weights = [], {}, []
    for task in task_names:
        label_col = f'label_{task}'
        path = f"./data/data/{task}.csv"
        df = pd.read_csv(path)
        df['task'] = task
        df = df.rename(columns={'label': label_col})

        smiles_raw = df.smiles.values
        canonical_smiles, valid_smiles = [], []
        for smi in smiles_raw:
            mol = Chem.MolFromSmiles(smi)
            if mol and len(mol.GetAtoms()) <= max_atom_num:
                cano = Chem.MolToSmiles(mol, isomericSmiles=True)
                canonical_smiles.append(cano)
                valid_smiles.append(smi)
        df = df[df.smiles.isin(valid_smiles)].copy()
        df['cano_smiles'] = canonical_smiles

        feat_file = path.replace('.csv', '.pickle')
        if os.path.isfile(feat_file):
            feat_dict = pickle.load(open(feat_file, "rb"))
        else:
            feat_dict = save_smiles_dicts(canonical_smiles, path.replace('.csv', ''))
        df = df[df['cano_smiles'].isin(feat_dict['smiles_to_atom_mask'])].copy()

        y = df[label_col]
        n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
        task_weights = [(n_pos + n_neg) / n_neg, (n_pos + n_neg) / n_pos] if n_pos > 0 and n_neg > 0 else [1.0, 1.0]

        feature_dicts[task] = feat_dict
        weights.append(task_weights)
        all_df_list.append(df)

    return pd.concat(all_df_list).reset_index(drop=True), feature_dicts, weights

def prepare_features_per_row(df, feature_dicts):
    atoms, bonds, a_idx, b_idx, masks = [], [], [], [], []
    for _, row in df.iterrows():
        smiles, task = row['cano_smiles'], row['task']
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, _ = get_smiles_array([smiles], feature_dicts[task])
        atoms.append(torch.tensor(x_atom).cpu().numpy())
        bonds.append(torch.tensor(x_bonds).cpu().numpy())
        a_idx.append(torch.tensor(x_atom_index).cpu().numpy())
        b_idx.append(torch.tensor(x_bond_index).cpu().numpy())
        masks.append(torch.tensor(x_mask).cpu().numpy())
    df['atom_list'], df['bond_list'] = atoms, bonds
    df['atom_degree_list'], df['bond_degree_list'], df['atom_mask'] = a_idx, b_idx, masks
    return df

def prepare_multitask_graphs(df_all, task_names, label_name_prefix='label_'):
    graphs_all = []
    df_list = []
    for task in task_names:
        df_task = df_all[df_all['task'] == task].dropna(subset=[f'{label_name_prefix}{task}']).copy()
        smiles = df_task['cano_smiles'].tolist()
        labels = df_task[f'{label_name_prefix}{task}'].astype(int).tolist()
        graphs_nx, _, _ = read_data(smiles, labels)
        graphs = [Network2TuDataset(g, 'cuda', label) for g, label in zip(graphs_nx, labels)]
        graphs_all.extend(graphs)
        df_list.append(df_task)
    return pd.concat(df_list).reset_index(drop=True), graphs_all

def pad_features(graphs, max_dim):
    for g in graphs:
        cur = g['x'].shape[1]
        if cur < max_dim:
            pad = torch.zeros((g['x'].shape[0], max_dim - cur), device=g['x'].device)
            g['x'] = torch.cat([g['x'], pad], dim=1)
    return graphs

def get_loss_function_dict(task_names, weights):
    return {
        task: nn.CrossEntropyLoss(torch.tensor(w).float().to(device))
        for task, w in zip(task_names, weights)
    }

def to_tensor_padded(batch_df, name, is_mask=False, dtype=torch.float32):
    arrays = [torch.tensor(x, dtype=dtype) for x in batch_df[name]]
    if is_mask:
        arrays = [a.squeeze(0) if a.dim() == 2 and a.size(0) == 1 else a for a in arrays]
    max_shape = tuple(max(s) for s in zip(*[a.shape for a in arrays]))
    padded = []
    for a in arrays:
        pad_size = [(0, max_d - a_d) for a_d, max_d in zip(a.shape[::-1], max_shape[::-1])]
        pad_size = [i for tup in pad_size for i in tup][::-1]
        padded.append(F.pad(a, pad_size))
    return torch.stack(padded).to(device)

def train_multitask(model, df, optimizer, loss_fn_dict, graphs, task_names, per_task_output_units_num):
    model.train()
    idx = np.arange(len(df))
    np.random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        batch_idx = idx[i:i + batch_size]
        batch_df = df.iloc[batch_idx]
        batch_graph = [graphs[i] for i in batch_idx]
        if not batch_graph:
            continue

        x = torch.cat([g['x'] for g in batch_graph], dim=0)
        edge_index = torch.cat([g['edge_index'] for g in batch_graph], dim=1)
        edge_weight = torch.cat([g['edge_feat'] for g in batch_graph], dim=0)
        batch_tensor = torch.cat([
            torch.full((g['x'].size(0),), j, dtype=torch.long)
            for j, g in enumerate(batch_graph)
        ]).to(device)

        atom_list = to_tensor_padded(batch_df, 'atom_list').squeeze(1)
        bond_list = to_tensor_padded(batch_df, 'bond_list').squeeze(1)
        atom_mask = to_tensor_padded(batch_df, 'atom_mask', is_mask=True)
        atom_degree_list = to_tensor_padded(batch_df, 'atom_degree_list', dtype=torch.long).squeeze(1)
        bond_degree_list = to_tensor_padded(batch_df, 'bond_degree_list', dtype=torch.long).squeeze(1)
        batch_indices = batch_df.index.tolist()
        output = model(atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask, x, edge_index, edge_weight, batch_tensor,
                       indices=batch_indices)

        total_loss = torch.tensor(0.0, device=device)

        for task_idx, task in enumerate(task_names):
            task_mask = (batch_df['task'] == task).values
            if task_mask.sum() == 0:
                continue

            task_mask_tensor = torch.tensor(task_mask, dtype=torch.bool).to(device)
            y = torch.tensor(batch_df.loc[task_mask, f'label_{task}'].values, dtype=torch.long).to(device)

            y_pred = output[task_mask_tensor][:,
                     task_idx * per_task_output_units_num:(task_idx + 1) * per_task_output_units_num]


            task_loss = loss_fn_dict[task](y_pred, y)
            total_loss = total_loss + task_weight[task] * task_loss

        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()

def eval_multitask(model, df, graphs, loss_fn_dict, task_names, per_task_output_units_num):
    model.eval()
    all_metrics = defaultdict(lambda: {'y_true': [], 'y_score': []})
    task_loss_raw = defaultdict(float)
    task_loss_weighted = defaultdict(float)
    task_sample_count = defaultdict(int)
    results = {}
    all_losses = []

    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i + batch_size]
            batch_graph = [graphs[j] for j in range(i, min(i + batch_size, len(graphs)))]

            if not batch_graph:
                continue

            x = torch.cat([g['x'] for g in batch_graph], dim=0)
            edge_index = torch.cat([g['edge_index'] for g in batch_graph], dim=1)
            edge_weight = torch.cat([g['edge_feat'] for g in batch_graph], dim=0)
            batch_tensor = torch.cat([
                torch.full((g['x'].size(0),), j, dtype=torch.long)
                for j, g in enumerate(batch_graph)
            ]).to(device)

            atom_list = to_tensor_padded(batch_df, 'atom_list').squeeze(1)
            bond_list = to_tensor_padded(batch_df, 'bond_list').squeeze(1)
            atom_mask = to_tensor_padded(batch_df, 'atom_mask', is_mask=True)
            atom_degree_list = to_tensor_padded(batch_df, 'atom_degree_list', dtype=torch.long).squeeze(1)
            bond_degree_list = to_tensor_padded(batch_df, 'bond_degree_list', dtype=torch.long).squeeze(1)
            eval_batch_indices = batch_df.index.tolist()
            output = model(atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask, x, edge_index, edge_weight, batch_tensor,
                           indices=eval_batch_indices)

            for task_idx, task in enumerate(task_names):
                task_mask = (batch_df['task'] == task).values
                task_mask_tensor = torch.tensor(task_mask, dtype=torch.bool).to(output.device)
                if task_mask_tensor.sum() == 0:
                    continue

                y = torch.tensor(batch_df.loc[task_mask, f'label_{task}'].values, dtype=torch.long).to(device)
                y_pred = output[task_mask_tensor][:,
                         task_idx * per_task_output_units_num:(task_idx + 1) * per_task_output_units_num]

                y_prob = F.softmax(y_pred, dim=1)[:, 1].cpu().numpy()
                all_metrics[task]['y_true'].extend(y.cpu().numpy())
                all_metrics[task]['y_score'].extend(y_prob)

                loss = loss_fn_dict[task](y_pred, y).item()
                task_loss_raw[task] += loss * len(y)
                task_loss_weighted[task] += task_weight[task] * loss * len(y)
                task_sample_count[task] += len(y)

    for task in task_names:
        y_true = np.array(all_metrics[task]['y_true'])
        y_score = np.array(all_metrics[task]['y_score'])
        if len(y_true) == 0:
            continue

        y_pred = (y_score > 0.5).astype(int)
        precision, recall, _ = precision_recall_curve(y_true, y_score)

        avg_loss = task_loss_raw[task] / task_sample_count[task]
        avg_loss_w = task_loss_weighted[task] / task_sample_count[task]
        results[task] = {
            'roc_auc': roc_auc_score(y_true, y_score),
            'prc_auc': auc(recall, precision),
            'f1': f1_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'acc': accuracy_score(y_true, y_pred),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'bac': balanced_accuracy_score(y_true, y_pred),
            'loss': avg_loss,
            'loss_w': avg_loss_w
        }

    results['loss'] = sum([results[task]['loss_w'] for task in task_names])

    return results

task_weight = {
    'BS': 1.1,
    'RT': 1.0,
    'FHM': 1.0,
    'SHM': 0.9
}

# ================= 主流程 =================
task_names = ['BS', 'RT', 'FHM', 'SHM']
per_task_output_units_num = 2
output_units_num = len(task_names) * per_task_output_units_num
batch_size = 64
epochs = 1000
p_dropout = 0.2
fingerprint_dim = 128
radius = 3
T = 2
learning_rate = 3
weight_decay = 4
seed = 42
set_seed(seed)

df_all, feature_dicts, weights = data_prep(task_names)
df_temp, df_test = train_test_split(
    df_all, test_size=0.1, stratify=df_all['task'], random_state=seed)
df_train, df_valid = train_test_split(
    df_temp, test_size=0.1111, stratify=df_temp['task'], random_state=seed)

df_train = prepare_features_per_row(df_train, feature_dicts)
df_valid = prepare_features_per_row(df_valid, feature_dicts)
df_test  = prepare_features_per_row(df_test,  feature_dicts)

df_train, graphs = prepare_multitask_graphs(df_train, task_names)
df_valid, graphs_v = prepare_multitask_graphs(df_valid, task_names)
df_test, graphs_t = prepare_multitask_graphs(df_test, task_names)

max_dim = max(g['x'].shape[1] for g in graphs + graphs_v + graphs_t)
graphs = pad_features(graphs, max_dim)
graphs_v = pad_features(graphs_v, max_dim)
graphs_t = pad_features(graphs_t, max_dim)

x_atom, x_bonds, *_ = get_smiles_array([df_train.iloc[0]['cano_smiles']], feature_dicts[df_train.iloc[0]['task']])
smiles_list = df_train['cano_smiles'].tolist()
fingerprint_model = ECFPModel(smiles_list, input_dim=2048, output_dim=128)

pretrained_model = GATNet().to(device)
pretrained_model.load_state_dict(torch.load('results/model/50_model_encoder_gat_100_100_in-vitro_0.3.pkl'))
fusion_model = FusionModel(fingerprint_model, pretrained_model, 128, out_dim=output_units_num).to(device)
optimizer = optim.Adam(fusion_model.parameters(), 10 ** -learning_rate, weight_decay=10 ** -weight_decay)
loss_function_dict = get_loss_function_dict(task_names, weights)
best_loss = float('inf')
patience = 30
counter = 0

for epoch in range(epochs):

    train_multitask(fusion_model, df_train, optimizer, loss_function_dict, graphs, task_names,
                    per_task_output_units_num)

    results = eval_multitask(fusion_model, df_valid, graphs_v, loss_function_dict, task_names,
                             per_task_output_units_num)
    print(f"Epoch {epoch}: val_loss={results['loss']:.3f}")

    for task in task_names:
        if task in results:
            print(
                f"[{task}] "
                f"AUC={results[task]['roc_auc']:.3f} "
                f"ACC={results[task]['acc']:.3f} "
                f"PRE={results[task]['precision']:.3f} "
                f"RE={results[task]['recall']:.3f} "
                f"LOSS={results[task]['loss_w']:.3f}"
            )

    if results['loss'] < best_loss:
        best_loss = results['loss']
        counter = 0
        torch.save(fusion_model.state_dict(), "saved_models/best_model.pt")
    else:
        counter += 1

    if counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

fusion_model.load_state_dict(torch.load("saved_models/best_model.pt"))
test_result = eval_multitask(fusion_model, df_test, graphs_t, loss_function_dict, task_names, per_task_output_units_num)

print("\n[Test Results]")
for task in task_names:
    if task in test_result:
        print(
            f"[{task}]"
            f"AUC={test_result[task]['roc_auc']:.3f},"
            f"ACC={test_result[task]['acc']:.3f},"
            f"PRE={test_result[task]['precision']:.3f},"
            f"RE={test_result[task]['recall']:.3f},"
            f"LOSS={test_result[task]['loss_w']:.3f}"
        )
