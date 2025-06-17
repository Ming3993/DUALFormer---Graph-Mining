import random
import numpy as np
import torch
import os
from torch_geometric.datasets import Planetoid, WebKB, Actor, Amazon, WikiCS, WikipediaNetwork, Coauthor
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj, dense_to_sparse, to_torch_coo_tensor

root = os.path.split(__file__)[0]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def DataLoader(name):
    name = name.lower()
    root_path = 'datasets/'
    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root_path, name, split='random', num_train_per_class=20, num_val=500, num_test=1000,
                            transform=T.NormalizeFeatures())
    elif name in ['computers', 'photo']:
        dataset = Amazon(root_path, name, T.NormalizeFeatures())
    elif name in ['cs', 'physics']:
        dataset = Coauthor(root_path, name, T.NormalizeFeatures())
    elif name in ['chameleon', 'squirrel']:
        # use everything from "geom_gcn_preprocess=False" and
        # only the node label y from "geom_gcn_preprocess=True"
        preProcDs = WikipediaNetwork(
            root=root_path, name=name, geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
        dataset = WikipediaNetwork(
            root=root_path, name=name, geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.edge_index = preProcDs[0].edge_index
        dataset.data = data
        return dataset

    elif name in ['film']:
        dataset = Actor(root=root_path+'/Actor', transform=T.NormalizeFeatures())
        dataset.name=name
    elif name in ['texas', 'cornell', 'wisconsin']:
        dataset = WebKB(root=root_path, name=name, transform=T.NormalizeFeatures())
    elif name in ['wikics']:
        dataset = WikiCS(root=root_path+'/WikiCS', transform=T.NormalizeFeatures())
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')
    return dataset

def adj_to_edgeIndex_edgeWeight(adj):
    edge_index, edge_weight = dense_to_sparse(adj)
    return edge_index, edge_weight

def adj_to_edgeIndex(adj):
    edge_index = adj.nonzero().t().contiguous()
    return edge_index

def dataset_split(data, run_id):
    if data.name in ['wikics', 'computers', 'photo', 'physics']:
        split = get_split(num_samples=data.num_nodes, train_ratio=0.1, test_ratio=0.8)
    elif data.name in ['cora', 'citeseer', 'pubmed']:
        split = get_public_split(data)

    return split

def get_public_split(data):
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    indices = torch.arange(0, data.num_nodes).to(train_mask.device)
    return {
        'train': indices[train_mask],
        'valid': indices[val_mask],
        'test': indices[test_mask]
    }

def get_split(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.1):
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    indices = torch.randperm(num_samples)
    return {
        'train': indices[:train_size],
        'test': indices[train_size: test_size + train_size],
        'valid': indices[test_size + train_size:]
    }
