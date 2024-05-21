import os
import torch
import random
import numpy as np
import yaml
import torch_geometric.transforms as T
import os.path as osp
import os
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, WikiCS, NELL, WebKB, CoraFull, WikipediaNetwork
import time
from sklearn.preprocessing import StandardScaler

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def save_best_model(args, model, optimizer, node_test, unique_id, cur_best, cur_best_txt):
    try:
        os.remove(cur_best)
        os.remove(cur_best_txt)
    except:
        pass
    best_time = str(time.strftime("%Y%m%d-%H%M%S"))
    filename = './pretrained/bestresults/' + str(args.dataset) + '_' + str(args.shot) + '_' + str(
        node_test) + '_' + unique_id + '_' + best_time + args.save_path
    filename_txt = './pretrained/bestresults/' + str(
        args.dataset) + '_' + str(args.shot)+ '_' + str(node_test) + '_' + unique_id + '_' + best_time + '.txt'

    with open(f'{filename_txt}', 'w') as file:
        for arg, value in vars(args).items():
            file.write(f'{arg}: {value}\n')
    # Save model and optimizer state
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'previous_unconflicted': model.previous_unconflicted,
        'cluster_pred': model.cluster_pred,
    }, filename)
    return filename, filename_txt

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    configs = configs[args.dataset.lower()]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args


def set_seed(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True) # 아마 해보자

def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats

def load_dataset(dataset_name, device):
    transform = T.Compose([T.ToUndirected(), T.ToDevice(device)])
    root = osp.join('~/public_data/pyg_data')
    if dataset_name in {'Arxiv'}:
        from ogb.nodeproppred import PygNodePropPredDataset
        print('loading ogb dataset...')
        dataset = PygNodePropPredDataset(root=root, name=f'ogbn-arxiv')
        data = transform(dataset[0])
        split_idx = dataset.get_idx_split()
        data.train_mask = split_idx['train']
        data.val_mask = split_idx['valid']
        data.test_mask = split_idx['test']

    elif dataset_name in {'Cora', 'Citeseer', 'Pubmed'}:
        dataset = Planetoid(root, dataset_name, transform=T.NormalizeFeatures())
        data = transform(dataset[0])
    elif dataset_name in {'Photo', 'Computers'}:
        dataset = Amazon(root, dataset_name)
        data = transform(dataset[0])
        data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
    elif dataset_name in {'CS', 'Physics'}:
        dataset = Coauthor(root, dataset_name)
        data = transform(dataset[0])
        data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
    elif dataset_name in {'WikiCS'}:
        dataset = WikiCS(root=root)
        data = transform(dataset[0])
        data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
    elif dataset_name in {'NELL'}:
        dataset = NELL(root=root)
        data = transform(dataset[0])
        data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
        data.x=data.x.to_dense()
    elif dataset_name in {'CoraFull'}:
        dataset = CoraFull(root="/home/jongwon208/MaskGAE/mine_encoder_list/data", transform=T.NormalizeFeatures())
        data = transform(dataset[0])
        data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data)
        # dataset = CitationFull(root="/home/jongwon208/MaskGAE/mine_encoder_list/data",name= "cora", transform=T.NormalizeFeatures())
        # data = transform(dataset[0])
    elif dataset_name in {'Chameleon'}:
        print('chameleon dataaset load')
        dataset = WikipediaNetwork(root, dataset_name, transform=T.NormalizeFeatures())
        data = transform(dataset[0])
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]
    elif dataset_name in {'Texas','Cornell','Wisconsin'}:
        print('texas')
        dataset = WebKB(root, dataset_name, transform=T.NormalizeFeatures())
        data = transform(dataset[0])
        data.train_mask = data.train_mask[:,0]
        data.val_mask = data.val_mask[:,0]
        data.test_mask = data.test_mask[:,0]

    else:
        raise ValueError(dataset_name)
        print('check dataset name')
    return data

