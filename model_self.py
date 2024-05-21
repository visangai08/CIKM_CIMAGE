import torch
import torch.nn.functional as F
import torch.nn as nn
import random
from torch_geometric.nn import Linear
import numpy as np
from torch_geometric.utils import add_self_loops, negative_sampling
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from loss import setup_loss_fn, ce_loss
from HSICLassoVI_torch.models import api
from utils import seed_worker

def create_activation_layer(activation):
    if activation is None:
        return nn.Identity()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    else:
        raise ValueError("Unknown activation")

class MLPEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(MLPEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        bn = nn.BatchNorm1d
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(Linear(first_channels, second_channels))
            self.bns.append(bn(second_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            if not isinstance(bn, nn.Identity):
                bn.reset_parameters()

    def forward(self, args, epoch, x, remaining_edges, train_neighbors, cluster_pred, train_mask, train_B):
        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x)
            x = self.bns[i](x)
            # x = self.activation(x)
        x = self.dropout(x)
        x = self.convs[-1](x)
        x = self.bns[-1](x)
        disen_embedding, score_list = self.disentangle(args, epoch, x, remaining_edges, train_neighbors, cluster_pred, train_mask, train_B, train_=True)
        return disen_embedding, score_list

    def gcn_agg(self, adj, X):
        adjacency_matrix = adj.coalesce()
        adj_indices = adjacency_matrix.indices()
        adj_values = adjacency_matrix.values()
        num_nodes = X.shape[0]
        # add self-loop
        self_edge_indices = torch.arange(num_nodes).unsqueeze(0).repeat(2, 1).to(X.device)
        self_edge_values = torch.ones(num_nodes).to(X.device)
        adj_indices = torch.cat([adj_indices, self_edge_indices], dim=1)
        adj_values = torch.cat([adj_values, self_edge_values])
        adjacency_matrix = torch.sparse_coo_tensor(adj_indices, adj_values, (X.shape[0],X.shape[0]))
        # calculate D
        adjacency_matrix = adjacency_matrix.coalesce()
        row_sum_inv = torch.sqrt(torch.sparse.sum(adjacency_matrix, dim=0).values()).pow(-1)
        row_sum_inv_diag = torch.sparse_coo_tensor(self_edge_indices, row_sum_inv, (X.shape[0],X.shape[0]))
        normalized_adjacency_matrix = row_sum_inv_diag @ adjacency_matrix @ row_sum_inv_diag
        return torch.sparse.mm(normalized_adjacency_matrix, X)

    def disentangle(self, args, epoch, z, remaining_edges, remaining_neighbors, cluster_pred, prev_unconflicted, block, train_=True):  # z shape 2708,256
        remain_data_adj_sp = torch.sparse_coo_tensor(remaining_edges, torch.ones(len(remaining_edges[0])).to(z.device),
                                                     [z.shape[0], z.shape[0]])
        ch_dim = z.shape[1] // args.ncaps
        x = routing_layer_32(z, args.ncaps, args.nlayer, args.max_iter, remaining_neighbors) # torch.Size([2708, 512])
        cluster_pred_use = cluster_pred[prev_unconflicted] # ([939])
        X_reshaped = x.view(-1, args.ncaps, z.shape[1] // args.ncaps) # 2708, 16, 32
        result = []
        if train_:
            if epoch %5 ==0:
                X_mean = torch.mean(X_reshaped[prev_unconflicted].transpose(1,2),dim=1) # torch.Size([939, 16]) torch.Size([939])
                # if epoch%15==0:
                model_PH1 = api.Proposed_HSIC_Lasso(device=z.device, ch_dim=ch_dim,
                                                    lam=[np.inf, args.hsic_lamb])  # 원래 tol 1e-5
                model_PH1.input(X_mean, cluster_pred_use)
                model_PH1.classification_multi(B=20, M=3, kernels=['Gaussian'], n_jobs=8)
                _, scores = model_PH1.get_index_score()
                self.scores_list = scores.flatten()
            print('HSIC node no nero scores', len(torch.nonzero(self.scores_list)))
        for idx_f in range(args.ncaps):
            cur_output = self.gcn_agg(adj=remain_data_adj_sp, X=X_reshaped[:, idx_f, :])
            result.append(cur_output)
        x = torch.cat(result, dim=-1)
        return x, self.scores_list

    @torch.no_grad()
    def get_embedding(self, args, x, edge_index, train_neighbors, y, train_mask, l2_normalize, block):
        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x)
            x = self.bns[i](x)
            x = self.activation(x)
        x = self.dropout(x)
        x = self.convs[-1](x)
        x = self.bns[-1](x)
        disen_embedding, score_list = self.disentangle(args, 0 ,x, edge_index, train_neighbors, y, train_mask, block, train_=False)
        if l2_normalize:
            disen_embedding = F.normalize(disen_embedding, p=2, dim=1)
        return disen_embedding, score_list


class EdgeDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels,
                 num_layers=2, dropout=0.5, activation='relu'):
        super().__init__()
        self.mlps = nn.ModuleList()

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = 1 if i == num_layers - 1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels, second_channels))
        self.dropout = nn.Dropout(dropout)
        self.activation = create_activation_layer(activation)

    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()

    def forward(self, z, edge):
        x = z[edge[0]] * z[edge[1]]
        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)
        x = self.mlps[-1](x)
        return x

class ChannelDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels,
                 num_layers=2, dropout=0.5, activation='elu', ncaps=16):
        super().__init__()
        self.ncaps = ncaps
        self.mlps = nn.ModuleList()
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = in_channels if i == num_layers - 1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels, second_channels))
        self.dropout = nn.Dropout(dropout)
        self.activation = create_activation_layer(activation)
    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()
    def forward(self, ch_init):
        x=ch_init
        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)
        ch_rec = self.mlps[-1](x)
        return ch_rec.view(ch_rec.shape[0], self.ncaps, -1)


def routing_layer_32(x, num_caps, nlayer, max_iter, neighbors):
    batch_size=500
    dev = x.device
    n, d, k, m = x.shape[0], x.shape[1], num_caps, len(neighbors[0])

    delta_d = int(d // k)
    _cache_zero_d = torch.zeros(1, d).to(dev)
    _cache_zero_k = torch.zeros(1, k).to(dev)
    final_chunks = []

    for nl in range(nlayer):
        if nl > 0:
            x = final_chunks[-1]
        x = F.normalize(x.view(n, k, delta_d), dim=2).view(n, d)
        temp_z = torch.cat([x, _cache_zero_d], dim=0)
        final_chunks_batch = []

        for idx in range(0, neighbors.shape[0], batch_size):
            torch.cuda.empty_cache()

            batch_end = min(idx + batch_size, neighbors.shape[0])
            neigh = neighbors[idx:batch_end, :]
            chunk_size = neigh.shape[0]
            z = temp_z[neigh].view(chunk_size, m, k, delta_d)

            u = None
            for clus_iter in range(max_iter):
                if clus_iter == 0:
                    p = _cache_zero_k.expand(chunk_size * m, k).view(chunk_size, m, k)
                else:
                    p = torch.sum(z * u.view(chunk_size, 1, k, delta_d), dim=3)

                p = F.softmax(p, dim=2)
                u = torch.sum(z * p.view(chunk_size, m, k, 1), dim=1)

                u += x[idx:batch_end, :].view(chunk_size, k, delta_d)
                if clus_iter < max_iter - 1:
                    u = F.normalize(u, dim=2)

            final_chunks_batch.append(u.view(chunk_size, d))

        final_chunks_batch = torch.cat(final_chunks_batch, dim=0)
        final_chunks.append(final_chunks_batch)

    return final_chunks[-1]

class ClusterAssignment(nn.Module):
    def __init__(self, cluster_number, embedding_dimension, alpha, cluster_centers=None):
        super(ClusterAssignment, self).__init__()
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(cluster_number, embedding_dimension, dtype=torch.float)
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = torch.nn.Parameter(initial_cluster_centers)

    def forward(self, inputs):
        norm_squared = torch.sum((inputs.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

class MaskGAE(nn.Module):
    def __init__(
            self,
            args,
            encoder,
            edge_decoder,
            ch_decoder,
            mask,
            torch_generator,
            num_labels
    ):
        super().__init__()
        self.encoder = encoder
        self.edge_decoder = edge_decoder
        self.ch_decoder = ch_decoder
        self.mask = mask
        self.torch_generator = torch_generator
        self.edge_loss_fn = ce_loss
        self.ch_loss_fn = setup_loss_fn(args.alpha_l)
        self.negative_sampler = negative_sampling
        self.previous_unconflicted = []
        # clustring
        self.assignment = ClusterAssignment(cluster_number = num_labels, embedding_dimension = 512, alpha=1)
        self.kl_loss = torch.nn.KLDivLoss(size_average=False)
        self.beta1 = 0.4
        self.beta2 = 0.10
        self.cluster_pred=0

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.edge_decoder.reset_parameters()
        self.ch_decoder.reset_parameters()

    def forward(self, args, x, edge_index, neighbors, y, train_mask, B):
        embedding, test_score_list= self.encoder(args, x, edge_index, neighbors, y, train_mask, B)
        return embedding

    def neigh_sampler_torch(self, args, num_nodes, edge_index, epoch):
        neighbors = torch.zeros(1, args.nb_size).to(edge_index.device)
        first = edge_index[0]
        second = edge_index[1]
        for v in range(num_nodes):
            temp = second[(first == v).nonzero(as_tuple=True)[0]]
            if temp.shape[0] <= args.nb_size:
                shortage = args.nb_size - temp.shape[0]
                sampled_values = torch.cat(
                    (temp.reshape(1, -1), torch.IntTensor([-1]).repeat(shortage).reshape(1, -1).to(edge_index.device)),
                    1)
                neighbors = torch.cat((neighbors, sampled_values), dim=0)
            else:
                indice = random.sample(range(temp.shape[0]), args.nb_size)
                indice = torch.tensor(indice)
                sampled_values = temp[indice].reshape(1, -1)
                neighbors = torch.cat((neighbors, sampled_values), dim=0)

        return neighbors[1:].long()

    def train_epoch(
            self, args, train_data, optimizer, scheduler, batch_size, grad_norm, train_B, epoch):
        optimizer.zero_grad()

        x, edge_index, y = train_data.x, train_data.edge_index, train_data.y
        remaining_edges, masked_edges = self.mask(edge_index)

        aug_edge_index, _ = add_self_loops(edge_index)
        neg_edges = self.negative_sampler(aug_edge_index, num_nodes=train_data.num_nodes,
                                          num_neg_samples=masked_edges.view(2, -1).size(1), ).view_as(masked_edges)
        self.nhidden = args.encoder_out // args.ncaps
        self.disen_y = torch.arange(args.ncaps).long().unsqueeze(dim=0).repeat(train_data.num_nodes, 1).flatten().to(
            x.device)  # tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3,
        # remaining loss
        for perm in DataLoader(
                range(masked_edges.size(1)), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker,
                generator=self.torch_generator):

            train_neighbors = self.neigh_sampler_torch(args, train_data.num_nodes, remaining_edges, epoch)
            z, score_list = self.encoder(args, epoch, x, remaining_edges, train_neighbors, self.cluster_pred, self.previous_unconflicted, train_B)
            batch_masked_edges = masked_edges[:, perm]
            batch_neg_edges = neg_edges[:, perm]
            pos_out = self.edge_decoder(z, batch_masked_edges)
            neg_out = self.edge_decoder(z, batch_neg_edges)
            edge_loss = self.edge_loss_fn(pos_out, neg_out)
            print('calculating ch loss')
            ch_masking_temp = z.view(z.shape[0], args.ncaps, -1)
            non_zero_indices = torch.nonzero(score_list).squeeze().tolist()
            if isinstance(non_zero_indices, int):
                non_zero_indices = [non_zero_indices]
            zero_indices = list(set(range(args.ncaps)) - set(non_zero_indices))
            if isinstance(zero_indices, int):
                zero_indices = [zero_indices]
            mask_init = torch.cat([ch_masking_temp[:, i, :] if i in zero_indices else torch.zeros_like(ch_masking_temp[:, i, :]) for i in range(args.ncaps)],dim=-1)
            mask_recon = self.ch_decoder(mask_init)
            ch_init = ch_masking_temp[:, non_zero_indices, :]
            ch_rec = mask_recon[:, non_zero_indices, :]
            ch_loss = args.recon_alpha*self.ch_loss_fn(ch_rec, ch_init)
            loss = edge_loss +  ch_loss
            print('\nedge_loss', edge_loss, 'ch_loss', ch_loss)
            # ******************************************************************
            loss.backward()
            if grad_norm > 0:
                nn.utils.clip_grad_norm_(self.parameters(), grad_norm)
            optimizer.step()
            scheduler.step()
        return loss.item(), len(torch.nonzero(score_list)), z

    @torch.no_grad()
    def batch_predict(self, z, edges, batch_size=2 ** 16):
        preds = []
        for perm in DataLoader(range(edges.size(1)), batch_size, worker_init_fn=seed_worker,
                               generator=self.torch_generator):
            edge = edges[:, perm]

            preds += [self.edge_decoder(z, edge).squeeze().cpu()]
        pred = torch.cat(preds, dim=0)
        return pred

    @torch.no_grad()
    def test(self, z, pos_edge_index, neg_edge_index):
        pos_pred = self.batch_predict(z, pos_edge_index)
        neg_pred = self.batch_predict(z, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        pos_y = pos_pred.new_ones(pos_pred.size(0))
        neg_y = neg_pred.new_zeros(neg_pred.size(0))
        y = torch.cat([pos_y, neg_y], dim=0)
        y, pred = y.cpu().numpy(), pred.cpu().numpy()
        return roc_auc_score(y, pred), average_precision_score(y, pred)


