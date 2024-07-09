from torch.optim import Adam, lr_scheduler
from struttura.models.core import *
from pytorch_lightning import LightningModule
from torchmetrics import F1Score, Accuracy
import torch
from struttura.models.core import MLP, GCN, UNet
from torch_scatter import scatter_add
from random import randint
import random
from torch import nn
from torch.nn import functional as F


class SMOTE(object):
    """
    Minority Sampling with SMOTE.
    """
    def __init__(self, distance='euclidian', dims=512, k=2):
        super(SMOTE, self).__init__()
        self.newindex = 0
        self.k = k
        self.dims = dims
        self.distance_measure = distance

    def populate(self, N, i, nnarray, min_samples, k):
        while N:
            nn = randint(0, k - 2)

            diff = min_samples[nnarray[nn]] - min_samples[i]
            gap = random.uniform(0, 1)

            self.synthetic_arr[self.newindex, :] = min_samples[i] + gap * diff

            self.newindex += 1

            N -= 1

    def k_neighbors(self, euclid_distance, k):
        nearest_idx = torch.zeros((euclid_distance.shape[0], euclid_distance.shape[0]), dtype=torch.int64)

        idxs = torch.argsort(euclid_distance, dim=1)
        nearest_idx[:, :] = idxs

        return nearest_idx[:, 1:k]

    def find_k(self, X, k):
        z = F.normalize(X, p=2, dim=1)
        distance = torch.mm(z, z.t())
        return self.k_neighbors(distance, k)

    # TODO: Need to find a matrix version of Euclid Distance and Cosine Distance.
    def find_k_euc(self, X, k):
        euclid_distance = torch.zeros((X.shape[0], X.shape[0]), dtype=torch.float32)

        for i in range(len(X)):
            dif = (X - X[i]) ** 2
            dist = torch.sqrt(dif.sum(axis=1))
            euclid_distance[i] = dist

        return self.k_neighbors(euclid_distance, k)

    def find_k_cos(self, X, k):
        cosine_distance = torch.zeros((X.shape[0], X.shape[0]), dtype=torch.float32)

        for i in range(len(X)):
            dist = F.cosine_similarity(X, X[i].unsqueeze(0), dim=1)
            cosine_distance[i] = dist

        return self.k_neighbors(cosine_distance, k)

    def generate(self, min_samples, N, k):
        """
        Returns (N/100) * n_minority_samples synthetic minority samples.
        Parameters
        ----------
        min_samples : Numpy_array-like, shape = [n_minority_samples, n_features]
            Holds the minority samples
        N : percetange of new synthetic samples:
            n_synthetic_samples = N/100 * n_minority_samples. Can be < 100.
        k : int. Number of nearest neighbours.
        Returns
        -------
        S : Synthetic samples. array,
            shape = [(N/100) * n_minority_samples, n_features].
        """
        T = min_samples.shape[0]
        self.synthetic_arr = torch.zeros(int(N / 100) * T, self.dims)
        N = int(N / 100)
        if self.distance_measure == 'euclidian':
            indices = self.find_k_euc(min_samples, k)
        elif self.distance_measure == 'cosine':
            indices = self.find_k_cos(min_samples, k)
        else:
            indices = self.find_k(min_samples, k)
        for i in range(indices.shape[0]):
            self.populate(N, i, indices[i], min_samples, k)
        self.newindex = 0
        return self.synthetic_arr

    def fit_generate(self, X, y):
        """
        Over-samples using SMOTE. Returns synthetic samples concatenated at the end of the original samples.
        Parameters
        ----------
        X : Numpy_array-like, shape = [n_samples, n_features]
            The input features
        y : Numpy_array-like, shape = [n_samples]
            The target labels.

        Returns
        -------
        X_resampled : Numpy_array, shape = [(n_samples + n_synthetic_samples), n_features]
            The array containing the original and synthetic samples.
        y_resampled : Numpy_array, shape = [(n_samples + n_synthetic_samples)]
            The corresponding labels of `X_resampled`.
        """
        # get occurence of each class
        occ = torch.eye(int(y.max() + 1), int(y.max() + 1))[y].sum(axis=0)
        # get the dominant class
        dominant_class = torch.argmax(occ)
        # get occurence of the dominant class
        n_occ = int(occ[dominant_class].item())
        for i in range(len(occ)):
            # For Mini-Batch Training exclude examples with less than k occurances in the mini banch.
            if i != dominant_class and occ[i] >= self.k:
                # calculate the amount of synthetic data to generate
                N = (n_occ - occ[i]) * 100 / occ[i]
                if N != 0:
                    candidates = X[y == i]
                    xs = self.generate(candidates, N, self.k)
                    # TODO Possibility to add Gaussian noise here for ADASYN approach, important for mini-batch training with respect to the max euclidian distance.
                    X = torch.cat((X, xs.to(X.get_device()))) if X.get_device() >= 0 else torch.cat((X, xs))
                    ys = torch.ones(xs.shape[0]) * i
                    y = torch.cat((y, ys.to(y.get_device()))) if y.get_device() >= 0 else torch.cat((y, ys))
        return X, y

class IdentityTuple(object):
    def __init__(self):
        super(IdentityTuple, self).__init__()

    def fit_generate(self, X, y):
        return X, y


class LinkPredictionModel(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, activation=F.relu, dropout=0.5, alpha=3.1, smote=False, block="ResConv", jk=True):
        super(LinkPredictionModel, self).__init__()
        self.n_hidden = n_hidden
        self.layers = nn.ModuleList()
        self.normalize = nn.LayerNorm(n_hidden)
        self.activation = activation
        self.use_jk = jk
        self.block = block
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        if block == "ResConv":
            self.layers.append(ResGatedGraphConv(in_feats, n_hidden, bias=True))
            for i in range(n_layers):
                self.layers.append(ResGatedGraphConv(n_hidden, n_hidden, bias=True))
        elif block == "SageConv" or block == "Sage":
            self.layers.append(SageConvLayer(in_feats, n_hidden, bias=True))
            for i in range(n_layers):
                self.layers.append(SageConvLayer(n_hidden, n_hidden, bias=True))
        else:
            raise ValueError("Block type not supported")
        if self.use_jk:
            self.jk = JumpingKnowledge(n_hidden, n_layers+1)
        self.predictor = nn.Sequential(
            nn.Linear(n_hidden*2+3, int(n_hidden)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(n_hidden), int(n_hidden/2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(n_hidden/2), 1))

    def reset_parameters(self):
        nn.init.constant_(self.score_weight, 0)

    def predict(self, h, edge_index, pitch_score, onset_score):
        src, dst = edge_index
        h_src, h_dst = h[src], h[dst]
        score = self.predictor(torch.cat([h_src, h_dst, pitch_score, onset_score], dim=1))
        return torch.sigmoid(score)

    def embed(self, x, edge_index):
        h = x
        hs = list()
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index)
            if i != len(self.layers) - 1:
                h = self.activation(h)
                h = self.normalize(h)
                h = self.dropout(h)
            hs.append(h)
        if self.use_jk:
            h = self.jk(hs)
        return h

    def forward(self, target_edge_index, x, embed_edge_index, pitch_score, onset_score):
        h = self.embed(x, embed_edge_index)
        pred = self.predict(h, target_edge_index, pitch_score, onset_score)
        return pred


class HeteroLinkPredictionModel(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, activation=F.relu, dropout=0.5, alpha=3.1, etypes={"onset":0, "consecutive":1, "during":2, "rests":3, "consecutive_rev":4, "during_rev":5, "rests_rev":6}, smote=False, block="ResConv", jk=True):
        super(HeteroLinkPredictionModel, self).__init__()
        self.n_hidden = n_hidden
        self.layers = nn.ModuleList()
        self.normalize = nn.LayerNorm(n_hidden)
        self.activation = activation
        self.use_jk = jk
        self.block = block
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        if block == "ResConv":
            self.layers.append(HeteroResGatedGraphConvLayer(in_feats, n_hidden, etypes, bias=True))
            for i in range(n_layers):
                self.layers.append(HeteroResGatedGraphConvLayer(n_hidden, n_hidden, etypes, bias=True))
        elif block == "SageConv" or block == "Sage":
            self.layers.append(HeteroSageConvLayer(in_feats, n_hidden, etypes, bias=True))
            for i in range(n_layers):
                self.layers.append(HeteroSageConvLayer(n_hidden, n_hidden, etypes, bias=True))
        else:
            raise ValueError("Block type not supported")
        if self.use_jk:
            self.jk = JumpingKnowledge(n_hidden, n_layers+1)
        self.predictor = nn.Sequential(
            nn.Linear(n_hidden*2+3, int(n_hidden)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(n_hidden), int(n_hidden/2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(n_hidden/2), 1))

    def reset_parameters(self):
        nn.init.constant_(self.score_weight, 0)

    def predict(self, h, edge_index, pitch_score, onset_score):
        src, dst = edge_index
        h_src, h_dst = h[src], h[dst]
        score = self.predictor(torch.cat([h_src, h_dst, pitch_score, onset_score], dim=1))
        return torch.sigmoid(score)

    def embed(self, x, edge_index, edge_type):
        h = x
        hs = list()
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index, edge_type)
            if i != len(self.layers) - 1:
                h = self.activation(h)
                h = self.normalize(h)
                h = self.dropout(h)
            hs.append(h)
        if self.use_jk:
            h = self.jk(hs)
        return h

    def forward(self, target_edge_index, x, embed_edge_index, edge_type, pitch_score, onset_score):
        h = self.embed(x, embed_edge_index, edge_type)
        pred = self.predict(h, target_edge_index, pitch_score, onset_score)
        return pred


class GVocSep(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, activation=F.relu, dropout=0.5):
        super(GVocSep, self).__init__()
        self.n_hidden = n_hidden
        self.layers = nn.ModuleList()
        self.normalize = nn.BatchNorm1d(n_hidden)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.smote = SMOTE(dims=n_hidden)
        self.layers.append(SageConvLayer(in_feats, n_hidden))
        for i in range(n_layers):
            self.layers.append(SageConvLayer(n_hidden, n_hidden))
        self.jk = JumpingKnowledge(n_hidden, n_layers+1)
        self.predictor = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 2))

    def predict(self, h_src, h_dst, y):
        h, y = self.smote.fit_generate(h_src * h_dst, y)
        return self.predictor(h), y.long()

    def val_predict(self, h_src, h_dst):
        return self.predictor(h_src * h_dst)

    def forward(self, edge_index, adj, x, y):
        h = x
        hs = []
        for i, layer in enumerate(self.layers):
            h = layer(h, adj)
            if i != len(self.layers) - 1:
                h = self.activation(h)
                h = self.normalize(h)
                h = self.dropout(h)
                hs.append(h)
        h = self.jk(hs)
        src_idx, dst_idx = edge_index
        out, y = self.predict(h[src_idx], h[dst_idx], y)
        return out, y

    def val_forward(self, edge_index, adj, x):
        h = x
        hs = []
        for i, layer in enumerate(self.layers):
            h = layer(h, adj)
            if i != len(self.layers) - 1:
                h = self.activation(h)
                h = self.normalize(h)
                h = self.dropout(h)
                hs.append(h)
        h = self.jk(hs)
        src_idx, dst_idx = edge_index
        out = self.val_predict(h[src_idx], h[dst_idx])
        return out


class NeoGNN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, activation=F.relu, dropout=0.5):
        super(NeoGNN, self).__init__()
        self.n_hidden = n_hidden
        self.normalize = nn.BatchNorm1d(n_hidden)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.gnn = GCN(in_feats, n_hidden, n_hidden, n_layers, activation, dropout)
        self.predictor = MLP(n_hidden, n_hidden, 1, n_layers, activation, dropout)
        # NeoGNN args
        self.alpha = torch.nn.Parameter(torch.FloatTensor([0, 0]))
        self.f_edge = MLP(1, n_hidden, 1)
        self.f_node = MLP(1, n_hidden, 1)
        self.g_phi = MLP(1, n_hidden, 1)

    def reset_parameters(self):
        nn.init.constant_(self.alpha, 0)
        nn.init.xavier_uniform_(self.f_edge.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.f_node.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.g_phi.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, edge_index, x, adj):
        h = self.gnn(adj, x)
        out_feat = self.predictor(h[edge_index[0]] * h[edge_index[1]])
        row_A, col_A = edge_index[0], edge_index[1]
        edge_weight_A = torch.ones((len(edge_index[0])))
        edge_weight_A = self.f_edge(edge_weight_A.unsqueeze(-1))
        node_struct_feat = scatter_add(edge_weight_A, col_A, dim=0, dim_size=len(x))

        edge_weight_src = adj[edge_index[0]].T * self.f_node(node_struct_feat[edge_index[0]]).squeeze()
        edge_weight_dst = adj[edge_index[1]].T * self.f_node(node_struct_feat[edge_index[1]]).squeeze()

        out_struct = torch.mm(edge_weight_src.t(), edge_weight_dst).diag()
        out_struct = self.g_phi(out_struct.unsqueeze(-1))
        out_struct_raw = out_struct
        out_struct = torch.sigmoid(out_struct)
        alpha = torch.softmax(self.alpha, dim=0)
        out = alpha[0] * out_struct + alpha[1] * out_feat + 1e-15
        return out, out_struct, out_feat, out_struct_raw


class GVAE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, activation=F.relu, dropout=0.5):
        super(GVAE, self).__init__()
        self.n_hidden = n_hidden
        self.normalize = nn.BatchNorm1d(n_hidden)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.gnn = GCN(in_feats, n_hidden, n_hidden, n_layers, activation, dropout)
        self.proj_1 = MLP(n_hidden, n_hidden, n_hidden, 1, activation, dropout)
        self.proj_2 = MLP(n_hidden, n_hidden, n_hidden, 1, activation, dropout)

    def decode(self, h):
        z1 = self.proj_1(h)
        z2 = self.proj_2(h)
        z = torch.sigmoid(torch.mm(z1, z2.T))
        return z

    def encode(self, adj, x):
        h = self.gnn(adj, x)
        return h

    def forward(self, adj, x):
        h = self.encode(adj, x)
        h = self.decode(h)
        return h


class GaugLoss(nn.Module):
    def __init__(self):
        super(GaugLoss, self).__init__()

    def forward(self, adj_rec, adj_tgt):
        if adj_tgt.is_sparse:
            shape = adj_tgt.size()
            indices = adj_tgt._indices().T
            adj_sum = torch.sparse.sum(adj_tgt)
            bce_weight = (shape[0] * shape[1] - adj_sum) / adj_sum
            norm_w = shape[0] * shape[1] / float((shape[0] * shape[1] - adj_sum) * 2)
            bce_loss = norm_w * F.binary_cross_entropy_with_logits(torch.transpose(adj_rec[:shape[1], :shape[0]], 0, 1),
                                                                   adj_tgt.to_dense(), pos_weight=bce_weight)
        else:
            shape = adj_tgt.shape
            indices = adj_tgt.nonzero()
            adj_sum = torch.sum(adj_tgt)
            bce_weight = (shape[0]*shape[1] - adj_sum) / adj_sum
            norm_w = shape[0]*shape[1] / float((shape[0]*shape[1] - adj_sum) * 2)
            bce_loss = norm_w * F.binary_cross_entropy_with_logits(torch.transpose(adj_rec[:shape[1], :shape[0]], 0, 1), adj_tgt, pos_weight=bce_weight)
        return bce_loss


class GraphVoiceSeparationModel(LightningModule):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_layers,
                 activation=F.relu,
                 dropout=0.5,
                 lr=0.001,
                 weight_decay=5e-4,
        ):
        super(GraphVoiceSeparationModel, self).__init__()
        self.save_hyperparameters()
        self.module = GVocSep(in_feats, n_hidden, n_layers, activation, dropout)
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loss = nn.CrossEntropyLoss()
        self.train_acc = Accuracy()
        self.train_f1 = F1Score(num_classes=2, average="macro")
        self.val_acc_score = Accuracy()
        self.val_f1_score = F1Score(num_classes=2, average="macro")

    def training_step(self, batch, batch_idx):
        batch_inputs, edges, batch_label = batch
        batch_inputs = batch_inputs.squeeze()
        edges = edges.squeeze().to(self.device)
        batch_label = batch_label.squeeze()
        batch_labels = torch.all(batch_label[edges[0]] == batch_label[edges[1]], dim=1).long().to(self.device)
        batch_inputs = batch_inputs.to(self.device)
        adj = torch.sparse_coo_tensor(
            edges, torch.ones(len(edges[0])).to(self.device), (len(batch_inputs), len(batch_inputs))).to_dense().to(self.device)
        batch_pred, batch_labels = self.module(edges, adj, batch_inputs, batch_labels)
        loss = self.train_loss(batch_pred, batch_labels)
        batch_acc = self.train_acc(batch_pred, batch_labels)
        batch_f1 = self.train_f1(batch_pred, batch_labels)
        self.log("train_loss", loss.item(), prog_bar=True, on_epoch=True)
        self.log("train_acc", batch_acc.item(), prog_bar=True, on_epoch=True)
        self.log("train_f1", batch_f1.item(), prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_inputs, edges, batch_label = batch
        batch_inputs = batch_inputs.squeeze()
        edges = edges.squeeze()
        edges = torch.cat((edges, torch.cat((edges[1].unsqueeze(0), edges[0].unsqueeze(0)))), dim=1).to(self.device)
        batch_label = batch_label.squeeze()
        batch_labels = torch.all(batch_label[edges[0]] == batch_label[edges[1]], dim=1).long().to(self.device)
        batch_inputs = batch_inputs.to(self.device).to(self.device)
        adj = torch.sparse_coo_tensor(
            edges, torch.ones(len(edges[0])).to(self.device), (len(batch_inputs), len(batch_inputs))).to_dense().to(self.device)
        batch_pred = self.module.val_forward(edges, adj, batch_inputs)

        val_acc_score = self.val_acc_score(batch_pred, batch_labels)
        val_f1_score = self.val_f1_score(batch_pred, batch_labels)
        self.log("val_acc", val_acc_score.item(), prog_bar=True, on_epoch=True)
        self.log("val_f1", val_f1_score.item(), prog_bar=True, on_epoch=True)
        # return val_auc_score

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
        }

