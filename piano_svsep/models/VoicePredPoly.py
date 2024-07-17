from torch import nn
from torch_geometric.nn import MessagePassing, SAGEConv, to_hetero
import torch_geometric.nn as gnn
from typing import Callable, List, NamedTuple, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import coalesce, scatter, softmax
# from struttura.models.vocsep.VoicePredGeom import GNNEncoder, EdgeDecoder
from piano_svsep.models.VoicePredGeom import EdgeDecoder, GNNEncoder


class UnpoolInfo(NamedTuple):
    edge_index: Tensor
    cluster: Tensor
    batch: Tensor
    new_edge_score: Tensor


class FakeEdgePooling(torch.nn.Module):
    """
    This code behave like an edge pooling, but don't really pool anything.
    Instead it just fing the edges that should be pooled, and average node features across them
    """
    def __init__(
        self,
        in_channels: int,
        pooling_mode: str = "mlp",
        dropout: Optional[float] = 0.0,
    ):
        super().__init__()           
        self.in_channels = in_channels
        self.dropout = dropout

        if pooling_mode == "mlp":
            self.chord_predictor = nn.Sequential(
                nn.Linear(2*in_channels, in_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(in_channels),
                nn.Linear(in_channels, 1),
            )
        elif pooling_mode in ["dot_product","cos_similarity","dot_product_d","cos_similarity_d"]:
            self.chord_predictor = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                # nn.ReLU(),
                # nn.Dropout(dropout),
                # nn.LayerNorm(in_channels),
                # nn.Linear(in_channels, in_channels),
            )
        else:
            raise ValueError(f"Pooling mode {pooling_mode} not supported")
        self.pooling_mode = pooling_mode

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, UnpoolInfo, Tensor]:
        if self.pooling_mode == "mlp":
            edge_concatenated_feats = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
            edge_score_logits = self.chord_predictor(edge_concatenated_feats).view(-1)
        elif self.pooling_mode == "cos_similarity":
            z = self.chord_predictor(x)
            src = z[edge_index[0]]
            dst = z[edge_index[1]]
            edge_score_logits = F.cosine_similarity(src, dst)
        elif self.pooling_mode == "cos_similarity_d":
            z = x
            src = z[edge_index[0]]
            dst = z[edge_index[1]]
            edge_score_logits = F.cosine_similarity(src, dst)
        elif self.pooling_mode == "dot_product":
            z = self.chord_predictor(x)
            src = z[edge_index[0]]
            dst = z[edge_index[1]]
            # dot product instead of cosine similarity
            edge_score_logits = (src*dst).sum(dim=-1)
        elif self.pooling_mode == "dot_product_d":
            z = x
            src = z[edge_index[0]]
            dst = z[edge_index[1]]
            # dot product instead of cosine similarity
            edge_score_logits = (src*dst).sum(dim=-1)
        else:
            raise ValueError(f"Pooling mode {self.pooling_mode} not supported")

        edge_score_prob = torch.sigmoid(edge_score_logits)

        x = self._average_pooled_node_features(
            x, edge_index, batch, edge_score_prob)

        return x, edge_score_logits
    
    def _average_pooled_node_features(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_score_prob: Tensor,
    ) -> Tuple[Tensor]:
        # TODO : implement this
        return x
        
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_channels})'


class MusGConv(MessagePassing):
    def __init__(self, in_channels, out_channels, in_edge_channels=0, bias=True, return_edge_emb=False, **kwargs):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.return_edge_emb = return_edge_emb
        self.aggregation = kwargs.get("aggregation", "cat")
        self.in_edge_channels = in_edge_channels if in_edge_channels > 0 else in_channels
        self.lin = nn.Linear(in_channels, out_channels)
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.in_edge_channels, out_channels),
            nn.ReLU(),
            nn.LayerNorm(out_channels),
            nn.Linear(out_channels, out_channels),
        )
        self.proj = nn.Linear(3 * out_channels, out_channels) if self.aggregation == "cat" else nn.Linear(2 * out_channels, out_channels)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.lin.weight, gain=gain)
        nn.init.xavier_uniform_(self.proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.edge_mlp[0].weight, gain=gain)
        nn.init.xavier_uniform_(self.edge_mlp[3].weight, gain=gain)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is None:
            edge_attr = torch.abs(x[edge_index[0]] - x[edge_index[1]])
        x = self.lin(x)
        edge_attr = self.edge_mlp(edge_attr)
        h = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        h = self.proj(torch.cat((x, h), dim=-1))
        if self.bias is not None:
            h = h + self.bias
        if self.return_edge_emb:
            return h, edge_attr
        return h

    def message(self, x_j, edge_attr):
        if self.aggregation == "cat":
            return torch.cat((x_j, edge_attr), dim=-1)
        elif self.aggregation == "add":
            return x_j + edge_attr
        elif self.aggregation == "mul":
            return x_j * edge_attr
        else:
            raise ValueError("Aggregation type not supported")




class PostProcessPooling(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, edge_index, edge_probs, chord_edge_index, chord_edge_score, batch, num_nodes):
        edge_score = (chord_edge_score > self.threshold).float()
        masked_edges = chord_edge_index[:, chord_edge_score > self.threshold]
        cluster = torch.empty_like(batch)
        cluster_mask = torch.ones(num_nodes, device=edge_index.device, dtype=torch.bool)
        cluster_idx = 0
        # Loop through the chord edges and assign the same cluster to the nodes connected by the same chord edge
        for edge_idx in range(masked_edges.shape[-1]):
            src = masked_edges[0, edge_idx]
            dst = masked_edges[1, edge_idx]
            if cluster_mask[src] and cluster_mask[dst]:
                cluster[src] = cluster_idx
                cluster[dst] = cluster_idx
                cluster_mask[src] = False
                cluster_mask[dst] = False
                cluster_idx += 1
            elif cluster_mask[src]:
                cluster[src] = cluster[dst]
                cluster_mask[src] = False
            elif cluster_mask[dst]:
                cluster[dst] = cluster[src]
                cluster_mask[dst] = False

        # The remaining nodes are assigned to a cluster by themselves
        cluster[cluster_mask] = torch.arange(cluster_idx, cluster_idx + cluster_mask.sum(), device=edge_index.device)
        reduced_num_nodes = cluster_idx + cluster_mask.sum()
        # NOTE: Maybe the reduce needs to be changed to max instead of mean (or ablated) definitely not sum
        new_edge_index, new_edge_probs = coalesce(cluster[edge_index], edge_probs, num_nodes=reduced_num_nodes, reduce="mean")
        new_batch = batch.new_empty(reduced_num_nodes, dtype=torch.long)
        new_batch = new_batch.scatter_(0, cluster, batch)

        unpool_info = UnpoolInfo(edge_index=edge_index, cluster=cluster,
                                 batch=batch, new_edge_score=edge_score)

        return new_edge_index, new_edge_probs, unpool_info, reduced_num_nodes

    def unpool(
            self,
            edge_index,
            num_nodes,
            unpool_info: UnpoolInfo,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        x = torch.arange(num_nodes, device=edge_index.device)
        new_x = x[unpool_info.cluster]
        new_edge_index = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        for edge_idx in range(edge_index.shape[-1]):
            row = edge_index[0, edge_idx]
            col = edge_index[1, edge_idx]
            multiple_rows = torch.where(new_x == row)[0]
            multiple_cols = torch.where(new_x == col)[0]
            new_edges = torch.cartesian_prod(multiple_rows, multiple_cols).T
            new_edge_index = torch.cat((new_edge_index, new_edges), dim=-1)
        return new_edge_index


class HeteroMusGConv(nn.Module):
    """
    Convert a Graph Convolutional module to a hetero GraphConv module.

    Parameters
    ----------
    module: torch.nn.Module
        Module to convert

    Returns
    -------
    module: torch.nn.Module
        Converted module
    """

    def __init__(self, in_features, out_features, metadata, in_edge_features=0, bias=True, reduction='mean', return_edge_emb=False, aggregation="cat"):
        super(HeteroMusGConv, self).__init__()
        self.out_features = out_features
        self.return_edge_emb = return_edge_emb
        self.etypes = metadata[1]
        if reduction == 'mean':
            self.reduction = lambda x: x.mean(dim=0)
        elif reduction == 'sum':
            self.reduction = lambda x: x.sum(dim=0)
        elif reduction == 'max':
            self.reduction = lambda x: x.max(dim=0)
        elif reduction == 'min':
            self.reduction = lambda x: x.min(dim=0)
        elif reduction == 'concat':
            self.reduction = lambda x: torch.cat(x, dim=0)
        else:
            raise NotImplementedError

        conv_dict = dict()
        for etype in self.etypes:
            etype_str = "_".join(etype)
            conv_dict[etype_str] = MusGConv(in_features, out_features, bias=bias, in_edge_channels=in_edge_features, return_edge_emb=return_edge_emb, aggregation=aggregation)
        self.conv = nn.ModuleDict(conv_dict)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.conv.values():
            conv.reset_parameters()

    def forward(self, x, edge_index_dict, edge_feature_dict=None):
        x = x["note"] if isinstance(x, dict) else x
        edge_feature_dict = {key: None for key in self.etypes} if edge_feature_dict is None else edge_feature_dict
        out = torch.zeros((len(self.conv.keys()), x.shape[0], self.out_features), device=x.device)
        for idx, ekey in enumerate(self.etypes):
            etype_str = "_".join(ekey)
            if self.return_edge_emb:
                out[idx], edge_feature_dict[ekey] = self.conv[etype_str](x, edge_index_dict[ekey], edge_feature_dict[ekey])
            else:
                out[idx] = self.conv[etype_str](x, edge_index_dict[ekey], edge_feature_dict[ekey])
        if self.return_edge_emb:
            return {"note": self.reduction(out)}, edge_feature_dict
        return {"note": self.reduction(out)}

class SageEncoder(nn.Module):
    def __init__(self, hidden_features, num_layers, activation, dropout):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        num_layers = 2 if num_layers < 2 else num_layers
        for _ in range(num_layers):
            self.conv_layers.append(SAGEConv(hidden_features, hidden_features))
        self.normalize = gnn.GraphNorm(hidden_features)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x, edge_index):
        h = x
        for conv in self.conv_layers[: -1]:
            h = conv(h, edge_index)
            h = self.activation(h)
            h = self.dropout(h)
            h = self.normalize(h)
        h = self.conv_layers[-1](h, edge_index)
        return h


class HSageEncoder(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers, activation, dropout, metadata):
        super().__init__()
        num_layers = 2 if num_layers < 2 else num_layers
        self.conv = to_hetero(SageEncoder(hidden_features, num_layers, activation, dropout), metadata, aggr="mean")
        self.metadata = metadata
        self.first_linear = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_features),
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        h = x_dict["note"]
        h = self.first_linear(h)
        h_dict = {"note": h}
        h_dict = self.conv(h_dict, edge_index_dict)
        return h_dict


class MusGConvEncoder(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers, activation, dropout, metadata, return_edge_emb=False):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        self.return_edge_emb = return_edge_emb
        num_layers = 2 if num_layers < 2 else num_layers
        in_edge_deep_dim = 0 if return_edge_emb else 21
        self.conv_layers.append(HeteroMusGConv(hidden_features, hidden_features, metadata, in_edge_features=21, return_edge_emb=return_edge_emb))
        for _ in range(num_layers - 2):
            self.conv_layers.append(HeteroMusGConv(hidden_features, hidden_features, metadata, return_edge_emb=return_edge_emb, in_edge_features=in_edge_deep_dim))
        self.conv_layers.append(HeteroMusGConv(hidden_features, hidden_features, metadata, return_edge_emb=False, in_edge_features=in_edge_deep_dim))

        self.metadata = metadata
        self.normalize = gnn.GraphNorm(hidden_features)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.first_linear = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_features),
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        h = x_dict["note"]
        h = self.first_linear(h)
        h_dict = {"note": h}
        for conv in self.conv_layers[: -1]:
            if self.return_edge_emb:
                h_dict, edge_attr_dict = conv(h_dict, edge_index_dict, edge_attr_dict)
            else:
                h_dict = conv(h_dict, edge_index_dict, edge_attr_dict)
            h_dict = {key: self.activation(h_dict[key]) for key in h_dict.keys()}
            h_dict = {key: self.dropout(h_dict[key]) for key in h_dict.keys()}
            h_dict = {key: self.normalize(h_dict[key]) for key in h_dict.keys()}
        h_dict = self.conv_layers[-1](h, edge_index_dict, edge_attr_dict)
        return h_dict


class PolyphonicLinkPredictionModel(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers, activation=F.relu, dropout=0.5,
                conv_type="SageConv", gnn_metadata=None, chord_pooling_mode="none", after_encoder_frontend = False, **kwargs):
        super().__init__()
        self.edge_feature_feedback = kwargs.get("edge_feature_feedback", False)
        if conv_type == "SageConv":
            self.encoder = HSageEncoder(input_features, hidden_features, num_layers, activation, dropout, gnn_metadata)
        elif conv_type == "MusGConv":
            self.encoder = MusGConvEncoder(input_features, hidden_features, num_layers, activation, dropout,
                                           gnn_metadata, return_edge_emb=self.edge_feature_feedback)
        else:
            raise ValueError(f"Convolution type {conv_type} not supported")
        if after_encoder_frontend:
            self.after_encoder_frontend = nn.Linear(hidden_features, hidden_features)
        else:
            self.after_encoder_frontend = nn.Identity()
        self.staff_feedback = kwargs.get("staff_feedback", False)
        self.decoder = EdgeDecoder(hidden_features, staff_feedback=self.staff_feedback)
        if chord_pooling_mode == "graclus":
            self.pooling_layer = GraclusPooling(hidden_features, dropout=dropout)
        if chord_pooling_mode != "none":
            self.pooling_layer = FakeEdgePooling(hidden_features, pooling_mode = chord_pooling_mode, dropout=dropout)
            # self.pooling_layer = MuEdgePooling(hidden_features, dropout=dropout)
        else:
            self.pooling_layer = None
        self.staff_clf = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_features),
            nn.Linear(hidden_features, 2),
            nn.Sigmoid(),
        )
        self.chord_pooling_mode = chord_pooling_mode
        self.apply(self._init_weights)

    def forward(self, x_dict, edge_index_dict, pot_edges, pot_chord_edges, batch, onsets, durations, pitches, onset_beat, duration_beat, ts_beats, edge_attr_dict=None):
        z_dict = self.encoder(x_dict, edge_index_dict, edge_attr_dict)
        hidden_features = z_dict["note"]
        hidden_features = self.after_encoder_frontend(hidden_features)
        if self.chord_pooling_mode != "none":
            hidden_features, pooling_logits = self.pooling_layer(hidden_features, pot_chord_edges, batch)
        else:
            # create dummy chord edge score
            pooling_logits = torch.zeros_like(pot_chord_edges)
        staff_logits = self.staff_clf(hidden_features)
        out = self.decoder(hidden_features, pot_edges, onsets, durations, pitches, onset_beat, duration_beat, ts_beats, staff_logits)
        return out, staff_logits, hidden_features, pooling_logits
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)

