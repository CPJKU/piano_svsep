# code readapted from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/hetero_link_pred.py

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool, GraphConv, TransformerConv, GCN2Conv, ResGatedGraphConv, SAGEConv, HEATConv, to_hetero, HGTConv, MetaPath2Vec
import torch_geometric.nn as gnn

# METADATA = (['note'], [('note', 'onset', 'note'), ('note', 'consecutive', 'note'), ('note', 'during', 'note')])


class GNNEncoder(torch.nn.Module):
	def __init__(self, input_channels, hidden_channels, num_layers=1, activation=F.relu, dropout=0.5, jk_mode=None, conv_type="gcn", metadata= None, m2vec = True):
		super().__init__()
		self.num_layers = num_layers
		self.conv_layers = nn.ModuleList()
		# first convolution input -> hidden
		conv = get_conv_from_string(conv_type, input_channels, hidden_channels, metadata)
		self.conv_layers.append(conv)
		# other convolutions hidden -> hidden
		for _ in range(num_layers - 1):
			conv = get_conv_from_string(conv_type, hidden_channels, hidden_channels, metadata) 
			self.conv_layers.append(conv)
		self.metadata = metadata
		self.normalize = gnn.GraphNorm(hidden_channels)
		self.dropout = nn.Dropout(dropout)
		self.activation = activation
		# self.first_linear = gnn.Linear(-1, hidden_channels)
		if jk_mode is not None:
			self.jk = gnn.JumpingKnowledge(mode=jk_mode, channels=hidden_channels, num_layers=num_layers+1)
		else:
			self.jk = None
		self.conv_type = conv_type

	def forward(self, x, edge_index):
		if self.conv_type in ['HGTConv', 'HEATConv','HANConv']: # some models are inherently heterogenous
			h_dict = x
			# h_dict = {key: self.first_linear(h) for key, h in h_dict.items()}
			# h_dict = {key: self.activation(h) for key, h in h_dict.items()}
			# h_dict = {key: self.normalize(h) for key, h in h_dict.items()}
			for conv in self.conv_layers:
				h_dict = conv(h_dict, edge_index)
				h_dict = {key: self.activation(h) for key, h in h_dict.items()}
				h_dict = {key: self.normalize(h) for key, h in h_dict.items()}
				h_dict = {key: self.dropout(h) for key, h in h_dict.items()}
			return h_dict
		else:
			hs = list() # to save hidden stated for jump connection
			h = x
			for conv in self.conv_layers[: -1]:
				h = conv(h, edge_index)
				h = self.activation(h)
				h = self.normalize(h)
				h = self.dropout(h)
				hs.append(h)
			h = self.conv_layers[-1](h, edge_index)
			# h = self.normalize(h)
			hs.append(h)
			if self.jk is not None:
				h = self.jk(hs)

			return h


class EdgeDecoder(torch.nn.Module):
	def __init__(self, hidden_channels, mult_factor=1, staff_feedback=False, dropout=0.5):
		super().__init__()
		self.staff_feedback = staff_feedback
		input_dim = 2 * hidden_channels*mult_factor+3
		input_dim = input_dim + 4 if staff_feedback else input_dim
		self.dropout = nn.Dropout(dropout)
		self.normalize = nn.LayerNorm(hidden_channels)
		self.lin1 = Linear(input_dim, hidden_channels)
		self.lin2 = Linear(hidden_channels, 1)

	def forward(self, hidden_features, pot_edges, onsets, durations, pitches, onset_beat, duration_beat, ts_beats, staff_pred=None):
		row, col = pot_edges
		# z = torch.cat([z_dict['note'][row], z_dict['note'][col]], dim=-1)
		# one_hot_encode_note_distance =self.one_hot_encode_note_distance(onsets[col] - offsets[row]).unsqueeze(1)
		# onset_score = self.onset_score(edge_label_index, onsets, durations).unsqueeze(1)
		oscore = self.onset_score(pot_edges, onsets, durations, onset_beat, duration_beat, ts_beats)
		pscore = self.pitch_score(pot_edges, pitches)
		# z = torch.cat([z_dict['note'][row], z_dict['note'][col], one_hot_encode_note_distance, onset_score ], dim=-1)
		hidden_features = torch.cat([hidden_features, staff_pred], dim=-1) if self.staff_feedback else hidden_features
		z = torch.cat([hidden_features[row], hidden_features[col], oscore, pscore], dim=-1)
		
		z = self.lin1(z).relu()
		z = self.normalize(self.dropout(z))
		z = self.lin2(z)
		return z.view(-1)

	def one_hot_encode_note_distance(self,distance):
		out = distance == 0
		return out.float()

	def pitch_score(self, edge_index, mpitch):
		"""Pitch score from midi to freq."""
		# a = 440  # frequency of A (coomon value is 440Hz)
		# fpitch = (a / 32) * (2 ** ((mpitch - 9) / 12))
		# pscore = torch.pow(
        #     torch.div(torch.min(fpitch[edge_index], dim=0)[0], torch.max(fpitch[edge_index], dim=0)[0]), 3.1)
		pscore = torch.abs(mpitch[edge_index[1]]- mpitch[edge_index[0]])/127
		return pscore.unsqueeze(1)

	def onset_score(self, edge_index, onset, duration, onset_beat, duration_beat, ts_beats):
		offset = onset + duration
		offset_beat = onset_beat + duration_beat
		note_distance_beat = onset_beat[edge_index[1]] - offset_beat[edge_index[0]]
		ts_beats_edges = ts_beats[edge_index[1]]
		# oscore = 1- (1/(1+torch.exp(-2*(note_distance_beat/ts_beats_edges)))-0.5)*2
		oscore = 1 - torch.tanh(note_distance_beat / ts_beats_edges)
		one_hot_pitch_score = (onset[edge_index[1]] == offset[edge_index[0]]).float()
		oscore = torch.cat((oscore.unsqueeze(1), one_hot_pitch_score.unsqueeze(1)), dim=1)
		return oscore
		
def get_conv_from_string(conv_type, input_channels, hidden_channels, metadata):
	if conv_type == 'GCNConv':
		return gnn.GCNConv(-1, hidden_channels)
	elif conv_type == 'GraphConv':
		return gnn.GraphConv(input_channels, hidden_channels)
	elif conv_type == 'SAGEConv':
		return gnn.SAGEConv(input_channels, hidden_channels)
	elif conv_type == 'GCN2Conv':
		return gnn.GCN2Conv(input_channels, hidden_channels)
	elif conv_type == 'ResGatedGraphConv':
		return gnn.ResGatedGraphConv(input_channels, hidden_channels)
	elif conv_type == 'HEATConv':
		return gnn.HEATConv(-1, hidden_channels,  num_node_types=1, num_edge_types=5, edge_type_emb_dim=1, edge_dim=1, edge_attr_emb_dim = 1, heads = 8, )
	elif conv_type == 'HGTConv':
		return gnn.HGTConv(-1, hidden_channels, metadata, heads=8)
	elif conv_type == 'GATConv':
		return gnn.GATConv(-1, hidden_channels, add_self_loops=True)
	elif conv_type == 'SGConv' :
		return gnn.SGConv(-1, hidden_channels)
	elif conv_type == 'HANConv':
		return gnn.HANConv(-1, hidden_channels, metadata, heads=8)
	else:
		raise TypeError(f"{conv_type} is not a supported convolution type.")

class PGLinkPredictionModel(nn.Module):
	def __init__(self, graph_metadata, input_features, hidden_features, num_layers, activation=F.relu, dropout=0.5, jk_mode="lstm", conv_type="gcn"):
		super().__init__()
		self.encoder = GNNEncoder(input_features, hidden_features, hidden_features, num_layers, activation, dropout, jk_mode, conv_type, graph_metadata)
		if conv_type not in ['HEATConv','HGTConv','HANConv']: # Some models are inherently heterogeneous
			self.encoder = to_hetero(self.encoder, graph_metadata, aggr='sum')
		if jk_mode == "cat": # if we use cat, we need to multiply the hidden features by the number of layers
			mult_factor = num_layers
		else:
			mult_factor = 1
		self.decoder = EdgeDecoder(hidden_features, mult_factor)

	def forward(self, x_dict, edge_index_dict, edge_label_index, onsets, durations, pitches, onset_beat, duration_beat, ts_beats):
		z_dict = self.encoder(x_dict, edge_index_dict)
		return self.decoder(z_dict, edge_label_index, onsets, durations, pitches, onset_beat, duration_beat, ts_beats)


class PygHomoLinkPredictionModel(nn.Module):
	def __init__(self, in_feats, n_hidden, n_layers, activation=F.relu, dropout=0.5, alpha=3.1, smote=False):
		super(PygHomoLinkPredictionModel, self).__init__()
		self.n_hidden = n_hidden
		self.layers = nn.ModuleList()
		self.normalize = gnn.BatchNorm(n_hidden)
		self.activation = activation
		self.alpha = alpha
		self.dropout = nn.Dropout(dropout)
		self.layers.append(gnn.GCNConv(in_feats, n_hidden))
		for i in range(n_layers):
			self.layers.append(gnn.SGConv(n_hidden, n_hidden))
		self.jk = gnn.JumpingKnowledge(mode='lstm', channels=n_hidden, num_layers=n_layers+1)
		self.predictor = nn.Sequential(
			nn.Linear(2*n_hidden, n_hidden),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.BatchNorm1d(n_hidden),
			nn.Linear(n_hidden, 1))

	def forward(self, edge_index, x, edge_label_index=None):
		h = x
		hs = list()
		for i, layer in enumerate(self.layers[:-1]):
			h = layer(h, edge_index)
			h = self.activation(h)
			h = self.normalize(h)
			h = self.dropout(h)
			hs.append(h)
		h = self.layers[-1](h, edge_index)
		hs.append(h)
		h = self.jk(hs)
		pred = self.predict(h, edge_label_index)
		return torch.sigmoid(pred)

	def predict(self, h, edge_label_index):
		row, col = edge_label_index
		z = torch.cat((h[row], h[col]), dim=-1)
		z = self.predictor(z)
		return z




