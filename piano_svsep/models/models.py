from torch import nn
from torch_geometric.nn import SAGEConv, to_hetero
import torch_geometric.nn as gnn
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, NamedTuple


class UnpoolInfo(NamedTuple):
    """
    A named tuple for storing information about the unpooling operation.
    """
    edge_index: Tensor
    cluster: Tensor
    batch: Tensor
    new_edge_score: Tensor


class ChordPredictor(torch.nn.Module):
    """
    A module for predicting chord edges from node embeddings.

    This class supports multiple pooling modes for chord prediction, including:
    - MLP
    - Dot Product
    - Cosine Similarity

    Attributes
    ----------
    in_channels : int
        Number of input channels.
    pooling_mode : str
        Pooling mode to use for chord prediction.
    dropout : float, optional
        Dropout rate (default is 0.0).

    Methods
    -------
    forward(x, edge_index, batch)
        Forward pass for chord prediction.
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
        elif self.pooling_mode == "dot_product":
            z = self.chord_predictor(x)
            src = z[edge_index[0]]
            dst = z[edge_index[1]]
            # dot product instead of cosine similarity
            edge_score_logits = (src*dst).sum(dim=-1)
        else:
            raise ValueError(f"Pooling mode {self.pooling_mode} not supported")

        return x, edge_score_logits
        
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_channels})'


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


class PianoSVSep(nn.Module):
    """
    A neural network module for piano staff and voice separation.

    This class implements a graph neural network (GNN) model for separating notes in a piano score into different voices and staves.

    Attributes
    ----------
    input_features : int
        Number of input features.
    hidden_features : int
        Number of hidden features.
    num_layers : int
        Number of layers in the encoder.
    activation : callable
        Activation function to use.
    dropout : float
        Dropout rate.
    conv_type : str
        Type of convolution to use in the encoder.
    gnn_metadata : tuple
        Metadata for the heterogeneous graph.
    chord_pooling_mode : str
        Pooling mode for chord prediction.
    staff_feedback : bool
        Whether to use staff feedback.
    pooling_layer : nn.Module or None
        Pooling layer for chord prediction.
    staff_clf : nn.Sequential
        Classifier for staff prediction.

    Methods
    -------
    forward(x_dict, edge_index_dict, pot_edges, pot_chord_edges, batch, onsets, durations, pitches, onset_beat, duration_beat, ts_beats, edge_attr_dict=None)
        Forward pass of the model.
    _init_weights(module)
        Initialize the weights of the model.
    """
    def __init__(self, input_features, hidden_features, num_layers, activation=F.relu, dropout=0.5,
                conv_type="SageConv", gnn_metadata=None, chord_pooling_mode="none", **kwargs):
        super().__init__()
        if conv_type == "SageConv":
            self.encoder = HSageEncoder(input_features, hidden_features, num_layers, activation, dropout, gnn_metadata)
        else:
            raise ValueError(f"Convolution type {conv_type} not supported")
        self.after_encoder_frontend = nn.Identity() # for compatibility with other experiments
        self.staff_feedback = kwargs.get("staff_feedback", False)
        self.decoder = EdgeDecoder(hidden_features, staff_feedback=self.staff_feedback)
        if chord_pooling_mode != "none":
            self.pooling_layer = ChordPredictor(hidden_features, pooling_mode = chord_pooling_mode, dropout=dropout)
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
        """
        Forward pass of the model.

        Parameters
        ----------
        x_dict : dict
            Dictionary of node feature tensors.
        edge_index_dict : dict
            Dictionary of edge indices.
        pot_edges : Tensor
            Potential edges between nodes.
        pot_chord_edges : Tensor
            Potential chord edges between nodes.
        batch : Tensor
            Batch indices.
        onsets : Tensor
            Onset times of the notes.
        durations : Tensor
            Durations of the notes.
        pitches : Tensor
            Pitches of the notes.
        onset_beat : Tensor
            Onset times in beats.
        duration_beat : Tensor
            Durations in beats.
        ts_beats : Tensor
            Time signature beats.
        edge_attr_dict : dict, optional
            Dictionary of edge attributes (default is None).

        Returns
        -------
        out : Tensor
            Output tensor.
        staff_logits : Tensor
            Staff prediction logits.
        hidden_features : Tensor
            Hidden features of the nodes.
        pooling_logits : Tensor
            Chord pooling logits.
        """
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


class EdgeDecoder(torch.nn.Module):
    """
    A decoder class for predicting edges (connections) between nodes in a graph.
    This class uses various features such as onset, duration, and pitch to predict
    the likelihood of edges between nodes.
    """
    def __init__(self, hidden_channels, mult_factor=1, staff_feedback=False, dropout=0.5):
        """
        Initialize the EdgeDecoder.

        Parameters
        ----------
        hidden_channels : int
            Number of hidden channels.
        mult_factor : int, optional
            Multiplication factor for hidden channels (default is 1).
        staff_feedback : bool, optional
            Whether to use staff feedback (default is False).
        dropout : float, optional
            Dropout rate (default is 0.5).
        """
        super().__init__()
        self.staff_feedback = staff_feedback
        input_dim = 2 * hidden_channels*mult_factor+3
        input_dim = input_dim + 4 if staff_feedback else input_dim
        self.dropout = nn.Dropout(dropout)
        self.normalize = nn.LayerNorm(hidden_channels)
        self.lin1 = torch.nn.Linear(input_dim, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, hidden_features, pot_edges, onsets, durations, pitches, onset_beat, duration_beat, ts_beats, staff_pred=None):

        """
        Forward pass for the EdgeDecoder.

        Parameters
        ----------
        hidden_features : Tensor
            Hidden features of the nodes.
        pot_edges : Tensor
            Potential edges between nodes.
        onsets : Tensor
            Onset times of the notes.
        durations : Tensor
            Durations of the notes.
        pitches : Tensor
            Pitches of the notes.
        onset_beat : Tensor
            Onset times in beats.
        duration_beat : Tensor
            Durations in beats.
        ts_beats : Tensor
            Time signature beats.
        staff_pred : Tensor, optional
            Staff predictions (default is None).

        Returns
        -------
        Tensor
            Predicted edge scores.
        """

        row, col = pot_edges
        oscore = self.onset_score(pot_edges, onsets, durations, onset_beat, duration_beat, ts_beats)
        pscore = self.pitch_score(pot_edges, pitches)
        hidden_features = torch.cat([hidden_features, staff_pred], dim=-1) if self.staff_feedback else hidden_features
        z = torch.cat([hidden_features[row], hidden_features[col], oscore, pscore], dim=-1)

        z = self.lin1(z).relu()
        z = self.normalize(self.dropout(z))
        z = self.lin2(z)
        return z.view(-1)

    def one_hot_encode_note_distance(self,distance):
        """
        One-hot encode the note distance.

        Parameters
        ----------
        distance : Tensor
            Distance between notes.

        Returns
        -------
        Tensor
            One-hot encoded distance.
        """
        out = distance == 0
        return out.float()

    def pitch_score(self, edge_index, mpitch):
        """
        Calculate the pitch score, i.e., a number from 0 to 1 measuring the distance 
        between two pitches. The smaller the number, the smaller the pitch distance.
        Pairs of notes with similar pitches are more likely to be connected.

        Parameters
        ----------
        edge_index: Tensor
            Edge indices.
        mpitch: Tensor
            MIDI pitches.

        Returns
        -------
            pscore: Tensor
                The Pitch scores between the notes.
        """
        pscore = torch.abs(mpitch[edge_index[1]]- mpitch[edge_index[0]])/127
        return pscore.unsqueeze(1)

    def onset_score(self, edge_index, onset, duration, onset_beat, duration_beat, ts_beats):
        """
        Calculate the onset score between notes, i.e., a measure of distance between the 
        previous note offset and the nect note onset.
        For each pair of notes, two scores are computed: a continous score between 0 and 1,
        and a binary score, telling if the next note is starting exactly when the previous note ends.

        Pairs of notes with closer onset times are more likely to be connected.

        Parameters
        ----------
        edge_index : Tensor
            Edge indices.
        onset : Tensor
            Onset times.
        duration : Tensor
            Durations.
        onset_beat : Tensor
            Onset times in beats.
        duration_beat : Tensor
            Durations in beats.
        ts_beats : Tensor
            Time signature beats.

        Returns
        -------
        Tensor
            Onset scores.
        """
        offset_beat = onset_beat + duration_beat
        note_distance_beat = onset_beat[edge_index[1]] - offset_beat[edge_index[0]]
        ts_beats_edges = ts_beats[edge_index[1]]
        continuous_oscore = 1 - torch.tanh(note_distance_beat / ts_beats_edges)

        offset = onset + duration
        binary_oscore = (onset[edge_index[1]] == offset[edge_index[0]]).float()
        oscore = torch.vstack((continuous_oscore,binary_oscore)).T
        return oscore
