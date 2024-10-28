from torch.nn import functional as F
import torch
from piano_svsep.utils import isin_pairwise, compute_voice_f1_score, linear_assignment
from pytorch_lightning import LightningModule
from piano_svsep.models.models import PianoSVSep
from piano_svsep.postprocessing import PostProcessPooling
from piano_svsep.utils import add_reverse_edges
from torchmetrics import F1Score, Accuracy
import numpy as np


class PLPianoSVSep(LightningModule):
    """
    PyTorch Lightning Module for Piano Symbolic Voice Separation.

    Parameters
    ----------
    in_feats : int
        Number of input features.
    n_hidden : int
        Number of hidden units.
    n_layers : int, optional
        Number of layers in the GNN, by default 2.
    activation : callable, optional
        Activation function, by default F.relu.
    dropout : float, optional
        Dropout rate, by default 0.5.
    lr : float, optional
        Learning rate, by default 0.001.
    weight_decay : float, optional
        Weight decay for the optimizer, by default 5e-4.
    rev_edges : str, optional
        Type of reverse edges to add, by default "new_type".
    pos_weights : dict, optional
        Positive weights for loss functions, by default None.
    conv_type : str, optional
        Type of convolution layer, by default "SAGEConv".
    chord_pooling_mode : str, optional
        Mode for chord pooling, by default "none".
    feat_norm_scale : float, optional
        Scale for feature normalization loss, by default 0.1.
    staff_feedback : bool, optional
        Whether to use staff feedback, by default False.
    """
    def __init__(
            self,
            in_feats,
            n_hidden,
            n_layers=2,
            activation=F.relu,
            dropout=0.5,
            lr=0.001,
            weight_decay=5e-4,
            rev_edges="new_type",
            pos_weights=None,
            conv_type="SAGEConv",
            chord_pooling_mode="none",
            feat_norm_scale = 0.1,
            staff_feedback=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.feat_norm_scale = feat_norm_scale
        self.gnn_metadata = (['note'],
                             [('note', 'onset', 'note'),
                              ('note', 'consecutive', 'note'),
                              ('note', 'during', 'note'),
                              ('note', 'rest', 'note'),
                              ('note', 'consecutive_rev', 'note'),
                              ('note', 'during_rev', 'note'),
                              ('note', 'rest_rev', 'note'),
                              ])
        self.module = PianoSVSep(
            input_features=in_feats, hidden_features=n_hidden, num_layers=n_layers, activation=activation,
            dropout=dropout, conv_type=conv_type, gnn_metadata=self.gnn_metadata, chord_pooling_mode=chord_pooling_mode,
            staff_feedback=staff_feedback)
        self.rev_edges = rev_edges
        self.threshold = 0.5
        self.chord_pooling_mode = chord_pooling_mode
        self.ps_pool = PostProcessPooling()
        self.pitch_embedding = torch.nn.Embedding(12, 16)
        # train metrics
        self.voice_loss = torch.nn.BCEWithLogitsLoss(pos_weight= torch.tensor([pos_weights["voice"]]))
        self.staff_loss = torch.nn.CrossEntropyLoss()
        self.pooling_loss = torch.nn.BCEWithLogitsLoss(pos_weight= torch.tensor([pos_weights["chord"]]))
        self.staff_f1 = F1Score(task="multiclass", num_classes=2)
        self.voice_f1 = F1Score(task="binary")
        self.chord_f1 = F1Score(task="binary")
        self.staff_acc = Accuracy(task="multiclass", num_classes=2)
        self.voice_acc = Accuracy(task="binary")

    def _common_step(self, graph, step_type="train", **kwargs):
        """
        Common step for training, validation, and testing.

        Parameters
        ----------
        graph : torch_geometric.data.HeteroData
            Input graph data with potential and truth edges.
            These edges are used as ground truth for training.
        step_type : str, optional
            Type of step, by default "train".

        Returns
        -------
        loss : torch.Tensor
            Computed loss.
        """
        if self.rev_edges is not None:
            add_reverse_edges(graph, mode=self.rev_edges)  # Add reverse edges to the graph if specified

        gbatch = graph["note"].batch
        edge_index_dict = graph.edge_index_dict
        x_dict = graph.x_dict
        pot_edges = edge_index_dict.pop(("note", "potential", "note"))
        truth_edges = edge_index_dict.pop(("note", "truth", "note"))
        pot_chord_edges = edge_index_dict.pop(("note", "chord_potential", "note"))
        truth_chord_edges = edge_index_dict.pop(("note", "chord_truth", "note"))
        staff = graph["note"].staff.long()
        voice = graph["note"].voice.long()
        durations = graph["note"].duration_div
        onset = graph["note"].onset_div
        pitches = graph["note"].pitch
        onset_beats = graph["note"].onset_beat
        duration_beats = graph["note"].duration_beat
        ts_beats = graph["note"].ts_beats
        na = torch.vstack((onset_beats, onset, durations, duration_beats, pitches)).t()
        edge_attr_dict = self.create_edge_attr(na, edge_index_dict)
        # which potential edges are in the truth edges
        truth_edges_mask = isin_pairwise(pot_edges, truth_edges, assume_unique=True)
        edge_pred_mask_logits, staff_pred_logits, features, pooling_mask_logits = self.module(
            x_dict, edge_index_dict, pot_edges, pot_chord_edges, gbatch, onset, durations, pitches, onset_beats,
            duration_beats, ts_beats,
            edge_attr_dict=edge_attr_dict
        )
        voice_loss = self.voice_loss(edge_pred_mask_logits.float(), truth_edges_mask.float())
        staff_loss = self.staff_loss(staff_pred_logits.float(), staff.long())
        # Feature normalization constrains the features to be within the unit sphere
        feature_normalization_loss = features.pow(2).mean()
        if self.chord_pooling_mode != "none":
            pooling_mask = isin_pairwise(pot_chord_edges, truth_chord_edges, assume_unique=True)
            pooling_loss = self.pooling_loss(pooling_mask_logits, pooling_mask.float())
            loss = voice_loss + staff_loss + pooling_loss + feature_normalization_loss * self.feat_norm_scale
        else:
            loss = voice_loss + staff_loss + feature_normalization_loss * self.feat_norm_scale

        edge_pred_mask_prob = torch.sigmoid(edge_pred_mask_logits)
        self.log(f"{step_type}_voice_loss", voice_loss.item(), on_step=False, on_epoch=True, prog_bar=False,
                 batch_size=graph.num_graphs)
        self.log(f"{step_type}_staff_loss", staff_loss.item(), on_step=False, on_epoch=True, prog_bar=False,
                 batch_size=graph.num_graphs)
        if self.chord_pooling_mode != "none" and step_type == "train":
            self.log(f"{step_type}_pooling_loss", pooling_loss.item(), on_step=False, on_epoch=True, prog_bar=False,
                     batch_size=graph.num_graphs)
        self.log(f"{step_type}_feature_normalization_loss", feature_normalization_loss.item(), on_step=False, on_epoch=True,
                 prog_bar=False, batch_size=graph.num_graphs)
        self.log(f"{step_type}_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=False, batch_size=graph.num_graphs)
        # self.log(f"{step_type}_voice_f1", self.voice_f1(edge_pred_mask_prob, truth_edges_mask.long()), on_step=False,
        #          on_epoch=True, prog_bar=False, batch_size=graph.num_graphs)
        self.log(f"{step_type}_staff_f1", self.staff_f1(staff_pred_logits, staff), on_step=False, on_epoch=True,
                 prog_bar=False, batch_size=graph.num_graphs)
        self.log(f"{step_type}_voice_acc", self.voice_acc(edge_pred_mask_prob, truth_edges_mask.long()), on_step=False,
                 on_epoch=True, prog_bar=False, batch_size=graph.num_graphs)
        self.log(f"{step_type}_staff_acc", self.staff_acc(staff_pred_logits, staff), on_step=False, on_epoch=True,
                 prog_bar=False, batch_size=graph.num_graphs)
        if step_type == "train":
            return loss
        else:
            num_nodes = len(graph.x_dict["note"])
            pred_edges = pot_edges[:, edge_pred_mask_prob > self.threshold]
            f1 = compute_voice_f1_score(pred_edges, truth_edges, num_nodes)
            self.log(f"{step_type}_voice_f1", f1.item(),
                     on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
            new_edge_index, new_edge_probs, unpool_info, reduced_num_nodes = self.ps_pool(
                pot_edges, edge_pred_mask_prob, pot_chord_edges, torch.sigmoid(pooling_mask_logits),
                batch=gbatch, num_nodes=num_nodes
                )
            post_monophonic_edges = linear_assignment(new_edge_probs, new_edge_index, reduced_num_nodes, threshold=self.threshold)
            post_pred_edges = self.ps_pool.unpool(post_monophonic_edges, reduced_num_nodes, unpool_info)
            f1_post = compute_voice_f1_score(post_pred_edges, truth_edges, num_nodes)
            self.log(f"{step_type}_post_processing_f1", f1_post.item(),
                     on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        return loss, pooling_mask_logits, pot_chord_edges, truth_chord_edges

    def training_step(self, batch, batch_idx, **kwargs):
        """
        Training step.

        Parameters
        ----------
        batch : dict
            Input batch data.
        batch_idx : int
            Batch index.

        Returns
        -------
        loss : torch.Tensor
            Computed loss.
        """
        graph = batch
        loss = self._common_step(graph, step_type="train", **kwargs)
        return loss

    def validation_step(self, batch, batch_idx, **kwargs):
        """
        Validation step.

        Parameters
        ----------
        batch : dict
            Input batch data.
        batch_idx : int
            Batch index.
        """
        graph = batch[0]
        loss, pooling_mask_logits, pot_chord_edges, truth_chord_edges = self._common_step(graph, step_type="val", **kwargs)
        if self.chord_pooling_mode != "none":
            pooling_mask_prob = torch.sigmoid(pooling_mask_logits)
            truth_pooling_mask = isin_pairwise(pot_chord_edges, truth_chord_edges, assume_unique=True)
            self.log("val_chord_f1", self.chord_f1(pooling_mask_prob, truth_pooling_mask.long()), on_step=False, on_epoch=True, prog_bar=False, batch_size=1)

    def test_step(self, batch, batch_idx, **kwargs):
        """
        Test step.

        Parameters
        ----------
        batch : dict
            Input batch data.
        batch_idx : int
            Batch index.
        """
        graph = batch[0]
        loss, pooling_mask_logits, pot_chord_edges, truth_chord_edges = self._common_step(
            graph, step_type="test", **kwargs)
        if self.chord_pooling_mode != "none":
            pooling_mask_prob = torch.sigmoid(pooling_mask_logits)
            truth_pooling_mask = isin_pairwise(pot_chord_edges, truth_chord_edges, assume_unique=True)
            self.log("test_chord_f1", self.chord_f1(pooling_mask_prob, truth_pooling_mask.long()),
                     on_step=False, on_epoch=True, prog_bar=False, batch_size=1)

    def predict_step(self, graph, **kwargs):
        """
        Prediction step.

        Parameters
        ----------
        graph : dict
            Input graph data.

        Returns
        -------
        post_pred_edges : torch.Tensor
            Predicted edges.
        staff_pred_logits : torch.Tensor
            Predicted staff logits.
        """
        if self.rev_edges is not None:
            add_reverse_edges(graph, mode=self.rev_edges)
        edge_index_dict = graph.edge_index_dict
        gbatch = torch.zeros(len(graph.x_dict["note"]), dtype=torch.long, device=graph.x_dict["note"].device)
        x_dict = graph.x_dict
        pot_edges = edge_index_dict.pop(("note", "potential", "note"))
        pot_chord_edges = edge_index_dict.pop(("note", "chord_potential", "note"))
        durations = graph["note"].duration_div
        onset = graph["note"].onset_div
        pitches = graph["note"].pitch
        onset_beats = graph["note"].onset_beat
        duration_beats = graph["note"].duration_beat
        ts_beats = graph["note"].ts_beats
        na = torch.vstack((onset_beats, onset, durations, duration_beats, pitches)).t()
        edge_attr_dict = self.create_edge_attr(na, edge_index_dict)
        # which potential edges are in the truth edges
        edge_pred_mask_logits, staff_pred_logits, features, pooling_mask_logits = self.module(
            x_dict, edge_index_dict, pot_edges, pot_chord_edges, gbatch, onset, durations, pitches, onset_beats,
            duration_beats, ts_beats,
            edge_attr_dict=edge_attr_dict)
        edge_pred_mask_prob = torch.sigmoid(edge_pred_mask_logits)
        num_nodes = len(graph.x_dict["note"])
        new_edge_index, new_edge_probs, unpool_info, reduced_num_nodes = self.ps_pool(
            pot_edges, edge_pred_mask_prob, pot_chord_edges, torch.sigmoid(pooling_mask_logits),
            batch=gbatch, num_nodes=num_nodes)
        post_monophonic_edges = linear_assignment(new_edge_probs, new_edge_index, reduced_num_nodes, threshold=self.threshold)
        post_pred_edges = self.ps_pool.unpool(post_monophonic_edges, reduced_num_nodes, unpool_info)
        return_graph = kwargs.get("return_graph", False)
        if return_graph:
            graph["note", "chord_potential", "note"].edge_index = pot_chord_edges
            graph["note", "potential", "note"].edge_index = pot_edges
            graph["note", "chord_predicted", "note"].edge_index = pot_chord_edges[:, torch.sigmoid(pooling_mask_logits) > self.threshold]
            graph["note", "predicted", "note"].edge_index = post_pred_edges
            post_pred_edges = torch.cat(
                (post_pred_edges, pot_chord_edges[:, torch.sigmoid(pooling_mask_logits) > self.threshold]), dim=1)
            return post_pred_edges, staff_pred_logits.argmax(dim=1).long(), graph
        post_pred_edges = torch.cat(
            (post_pred_edges, pot_chord_edges[:, torch.sigmoid(pooling_mask_logits) > self.threshold]), dim=1)
        # add the chord edges to the post_pred_edges
        return post_pred_edges, staff_pred_logits.argmax(dim=1).long()

    def create_edge_attr(self, na, edge_index_dict):
        """
        Create edge attributes.

        Parameters
        ----------
        na : torch.Tensor
            Node attributes.
        edge_index_dict : dict
            Edge index dictionary.

        Returns
        -------
        edge_attr_dict : dict
            Edge attribute dictionary.
        """
        edge_attr_dict = {}
        for key, value in edge_index_dict.items():
            new_v = na[value[0]] - na[value[1]]
            new_v_pitch = self.pitch_embedding(
                torch.remainder(new_v[:, -1], 12).long())
            new_v = F.normalize(torch.abs(new_v), dim=0)
            edge_attr_dict[key] = torch.cat([new_v, new_v_pitch], dim=-1)
        return edge_attr_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, eps=1e-4)
        return {
            "optimizer": optimizer,
        }



class AlgorithmicVoiceSeparationModel(LightningModule):
    """Lightning Model for running the algorithmic approach we use as baseline."""
    def __init__(self):
        super().__init__()

    def training_step(self, batch, batch_idx, **kwargs):
        pass

    def validation_step(self, batch, batch_idx, **kwargs):
        pass

    def test_step(self, batch, batch_idx, **kwargs):
        graph = batch[0]
        pred_edges, pred_staff, voice_f1, staff_f1, chord_f1 = infer_vocstaff_algorithm(graph)
        # curate nan values
        voice_f1 = 0 if np.isnan(voice_f1) else voice_f1
        staff_acc = 0 if np.isnan(staff_f1) else staff_f1
        chord_f1 = 0 if np.isnan(chord_f1) else chord_f1
        self.log("test_voice_f1", voice_f1, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.log("test_staff_acc", staff_acc, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.log("test_chord_f1", chord_f1, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)

    def predict_step(self, graph, **kwargs):
        pred_edges, pred_staff = infer_vocstaff_algorithm(graph, return_score=False)
        return pred_edges, pred_staff
    

def infer_vocstaff_algorithm(graph, return_score=True, normalize=True, return_graph=False):
    """
    This function infers voice and staff assignments with an algorithmic approach, instead of a neural network.
    We use this as baseline.

    Parameters:
    ----------
    graph : torch_geometric.data.HeteroData
        A musical graph where each node represents a note and edges represent potential voice connections between notes.
    return_score : (bool, optional)
        Determines whether the function should return the F1 scores for the voice, staff, and chord predictions.
        Default is True.

    Returns:
    ----------
    pred_edges_voice: torch.Tensor
        A tensor of shape (2, N) where N is the number of predicted voice edges.
        Each column in the tensor represents an edge, with the first row being the source node and
        the second row being the target node.
    staff_pred: torch.Tensor
        A tensor of the same length as the number of notes in the graph.
        It contains the predicted staff assignment for each note.

    If return_score is True, the function also returns:
    voice_f1: float
        The F1 score for the voice predictions.
    staff_f1 : float
        The F1 score for the staff predictions.
    chord_f1 : float
        The F1 score for the chord predictions.
    """
    ps_pool = PostProcessPooling(threshold=0.01)
    edge_index_dict = graph.edge_index_dict
    pot_edges = edge_index_dict.pop(("note", "potential", "note"))
    pot_chord_edges = edge_index_dict.pop(("note", "chord_potential", "note"))
    staff = graph["note"].staff.long()
    pitches = graph["note"].pitch
    batch = torch.zeros_like(pitches, dtype=torch.long)
    num_nodes = len(pitches)
    onset_beats = graph["note"].onset_beat
    duration_beats = graph["note"].duration_beat
    offset_beats = onset_beats + duration_beats
    # split staffs on pitch 60
    staff_pred = torch.zeros_like(pitches)
    staff_pred[pitches < 60] = 1
    # NOTE: need to trim the chord edges to keep highest.
    pred_edges = list()
    pred_chord_edges = list()
    for i in range(2):
        staff_idx = torch.where(staff_pred == i)[0]
        staff_pot_edges = pot_edges[:, torch.isin(pot_edges[0], staff_idx) & torch.isin(pot_edges[1], staff_idx)]
        onset_score = torch.abs(
            onset_beats[staff_pot_edges[1]] - offset_beats[staff_pot_edges[0]])
        pitch_score = torch.abs(pitches[staff_pot_edges[1]] - pitches[staff_pot_edges[0]])
        # normalize the scores between 0 and 1
        if normalize:
            onset_score = 1 - (onset_score - onset_score.min()) / (onset_score.max() - onset_score.min() + 1e-8)
            pitch_score = 1 - (pitch_score - pitch_score.min()) / (pitch_score.max() - pitch_score.min() + 1e-8)
        staff_pot_score = onset_score * pitch_score
        staff_pot_chord_edges = pot_chord_edges[:, torch.isin(pot_chord_edges[0], staff_idx) & torch.isin(pot_chord_edges[1], staff_idx)]
        staff_pot_chord_score = torch.ones(staff_pot_chord_edges.shape[1])
        new_edge_index, new_edge_probs, unpool_info, reduced_num_nodes = ps_pool(
            staff_pot_edges, staff_pot_score, staff_pot_chord_edges, staff_pot_chord_score, batch, num_nodes)
        post_monophonic_edges = linear_assignment(new_edge_probs, new_edge_index, reduced_num_nodes, threshold=0.01)
        post_pred_edges = ps_pool.unpool(post_monophonic_edges, reduced_num_nodes, unpool_info)
        pred_edges.append(post_pred_edges)
        pred_chord_edges.append(staff_pot_chord_edges)

    pred_chord_edges = torch.cat(pred_chord_edges, dim=-1)
    pred_edges_voice = torch.cat(pred_edges, dim=-1)

    if return_score:
        # sort the predicted chord edges
        pred_chord_edges = pred_chord_edges[:, torch.argsort(pred_chord_edges[0])]
        truth_edges = edge_index_dict.pop(("note", "truth", "note"))
        voice_f1 = compute_voice_f1_score(pred_edges_voice, truth_edges, num_nodes).item()
        truth_chord_edges = edge_index_dict.pop(("note", "chord_truth", "note"))
        staff_metric = Accuracy(task="multiclass", num_classes=2).to(pred_edges_voice.device)
        staff_acc = staff_metric(staff_pred, staff).item()
        chord_f1 = compute_voice_f1_score(pred_chord_edges, truth_chord_edges, num_nodes).item()
        return pred_edges_voice, staff_pred, voice_f1, staff_acc, chord_f1
    # add the chord edges to the pred_edges_voice
    pred_edges_voice = torch.cat((pred_edges_voice, pred_chord_edges), dim=-1)
    # sort the predicted voice edges
    pred_edges_voice = pred_edges_voice[:, torch.argsort(pred_edges_voice[0])]
    if return_graph:
        graph["note", "chord_potential", "note"].edge_index = pot_chord_edges
        graph["note", "potential", "note"].edge_index = pot_edges
        graph["note", "chord_predicted", "note"].edge_index = pred_chord_edges
        graph["note", "predicted", "note"].edge_index = pred_edges_voice
        return pred_edges_voice, staff_pred, graph
    return pred_edges_voice, staff_pred
