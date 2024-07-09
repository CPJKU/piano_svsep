from .lightning_base import VocSepLightningModule
from .VoicePred import LinkPredictionModel, HeteroLinkPredictionModel, MetricalLinkPredictionModel
from torch.nn import functional as F
from struttura.models.core import UNet
import torch
import torch_geometric as pyg
from pytorch_lightning import LightningModule
from struttura.utils.pianoroll import pr_to_voice_pred
from struttura.metrics.slow_eval import AverageVoiceConsistency, MonophonicVoiceF1
from .VoicePredPoly import PolyphonicLinkPredictionModel, PostProcessPooling
from .VoicePredGeom import PGLinkPredictionModel
from piano_svsep.utils import add_reverse_edges, add_reverse_edges_from_edge_index
from torchmetrics import F1Score, Accuracy
from scipy.sparse.csgraph import connected_components, min_weight_full_bipartite_matching
from scipy.sparse import coo_matrix
from scipy.optimize import linear_sum_assignment
from torch_scatter import scatter_add
import numpy as np


class VoiceLinkPredictionModel(VocSepLightningModule):
    def __init__(
        self,
        in_feats,
        n_hidden,
        n_layers,
        activation=F.relu,
        dropout=0.5,
        lr=0.001,
        weight_decay=5e-4,
        linear_assignment=True,
        model="ResConv",
        jk=True,
        reg_loss_weight="auto"
    ):
        super(VoiceLinkPredictionModel, self).__init__(
            in_feats,
            n_hidden,
            n_layers,
            activation,
            dropout,
            lr,
            weight_decay,
            LinkPredictionModel,
            linear_assignment=linear_assignment,
            model_name=model,
            jk=jk,
            reg_loss_weight=reg_loss_weight
        )

    def training_step(self, batch, batch_idx):
        batch_inputs, edges, batch_labels, edge_types, pot_edges, truth_edges, na, name = batch
        edges, edge_types = add_reverse_edges_from_edge_index(edges, edge_types)
        pos_edges = pot_edges[:, batch_labels.bool()]
        neg_labels = torch.where(~batch_labels.bool())[0]
        neg_edges = pot_edges[
            :, neg_labels[torch.randperm(len(neg_labels))][: pos_edges.shape[1]]
        ]
        h = self.module.embed(batch_inputs, edges)
        pos_pitch_score = self.pitch_score(pos_edges, na[:, 0])
        pos_onset_score = self.onset_score(pos_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        neg_pitch_score = self.pitch_score(neg_edges, na[:, 0])
        neg_onset_score = self.onset_score(neg_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        pitch_score = self.pitch_score(pot_edges, na[:, 0])
        onset_score = self.onset_score(pot_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        pos_out = self.module.predict(h, pos_edges, pos_pitch_score, pos_onset_score)
        neg_out = self.module.predict(h, neg_edges, neg_pitch_score, neg_onset_score)
        reg_loss = self.reg_loss(
            pot_edges, self.module.predict(h, pot_edges, pitch_score, onset_score), pos_edges, len(batch_inputs))
        batch_pred = torch.cat((pos_out, neg_out), dim=0)
        loss = self.train_loss(pos_out, neg_out)
        batch_pred = torch.cat((1 - batch_pred, batch_pred), dim=1).squeeze()
        targets = (
            torch.cat(
                (torch.ones(pos_out.shape[0]), torch.zeros(neg_out.shape[0])), dim=0
            )
            .long()
            .to(self.device)
        )
        self.log("train_regloss", reg_loss.item(), on_step=True, on_epoch=True, prog_bar=False, batch_size=1)
        self.log("regloss_weight", self.reg_loss_weight, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.log("regloss_weighted", self.reg_loss_weight*reg_loss.item(), on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.train_metric_logging_step(loss, batch_pred, targets)
        loss = loss + self.reg_loss_weight * reg_loss
        self.log("train_joinloss", loss.item(), on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_inputs, edges, batch_labels, edge_types, pot_edges, truth_edges, na, name = batch
        edges, edge_types = add_reverse_edges_from_edge_index(edges, edge_types)
        pitch_score = self.pitch_score(pot_edges, na[:, 0])
        onset_score = self.onset_score(pot_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        batch_pred = self.module(pot_edges, batch_inputs, edges, pitch_score, onset_score)
        self.val_metric_logging_step(
            batch_pred, pot_edges, truth_edges, len(batch_inputs)
        )

    def test_step(self, batch, batch_idx):
        batch_inputs, edges, batch_labels, edge_types, pot_edges, truth_edges, na, name = batch
        edges, edge_types = add_reverse_edges_from_edge_index(edges, edge_types)
        pitch_score = self.pitch_score(pot_edges, na[:, 0])
        onset_score = self.onset_score(pot_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        batch_pred = self.module(pot_edges, batch_inputs, edges, pitch_score, onset_score)
        self.test_metric_logging_step(
            batch_pred, pot_edges, truth_edges, len(batch_inputs)
        )

    def compute_linkpred_loss(self, pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
        )
        w_coef = pos_score.shape[0] / neg_score.shape[0]
        weight = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.ones(neg_score.shape[0]) * w_coef]
        )
        return F.binary_cross_entropy(scores.squeeze(), labels, weight=weight)



class HeteroVoiceLinkPredictionModel(VocSepLightningModule):
    def __init__(
        self,
        in_feats,
        n_hidden,
        n_layers,
        activation=F.relu,
        dropout=0.5,
        lr=0.001,
        weight_decay=5e-4,
        linear_assignment=True,
        model="ResConv",
        jk=True,
        reg_loss_weight="auto",
        reg_loss_type="la",
        tau=0.5
    ):
        super(HeteroVoiceLinkPredictionModel, self).__init__(
            in_feats,
            n_hidden,
            n_layers,
            activation,
            dropout,
            lr,
            weight_decay,
            HeteroLinkPredictionModel,
            linear_assignment=linear_assignment,
            model_name=model,
            jk=jk,
            reg_loss_weight=reg_loss_weight,
            reg_loss_type=reg_loss_type,
            tau=tau
        )

    def training_step(self, batch, batch_idx):
        batch_inputs, edges, batch_labels, edge_types, pot_edges, truth_edges, na, name = batch
        edges, edge_types = add_reverse_edges_from_edge_index(edges, edge_types)
        pos_edges = pot_edges[:, batch_labels.bool()]
        neg_labels = torch.where(~batch_labels.bool())[0]
        neg_edges = pot_edges[
            :, neg_labels[torch.randperm(len(neg_labels))][: pos_edges.shape[1]]
        ]
        h = self.module.embed(batch_inputs, edges, edge_types)
        pos_pitch_score = self.pitch_score(pos_edges, na[:, 0])
        pos_onset_score = self.onset_score(pos_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        neg_pitch_score = self.pitch_score(neg_edges, na[:, 0])
        neg_onset_score = self.onset_score(neg_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        pitch_score = self.pitch_score(pot_edges, na[:, 0])
        onset_score = self.onset_score(pot_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        pos_out = self.module.predict(h, pos_edges, pos_pitch_score, pos_onset_score)
        neg_out = self.module.predict(h, neg_edges, neg_pitch_score, neg_onset_score)
        reg_loss = self.reg_loss(
            pot_edges, self.module.predict(h, pot_edges, pitch_score, onset_score), pos_edges, len(batch_inputs))
        batch_pred = torch.cat((pos_out, neg_out), dim=0)
        loss = self.train_loss(pos_out, neg_out)
        batch_pred = torch.cat((1 - batch_pred, batch_pred), dim=1).squeeze()
        targets = (
            torch.cat(
                (torch.ones(pos_out.shape[0]), torch.zeros(neg_out.shape[0])), dim=0
            )
            .long()
            .to(self.device)
        )
        self.log("train_regloss", reg_loss.item(), on_step=True, on_epoch=True, prog_bar=False, batch_size=1)
        self.log("regloss_weight", self.reg_loss_weight, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.log("regloss_weighted", self.reg_loss_weight*reg_loss.item(), on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.train_metric_logging_step(loss, batch_pred, targets)
        loss = loss + self.reg_loss_weight * reg_loss
        self.log("train_joinloss", loss.item(), on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_inputs, edges, batch_labels, edge_types, pot_edges, truth_edges, na, name = batch
        edges, edge_types = add_reverse_edges_from_edge_index(edges, edge_types)
        pitch_score = self.pitch_score(pot_edges, na[:, 0])
        onset_score = self.onset_score(pot_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        batch_pred = self.module(pot_edges, batch_inputs, edges, edge_types, pitch_score, onset_score)
        self.val_metric_logging_step(
            batch_pred, pot_edges, truth_edges, len(batch_inputs)
        )

    def test_step(self, batch, batch_idx):
        batch_inputs, edges, batch_labels, edge_types, pot_edges, truth_edges, na, name = batch
        edges, edge_types = add_reverse_edges_from_edge_index(edges, edge_types)
        pitch_score = self.pitch_score(pot_edges, na[:, 0])
        onset_score = self.onset_score(pot_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        batch_pred = self.module(pot_edges, batch_inputs, edges, edge_types, pitch_score, onset_score)
        self.test_metric_logging_step(
            batch_pred, pot_edges, truth_edges, len(batch_inputs)
        )

    def predict_step(self, batch, batch_idx):
        batch_inputs, edges, batch_labels, edge_types, pot_edges, truth_edges, na, name = batch
        edges, edge_types = add_reverse_edges_from_edge_index(edges, edge_types)
        pitch_score = self.pitch_score(pot_edges, na[:, 0])
        onset_score = self.onset_score(pot_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        batch_pred = self.module(pot_edges, batch_inputs, edges, edge_types, pitch_score, onset_score)
        adj_pred, fscore = self.predict_metric_step(
            batch_pred, pot_edges, truth_edges, len(batch_inputs)
        )
        print(f"Piece {name} F-score: {fscore}")
        nov_pred, voices_pred = connected_components(csgraph=adj_pred, directed=False, return_labels=True)
        adj_target = pyg.utils.to_dense_adj(truth_edges, max_num_nodes=len(batch_inputs)).squeeze().long().cpu()
        nov_target, voices_target = connected_components(csgraph=adj_target, directed=False, return_labels=True)
        return (
            name,
            voices_pred,
            voices_target,
            nov_pred,
            nov_target,
            na[:, 1],
            na[:, 2],
            na[:, 0],
        )

    def compute_linkpred_loss(self, pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
        )
        w_coef = pos_score.shape[0] / neg_score.shape[0]
        weight = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.ones(neg_score.shape[0]) * w_coef]
        )
        return F.binary_cross_entropy(scores.squeeze(), labels, weight=weight)



class MetricalVoiceLinkPredictionModel(VocSepLightningModule):
    def __init__(
        self,
        in_feats,
        n_hidden,
        n_layers,
        activation=F.relu,
        dropout=0.5,
        lr=0.001,
        weight_decay=5e-4,
        **kwargs
    ):
        super(MetricalVoiceLinkPredictionModel, self).__init__(
            in_feats,
            n_hidden,
            n_layers,
            activation,
            dropout,
            lr,
            weight_decay,
            MetricalLinkPredictionModel,
            **kwargs
        )
        pitch_embedding = kwargs.get("pitch_embedding", None)
        self.pitch_embedding = torch.nn.Embedding(12, 16) if pitch_embedding is not None else pitch_embedding

    def training_step(self, batch, batch_idx):
        edges, edge_types = add_reverse_edges_from_edge_index(batch["edge_index"], batch["edge_type"])
        pos_edges = batch["potential_edges"][:, batch["y"].bool()]
        neg_labels = torch.where(~batch["y"].bool())[0]
        neg_edges = batch["potential_edges"][
            :, neg_labels[torch.randperm(len(neg_labels))][: pos_edges.shape[1]]
        ]
        na = batch["note_array"]
        edge_features = F.normalize(torch.abs(na[:, :5][edges[0]] - na[:, :5][edges[1]]), dim=0) if self.use_reledge else None
        if self.pitch_embedding is not None and edge_features is not None:
            pitch = self.pitch_embedding(torch.remainder(na[:, 0][edges[0]] - na[:, 0][edges[1]], 12).long())
            edge_features = torch.cat([edge_features, pitch], dim=1)
        beat_nodes = batch["beat_nodes"] if "beat_nodes" in batch.keys() else None
        measure_nodes = batch["measure_nodes"] if "measure_nodes" in batch.keys() else None
        beat_edges = batch["beat_edges"] if "beat_edges" in batch.keys() else None
        measure_edges = batch["measure_edges"] if "measure_edges" in batch.keys() else None
        beat_lengths = batch["beat_lengths"] if "beat_lengths" in batch.keys() else None
        measure_lengths = batch["measure_lengths"] if "measure_lengths" in batch.keys() else None
        h = self.module.embed(batch["x"], edges, edge_types,
                              beat_nodes=beat_nodes, measure_nodes=measure_nodes,
                              beat_edges=beat_edges, measure_edges=measure_edges, rel_edge=edge_features,
                              beat_lengths=beat_lengths, measure_lengths=measure_lengths)
        pos_pitch_score = self.pitch_score(pos_edges, na[:, 0])
        pos_onset_score = self.onset_score(pos_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        neg_pitch_score = self.pitch_score(neg_edges, na[:, 0])
        neg_onset_score = self.onset_score(neg_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        pitch_score = self.pitch_score(batch["potential_edges"], na[:, 0])
        onset_score = self.onset_score(batch["potential_edges"], na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        pos_out = self.module.predict(h, pos_edges, pos_pitch_score, pos_onset_score)
        neg_out = self.module.predict(h, neg_edges, neg_pitch_score, neg_onset_score)
        reg_loss = self.reg_loss(
            batch["potential_edges"], self.module.predict(h, batch["potential_edges"],
                                                          pitch_score, onset_score), pos_edges, len(batch["x"]))
        batch_pred = torch.cat((pos_out, neg_out), dim=0)
        loss = self.train_loss(pos_out, neg_out)
        batch_pred = torch.cat((1 - batch_pred, batch_pred), dim=1).squeeze()
        targets = (
            torch.cat(
                (torch.ones(pos_out.shape[0]), torch.zeros(neg_out.shape[0])), dim=0
            )
            .long()
            .to(self.device)
        )
        self.log("train_regloss", reg_loss.item(), on_step=True, on_epoch=True, prog_bar=False, batch_size=1)
        self.log("regloss_weight", self.reg_loss_weight, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.log("regloss_weighted", self.reg_loss_weight*reg_loss.item(), on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.train_metric_logging_step(loss, batch_pred, targets)
        loss = loss + self.reg_loss_weight * reg_loss
        self.log("train_joinloss", loss.item(), on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx):
        edges, edge_types = add_reverse_edges_from_edge_index(batch["edge_index"], batch["edge_type"])
        na = batch["note_array"]
        edge_features = F.normalize(torch.abs(na[:, :5][edges[0]] - na[:, :5][edges[1]]),
                                    dim=0) if self.use_reledge else None
        if self.pitch_embedding is not None and edge_features is not None:
            pitch = self.pitch_embedding(torch.remainder(na[:, 0][edges[0]] - na[:, 0][edges[1]], 12).long())
            edge_features = torch.cat([edge_features, pitch], dim=1)
        beat_nodes = batch["beat_nodes"] if "beat_nodes" in batch.keys() else None
        measure_nodes = batch["measure_nodes"] if "measure_nodes" in batch.keys() else None
        beat_edges = batch["beat_edges"] if "beat_edges" in batch.keys() else None
        measure_edges = batch["measure_edges"] if "measure_edges" in batch.keys() else None
        beat_lengths = batch["beat_lengths"] if "beat_lengths" in batch.keys() else None
        measure_lengths = batch["measure_lengths"] if "measure_lengths" in batch.keys() else None
        pitch_score = self.pitch_score(batch["potential_edges"], na[:, 0])
        onset_score = self.onset_score(batch["potential_edges"], na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        batch_pred = self.module(batch["potential_edges"], batch["x"], edges, edge_types, pitch_score, onset_score,
                                 beat_nodes=beat_nodes, measure_nodes=measure_nodes, beat_edges=beat_edges,
                                 measure_edges=measure_edges, rel_edge=edge_features, beat_lengths=beat_lengths,
                                 measure_lengths=measure_lengths)
        self.val_metric_logging_step(
            batch_pred, batch["potential_edges"], batch["truth_edges"], len(batch["x"])
        )

    def test_step(self, batch, batch_idx):
        edges, edge_types = add_reverse_edges_from_edge_index(batch["edge_index"], batch["edge_type"])
        na = batch["note_array"]
        edge_features = F.normalize(torch.abs(na[:, :5][edges[0]] - na[:, :5][edges[1]]),
                                    dim=0) if self.use_reledge else None
        if self.pitch_embedding is not None and edge_features is not None:
            pitch = self.pitch_embedding(torch.remainder(na[:, 0][edges[0]] - na[:, 0][edges[1]], 12).long())
            edge_features = torch.cat([edge_features, pitch], dim=1)
        beat_nodes = batch["beat_nodes"] if "beat_nodes" in batch.keys() else None
        measure_nodes = batch["measure_nodes"] if "measure_nodes" in batch.keys() else None
        beat_edges = batch["beat_edges"] if "beat_edges" in batch.keys() else None
        measure_edges = batch["measure_edges"] if "measure_edges" in batch.keys() else None
        beat_lengths = batch["beat_lengths"] if "beat_lengths" in batch.keys() else None
        measure_lengths = batch["measure_lengths"] if "measure_lengths" in batch.keys() else None
        pitch_score = self.pitch_score(batch["potential_edges"], na[:, 0])
        onset_score = self.onset_score(batch["potential_edges"], na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        batch_pred = self.module(batch["potential_edges"], batch["x"], edges, edge_types, pitch_score, onset_score,
                                 beat_nodes=beat_nodes, measure_nodes=measure_nodes, beat_edges=beat_edges,
                                 measure_edges=measure_edges, rel_edge=edge_features, beat_lengths=beat_lengths,
                                 measure_lengths=measure_lengths)
        self.test_metric_logging_step(
            batch_pred, batch["potential_edges"], batch["truth_edges"], len(batch["x"])
        )

    def predict_step(self, batch, batch_idx):
        batch_inputs, edges, batch_labels, edge_types, pot_edges, truth_edges, na, name, beat_nodes, beat_index, measure_nodes, measure_index = batch
        edges, edge_types = add_reverse_edges_from_edge_index(edges, edge_types)
        pitch_score = self.pitch_score(pot_edges, na[:, 0])
        onset_score = self.onset_score(pot_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        batch_pred = self.module(pot_edges, batch_inputs, edges, edge_types, pitch_score, onset_score, beat_nodes, measure_nodes, beat_index,
                                                              measure_index)
        adj_pred, fscore = self.predict_metric_step(
            batch_pred, pot_edges, truth_edges, len(batch_inputs)
        )
        print(f"Piece {name} F-score: {fscore}")
        nov_pred, voices_pred = connected_components(csgraph=adj_pred, directed=False, return_labels=True)
        adj_target = pyg.utils.to_dense_adj(truth_edges, max_num_nodes=len(batch_inputs)).squeeze().long().cpu()
        nov_target, voices_target = connected_components(csgraph=adj_target, directed=False, return_labels=True)
        return (
            name,
            voices_pred,
            voices_target,
            nov_pred,
            nov_target,
            na[:, 1],
            na[:, 2],
            na[:, 0],
        )

    def compute_linkpred_loss(self, pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
        )
        w_coef = pos_score.shape[0] / neg_score.shape[0]
        weight = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.ones(neg_score.shape[0]) * w_coef]
        )
        return F.binary_cross_entropy(scores.squeeze(), labels, weight=weight)


class VoiceLinkPredictionLightModelPG(VocSepLightningModule):
    def __init__(
        self,
        graph_metadata,
        in_feats,
        n_hidden,
        n_layers=2,
        activation=F.relu,
        dropout=0.5,
        lr=0.001,
        weight_decay=5e-4,
        linear_assignment=False,
        rev_edges=None,
        pos_weight = None,
        jk_mode = "lstm",
        conv_type = "gcn",
    ):
        super(VoiceLinkPredictionLightModelPG, self).__init__(
            in_feats, n_hidden, n_layers, activation, dropout, lr, weight_decay, None, linear_assignment
        )
        self.save_hyperparameters()
        self.module = PGLinkPredictionModel(
            graph_metadata,
            in_feats,
            n_hidden,
            n_layers,
            activation=activation,
            dropout=dropout,
            jk_mode = jk_mode,
            conv_type = conv_type,
        )
        print(f"Graph edge types: {graph_metadata}")
        self.rev_edges = rev_edges
        self.train_loss = torch.nn.BCEWithLogitsLoss(pos_weight= torch.tensor([pos_weight]))
        # self.train_loss_func = F.binary_cross_entropy_with_logits

    def training_step(self, batch, batch_idx):
        graph = batch[0]
        if self.rev_edges is not None:
            add_reverse_edges(graph, mode=self.rev_edges)
        pot_edges = graph["pot_edges"]
        edge_target_mask = graph["truth_edges_mask"]
        onsets = graph["note"].onset_div
        durations = graph["note"].duration_div
        pitches = graph["note"].pitch
        onset_beats = graph["note"].onset_beat
        duration_beats = graph["note"].duration_beat
        ts_beats = graph["note"].ts_beats
        edge_pred_mask_logits = self.module(
            graph.x_dict, graph.edge_index_dict, pot_edges, onsets, durations, pitches, onset_beats, duration_beats, ts_beats
        )
        loss = self.train_loss(edge_pred_mask_logits.float(), edge_target_mask.float())
        # get predicted class for the edges (e.g. 0 or 1)
        edge_pred__mask_normalized = torch.sigmoid(edge_pred_mask_logits)
        edge_pred_mask_bool = torch.round(edge_pred__mask_normalized).bool()
        self.train_metric_logging_step(loss, edge_pred_mask_bool, edge_target_mask)
        return loss

    def validation_step(self, batch, batch_idx):
        graph = batch[0]
        if self.rev_edges is not None:
            add_reverse_edges(graph, mode=self.rev_edges)
        pot_edges = graph["pot_edges"]
        num_notes = len(graph.x_dict["note"])
        edge_target = graph["truth_edges"]
        onsets = graph["note"].onset_div
        durations = graph["note"].duration_div
        pitches = graph["note"].pitch
        onset_beats = graph["note"].onset_beat
        duration_beats = graph["note"].duration_beat
        ts_beats = graph["note"].ts_beats
        edge_pred_mask_logits = self.module(
            graph.x_dict, graph.edge_index_dict, pot_edges, onsets, durations, pitches, onset_beats, duration_beats, ts_beats
        )
        edge_pred__mask_normalized = torch.sigmoid(edge_pred_mask_logits)
        self.val_metric_logging_step(
            edge_pred__mask_normalized, pot_edges, edge_target, num_notes, linear_assignment=self.linear_assignment
        )

    def test_step(self, batch, batch_idx):
        graph = batch[0]
        if self.rev_edges is not None:
            add_reverse_edges(graph, mode=self.rev_edges)
        pot_edges = graph["pot_edges"]
        num_notes = len(graph.x_dict["note"])
        edge_target = graph["truth_edges"]
        onsets = graph["note"].onset_div
        durations = graph["note"].duration_div
        pitches = graph["note"].pitch
        onset_beats = graph["note"].onset_beat
        duration_beats = graph["note"].duration_beat
        ts_beats = graph["note"].ts_beats
        edge_pred_mask_logits = self.module(
            graph.x_dict, graph.edge_index_dict, pot_edges, onsets, durations, pitches, onset_beats, duration_beats, ts_beats
        )
        edge_pred__mask_normalized = torch.sigmoid(edge_pred_mask_logits)
        # log without linear assignment
        self.test_metric_logging_step(
            edge_pred__mask_normalized, pot_edges, edge_target, num_notes
        )
        # log with linear assignment
        self.test_metric_logging_step(
            edge_pred__mask_normalized, pot_edges, edge_target, num_notes
        )

    def predict_step(self, batch, batch_idx):
        graph = batch[0]
        if self.rev_edges is not None:
            add_reverse_edges(graph, mode=self.rev_edges)
        pot_edges = graph["pot_edges"]
        num_notes = len(graph.x_dict["note"])
        edge_target = graph["truth_edges"]
        onsets = graph["note"].onset_div
        durations = graph["note"].duration_div
        pitches = graph["note"].pitch
        onset_beats = graph["note"].onset_beat
        duration_beats = graph["note"].duration_beat
        ts_beats = graph["note"].ts_beats
        edge_pred_mask_logits = self.module(
            graph.x_dict, graph.edge_index_dict, pot_edges, onsets, durations, pitches, onset_beats, duration_beats, ts_beats
        )
        edge_pred__mask_normalized = torch.sigmoid(edge_pred_mask_logits)
        adj_pred, fscore = self.predict_metric_step(
            edge_pred__mask_normalized, pot_edges, edge_target, num_notes
        )
        print(f"Piece {graph['name']} F-score: {fscore}")
        nov_pred, voices_pred = connected_components(csgraph=adj_pred, directed=False, return_labels=True)
        adj_target = pyg.utils.to_dense_adj(edge_target, max_num_nodes=num_notes).squeeze().long().cpu()
        nov_target, voices_target = connected_components(csgraph=adj_target, directed=False, return_labels=True)
        return ( 
            voices_pred, 
            voices_target, 
            nov_pred, 
            nov_target, 
            graph["note"].onset_div,
            graph["note"].duration_div,
            graph["note"].pitch
        )


class PolyphonicVoiceSeparationModel(LightningModule):
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
            edge_feature_feedback=False,
            after_encoder_frontend=False,
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
        self.module = PolyphonicLinkPredictionModel(
            input_features=in_feats, hidden_features=n_hidden, num_layers=n_layers, activation=activation,
            dropout=dropout, conv_type=conv_type, gnn_metadata=self.gnn_metadata, chord_pooling_mode=chord_pooling_mode,
            staff_feedback=staff_feedback, edge_feature_feedback=edge_feature_feedback, after_encoder_frontend=after_encoder_frontend)
        self.rev_edges = rev_edges
        self.threshold = 0.1
        self.chord_pooling_mode = chord_pooling_mode
        self.ps_pool = PostProcessPooling()
        self.pitch_embedding = torch.nn.Embedding(12, 16)
        # train metrics
        self.voice_loss = torch.nn.BCEWithLogitsLoss(pos_weight= torch.tensor([pos_weights["voice"]]))
        self.staff_loss = torch.nn.CrossEntropyLoss()
        self.pooling_loss = torch.nn.BCEWithLogitsLoss(pos_weight= torch.tensor([pos_weights["chord"]]))
        self.staff_f1 = F1Score(task="multiclass", num_classes=2, average="macro")
        self.voice_f1 = F1Score(task="binary")
        self.chord_f1 = F1Score(task="binary")
        self.staff_acc = Accuracy(task="multiclass", num_classes=2)
        self.voice_acc = Accuracy(task="binary")

    def _common_step(self, graph, step_type="train", **kwargs):
        if self.rev_edges is not None:
            add_reverse_edges(graph, mode=self.rev_edges)

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
        # TODO: what should be batch size be?
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
            f1 = self.compute_voice_f1_score(pred_edges, truth_edges, num_nodes)
            self.log(f"{step_type}_voice_f1", f1.item(),
                     on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
            new_edge_index, new_edge_probs, unpool_info, reduced_num_nodes = self.ps_pool(
                pot_edges, edge_pred_mask_prob, pot_chord_edges, torch.sigmoid(pooling_mask_logits),
                batch=gbatch, num_nodes=num_nodes
                )
            post_monophonic_edges = self.linear_assignment(new_edge_probs, new_edge_index, reduced_num_nodes)
            post_pred_edges = self.ps_pool.unpool(post_monophonic_edges, reduced_num_nodes, unpool_info)
            f1_post = self.compute_voice_f1_score(post_pred_edges, truth_edges, num_nodes)
            # which_pred_edges_in_true = isin_pairwise(pred_edges, truth_edges, assume_unique=True)
            # which_true_edges_in_pred = isin_pairwise(truth_edges, pred_edges, assume_unique=True)
            # tp = which_pred_edges_in_true.sum()
            # fp = (~which_true_edges_in_pred).sum()
            # fn = (~which_pred_edges_in_true).sum()
            # f1 = 2 * tp / (2 * tp + fp + fn)
            self.log(f"{step_type}_post_processing_f1", f1_post.item(),
                     on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        return loss, pooling_mask_logits, pot_chord_edges, truth_chord_edges

    def training_step(self, batch, batch_idx, **kwargs):
        graph = batch
        loss = self._common_step(graph, step_type="train", **kwargs)
        return loss

    def validation_step(self, batch, batch_idx, **kwargs):
        graph = batch[0]
        loss, pooling_mask_logits, pot_chord_edges, truth_chord_edges = self._common_step(graph, step_type="val", **kwargs)
        # TODO: this is problematic Post-processing
        # IDEA: Implement slow version that loops through potential edges, and unpool_version
        # Trim first pot_edges by threshold, document this process (it can be interesting for the paper)

        if self.chord_pooling_mode != "none":
            pooling_mask_prob = torch.sigmoid(pooling_mask_logits)
            truth_pooling_mask = isin_pairwise(pot_chord_edges, truth_chord_edges, assume_unique=True)
            self.log("val_chord_f1", self.chord_f1(pooling_mask_prob, truth_pooling_mask.long()), on_step=False, on_epoch=True, prog_bar=False, batch_size=1)

    def test_step(self, batch, batch_idx, **kwargs):
        graph = batch[0]
        loss, pooling_mask_logits, pot_chord_edges, truth_chord_edges = self._common_step(
            graph, step_type="test", **kwargs)
        if self.chord_pooling_mode != "none":
            pooling_mask_prob = torch.sigmoid(pooling_mask_logits)
            truth_pooling_mask = isin_pairwise(pot_chord_edges, truth_chord_edges, assume_unique=True)
            self.log("test_chord_f1", self.chord_f1(pooling_mask_prob, truth_pooling_mask.long()),
                     on_step=False, on_epoch=True, prog_bar=False, batch_size=1)

    def predict_step(self, batch, batch_idx, **kwargs):
        graph = batch[0]
        if self.rev_edges is not None:
            add_reverse_edges(graph, mode=self.rev_edges)
        gbatch = graph["note"].batch
        edge_index_dict = graph.edge_index_dict
        x_dict = graph.x_dict
        pot_edges = edge_index_dict.pop(("note", "potential", "note"))
        truth_edges = edge_index_dict.pop(("note", "truth", "note"))
        pot_chord_edges = edge_index_dict.pop(("note", "chord_potential", "note"))
        truth_chord_edges = edge_index_dict.pop(("note", "chord_truth", "note"))
        staff = graph["note"].staff.long()
        durations = graph["note"].duration_div
        onset = graph["note"].onset_div
        pitches = graph["note"].pitch
        onset_beats = graph["note"].onset_beat
        duration_beats = graph["note"].duration_beat
        ts_beats = graph["note"].ts_beats
        truth_edges_mask = isin_pairwise(pot_edges, truth_edges, assume_unique=True)
        edge_pred_mask_logits, staff_pred_logits, features, pooling_mask_logits = self.module(
            x_dict, edge_index_dict, pot_edges, pot_chord_edges, gbatch, onset, durations, pitches, onset_beats, duration_beats, ts_beats
        )
        voice_loss = self.voice_loss(edge_pred_mask_logits.float(), truth_edges_mask.float())
        staff_loss = self.staff_loss(staff_pred_logits.float(), staff.float())
        loss = voice_loss + staff_loss
        edge_pred_mask_prob = torch.sigmoid(edge_pred_mask_logits)
        staff_pred_class = torch.sigmoid(staff_pred_logits) > 0.5
        pooling_mask_prob = torch.sigmoid(pooling_mask_logits)
        # Calculation of F1 score
        true_adj = pyg.utils.to_dense_adj(truth_edges, gbatch, max_num_nodes=graph.num_nodes)
        pred_adj = pyg.utils.to_dense_adj(pot_edges, gbatch, max_num_nodes=graph.num_nodes,
                                          edge_attr=edge_pred_mask_prob.squeeze())
        w_true = true_adj.sum(dim=1) + 1e-8
        w_pred = pred_adj.sum(dim=1) + 1e-8
        voice_precision = (true_adj * pred_adj / w_pred).sum() / (pred_adj / w_pred).sum()
        voice_recall = (true_adj * pred_adj / w_true).sum() / (true_adj / w_true).sum()
        f1_score = 2 * voice_precision * voice_recall / (voice_precision + voice_recall)
        if self.chord_pooling_mode != "none":
            pooling_mask_prob = torch.sigmoid(pooling_mask_logits)
            truth_pooling_mask = isin_pairwise(pot_chord_edges, truth_chord_edges, assume_unique=True)
        # save visualization

    def linear_assignment(self, edge_pred_mask_prob, pot_edges, num_notes):
        # Solve with Hungarian Algorithm and then trim predictions.
        row = pot_edges[0].cpu().numpy()
        col = pot_edges[1].cpu().numpy()
        new_probs = edge_pred_mask_prob.clone()
        # cost = edge_pred_mask_prob.max() - edge_pred_mask_prob
        cost = edge_pred_mask_prob.cpu().numpy()
        cost_matrix = coo_matrix((cost, (row, col)), shape=(num_notes, num_notes))
        # Sparse version is not working yet
        # row_ind, col_ind = min_weight_full_bipartite_matching(cost_matrix)
        row_ind, col_ind = linear_sum_assignment(cost_matrix.todense(), maximize=True)
        # numpy function to get [row_ind, col_ind]
        new_edge_index = torch.tensor(np.vstack((row_ind, col_ind))).to(edge_pred_mask_prob.device)
        mask_potential = isin_pairwise(pot_edges, new_edge_index, assume_unique=True)
        new_probs[~mask_potential] = 0
        mask_over_potential = new_probs > self.threshold
        pred_edges = pot_edges[:, mask_over_potential]
        return pred_edges

    def create_edge_attr(self, na, edge_index_dict):
        edge_attr_dict = {}
        for key, value in edge_index_dict.items():
            new_v = na[value[0]] - na[value[1]]
            new_v_pitch = self.pitch_embedding(
                torch.remainder(new_v[:, -1], 12).long())
            new_v = F.normalize(torch.abs(new_v), dim=0)
            edge_attr_dict[key] = torch.cat([new_v, new_v_pitch], dim=-1)
        return edge_attr_dict

    def compute_voice_f1_score(self, pred_edges, truth_edges, num_nodes):
        which_true_edges_in_pred = isin_pairwise(truth_edges, pred_edges, assume_unique=True).float()
        which_pred_edges_in_true = isin_pairwise(pred_edges, truth_edges, assume_unique=True).float()
        ones_pred = torch.ones(pred_edges.shape[1]).to(pred_edges.device)
        ones_true = torch.ones(truth_edges.shape[1]).to(truth_edges.device)
        multiplicity_pred = scatter_add(ones_pred, pred_edges[0], out=torch.zeros(num_nodes).to(pred_edges.device)) + 1e-8
        weights_pred = multiplicity_pred[pred_edges[0]]
        multiplicity_true = scatter_add(ones_true, truth_edges[0], out=torch.zeros(num_nodes).to(truth_edges.device)) + 1e-8
        weights_true = multiplicity_true[truth_edges[0]]
        precision = (which_pred_edges_in_true / weights_pred).sum() / (ones_pred / weights_pred).sum()
        recall = (which_true_edges_in_pred / weights_true).sum() / (ones_true / weights_true).sum()
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, eps=1e-4)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, verbose=False)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": scheduler,
            # "monitor": "val_loss"
        }


class UnetVoiceSeparationModel(LightningModule):
    def __init__(
        self,
        n_classes,
        input_channels=1,
        lr=0.0005,
        weight_decay=5e-4,
    ):
        super(UnetVoiceSeparationModel, self).__init__()
        self.save_hyperparameters()
        self.module = UNet(input_channels, n_classes)
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
        # self.train_f1 = F1Score(num_classes=2, average="macro")
        self.val_avc = AverageVoiceConsistency(allow_permutations=False)
        self.val_monophonic_f1 = MonophonicVoiceF1(num_classes=2, average="macro")

    def training_step(self, batch, batch_idx):
        pr_dict = batch[0]
        voice_pr = pr_dict["voice_pianoroll"].squeeze().T
        input_pr = (
            torch.clip(voice_pr + 1, 0, 1)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device, dtype=torch.float64)
        )
        labels = voice_pr.to(self.device).unsqueeze(0)

        pred = self.module(input_pr)
        loss = self.train_loss(pred, labels)
        # batch_f1 = self.train_f1(batch_pred, batch_labels)
        self.log(
            "train_loss",
            loss.item(),
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            batch_size=1,
            sync_dist=True,
        )
        # self.log("train_f1", batch_f1.item(), prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pr_dict = batch[0]
        voice_pr = pr_dict["voice_pianoroll"].squeeze().T
        input_pr = (
            torch.clip(voice_pr + 1, 0, 1)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device, dtype=torch.float64)
        )
        labels = voice_pr.to(self.device).unsqueeze(0)

        pred = self.module(input_pr)
        loss = self.train_loss(pred, labels)
        self.log(
            "val_loss",
            loss.item(),
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            batch_size=1,
        )
        voice_pred = pr_to_voice_pred(
            F.log_softmax(pred.squeeze(), dim=0),
            pr_dict["notearray_onset_beat"].squeeze(),
            pr_dict["notearray_duration_beat"].squeeze(),
            pr_dict["notearray_pitch"].squeeze(),
            piano_range=True,
            time_div=12,
        )
        voice_pred = voice_pred.to(self.device)
        fscore = self.val_monophonic_f1(
            voice_pred,
            pr_dict["notearray_voice"].squeeze(),
            pr_dict["notearray_onset_beat"].squeeze(),
            pr_dict["notearray_duration_beat"].squeeze(),
        )
        self.log(
            "val_f1",
            fscore.item(),
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            batch_size=1,
            sync_dist=True,
        )
        avc = self.val_avc(voice_pred, pr_dict["notearray_voice"].squeeze())
        self.log(
            "val_avc",
            avc.item(),
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            batch_size=1,
            sync_dist=True,
        )
        # add F1 computation
        return avc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return {
            "optimizer": optimizer,
        }


def isin_pairwise(element,test_elements, assume_unique=True):
    """Like isin function of torch, but every element in the sequence is a pair of integers.
    # TODO: check if this solution can be better https://stackoverflow.com/questions/71708091/is-there-an-equivalent-numpy-function-to-isin-that-works-row-based
    
    Args:
        element (torch.Tensor): Tensor of shape (2, N) where N is the number of elements.
        test_elements (torch.Tensor): Tensor of shape (2, M) where M is the number of elements to test.
        assume_unique (bool, optional): If True, the input arrays are both assumed to be unique, which can speed up the calculation. Defaults to True.
        
        Returns:
            torch.Tensor: Tensor of shape (M,) with boolean values indicating whether the element is in the test_elements.
                        
    """
    def cantor_pairing(x, y):
        return (x + y) * (x + y + 1) // 2 + y

    element_cantor_proj = cantor_pairing(element[0], element[1])
    test_elements_cantor_proj = cantor_pairing(test_elements[0], test_elements[1])
    return torch.isin(element_cantor_proj, test_elements_cantor_proj, assume_unique=assume_unique)

