from torch.nn import functional as F
import torch
from piano_svsep.utils import isin_pairwise, compute_voice_f1_score, linear_assignment, infer_vocstaff_algorithm
from pytorch_lightning import LightningModule
from piano_svsep.models.models import PianoSVSep
from piano_svsep.postprocessing import PostProcessPooling
from piano_svsep.utils import add_reverse_edges
from torchmetrics import F1Score, Accuracy
import numpy as np


class AlgorithmicVoiceSeparationModel(LightningModule):
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


class PLPianoSVSep(LightningModule):
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
        self.module = PianoSVSep(
            input_features=in_feats, hidden_features=n_hidden, num_layers=n_layers, activation=activation,
            dropout=dropout, conv_type=conv_type, gnn_metadata=self.gnn_metadata, chord_pooling_mode=chord_pooling_mode,
            staff_feedback=staff_feedback, edge_feature_feedback=edge_feature_feedback, after_encoder_frontend=after_encoder_frontend)
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
        graph = batch
        loss = self._common_step(graph, step_type="train", **kwargs)
        return loss

    def validation_step(self, batch, batch_idx, **kwargs):
        graph = batch[0]
        loss, pooling_mask_logits, pot_chord_edges, truth_chord_edges = self._common_step(graph, step_type="val", **kwargs)
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

    def predict_step(self, graph, **kwargs):
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
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, verbose=False)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": scheduler,
            # "monitor": "val_loss"
        }

