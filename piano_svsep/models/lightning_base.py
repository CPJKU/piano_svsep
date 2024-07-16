import numpy as np
from pytorch_lightning import LightningModule
import torch
# from struttura.metrics.losses import LinkPredictionLoss, LinearAssignmentLoss, LinearAssignmentLossCE
from torchmetrics import F1Score, Accuracy, Precision, Recall
# from struttura.metrics.slow_eval import MonophonicVoiceF1
# from struttura.metrics.eval import LinearAssignmentScore
import torch_geometric as pyg
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


class VocSepLightningModule(LightningModule):
    """
    This is the Core Lightning Module for logging and computing Voice Separation with GNNs.
    """
    def __init__(self, in_feats, n_hidden, n_layers, activation, dropout, lr, weight_decay, module=None, **kwargs):

        super(VocSepLightningModule, self).__init__()
        self.save_hyperparameters()
        if module is not None:
            self.module = module(in_feats, n_hidden, n_layers, activation, dropout, **kwargs)
        self.lr = lr
        self.weight_decay = weight_decay
        self.threshold = kwargs["tau"] if "tau" in kwargs.keys() else 0.5
        self.use_reledge = kwargs["use_reledge"] if "use_reledge" in kwargs.keys() else False
        self.use_metrical = kwargs["use_metrical"] if "use_metrical" in kwargs.keys() else False
        if kwargs["reg_loss_weight"] == "auto":
            self.reg_loss_type = "auto"
            self.reg_loss_weight = 0.0
        elif kwargs["reg_loss_weight"] == "fixed":
            self.reg_loss_type = "fixed"
            self.reg_loss_weight = 1.0
        elif kwargs["reg_loss_weight"] == "none":
            self.reg_loss_type = "none"
            self.reg_loss_weight = 0.0
        else:
            self.reg_loss_weight = kwargs["reg_loss_weight"] if "reg_loss_weight" in kwargs.keys() else 1.0
        # metrics
        self.train_acc_score = Accuracy()
        self.val_acc_score = Accuracy()
        self.train_loss = LinkPredictionLoss()
        if self.reg_loss_type == "ca":
            self.reg_loss = LinearAssignmentLossCE()
        else:
            self.reg_loss = LinearAssignmentLoss()
        self.val_loss = LinkPredictionLoss()
        self.linear_assignment = kwargs["linear_assignment"] if "linear_assignment" in kwargs.keys() else False
        self.val_f1_score = F1Score(average="macro", num_classes=1).cpu()
        self.val_precision = Precision(average="macro", num_classes=1).cpu()
        self.val_recall = Recall(average="macro", num_classes=1).cpu()
        self.val_monvoicef1 = MonophonicVoiceF1(average="macro", num_classes=2)
        self.test_f1_score = F1Score(average="macro", num_classes=1)
        self.test_precision = Precision(average="macro", num_classes=1)
        self.test_recall = Recall(average="macro", num_classes=1)
        self.test_f1_allignment = F1Score(average="macro", num_classes=1)
        self.test_precision_allignment = Precision(average="macro", num_classes=1)
        self.test_recall_allignment = Recall(average="macro", num_classes=1)
        self.test_monvoicef1 = MonophonicVoiceF1(average="macro", num_classes=2)
        self.test_linear_assignment = LinearAssignmentScore()
        self.val_linear_assignment = LinearAssignmentScore()
        # Alpha and beta are hyperparams from reference for pitch and onset score.
        self.alpha = 3.1
        self.beta = 5

    def training_step(self, *args, **kwargs):
        """To be re-written by child"""
        pass

    def training_epoch_end(self, *args, **kwargs):
        if self.reg_loss_type == "auto":
            self.reg_loss_weight += 0.002

    def validation_step(self, *args, **kwargs):
        """To be re-written by child"""
        pass

    def test_step(self, *args, **kwargs):
        """To be re-written by child"""
        pass

    def predict_step(self, *args, **kwargs):
        """To be re-written by child"""
        pass

    def compute_linkpred_loss(self, pos_score, neg_score):
        """Standard Link Prediction loss with possitive and negative edges, no need for weighting."""
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
        w_coef = pos_score.shape[0] / neg_score.shape[0]
        weight = torch.cat([torch.ones(pos_score.shape[0]), torch.ones(neg_score.shape[0]) * w_coef])
        return F.binary_cross_entropy(scores.squeeze(), labels, weight=weight)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, verbose=False)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }

    def linear_assignment_step(self, batch_pred, pot_edges, target_edges, num_nodes):
        adj_target = pyg.utils.to_dense_adj(target_edges, max_num_nodes=num_nodes).squeeze().long().cpu()
        if self.linear_assignment:
            # Solve with Hungarian Algorithm
            cost_matrix = torch.sparse_coo_tensor(
                pot_edges, batch_pred.squeeze(), (num_nodes, num_nodes)).to_dense().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
            pot_edges = np.vstack((row_ind, col_ind))
            trimmed_preds = torch.tensor(cost_matrix[row_ind, col_ind])
            adj_pred = torch.sparse_coo_tensor(torch.tensor(pot_edges), trimmed_preds,
                                               (num_nodes, num_nodes)).to_dense()
            # Eliminate prediction that do not reach the threshold to fix zero nodes
            adj_pred = (adj_pred > self.threshold).float()
            # adj_pred = torch.round(adj_pred)
        else:
            pred_edges = pot_edges[:, torch.round(batch_pred).squeeze().long()]
            # compute pred and ground truth adj matrices
            if torch.sum(pred_edges) > 0:
                adj_pred = pyg.utils.to_dense_adj(pred_edges, max_num_nodes=num_nodes).squeeze().cpu()
            else:  # to avoid exception in to_dense_adj when there is no predicted edge
                adj_pred = torch.zeros((num_nodes, num_nodes)).squeeze().to(self.device).cpu()
        return adj_pred, adj_target

    def train_metric_logging_step(self, loss, batch_pred, targets):
        """Logging for the training step using the standard loss"""
        acc = self.train_acc_score(batch_pred, targets.long())
        self.log("train_loss", loss.item(), prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log("train_acc", acc.item(), prog_bar=True, on_step=False, on_epoch=True, batch_size=1)

    def val_metric_logging_step(self, batch_pred, pot_edges, target_edges, num_nodes):
        """Logging for validation step
        Args:
            batch_pred: Predicted edges, 1d array of length (num_pot_edges). This is a binary mask over pot_edges.
            pot_edges: Potential edges, shape (2, num_pot_edges). Each column is the start and destination of a potential edge.
            target_edges: Target edges, shape (2, num_edges). Each column is the start and destination of a truth voice edge.
            num_nodes: Number of nodes in the graph, i.e., notes in the score.
        """
        adj_target = pyg.utils.to_dense_adj(target_edges, max_num_nodes=num_nodes).squeeze().long().cpu()
        if self.linear_assignment:
            # Solve with Hungarian Algorithm
            cost_matrix = torch.sparse_coo_tensor(
                pot_edges, batch_pred.squeeze(), (num_nodes, num_nodes)).to_dense().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
            pot_edges = np.vstack((row_ind, col_ind))
            trimmed_preds = torch.tensor(cost_matrix[row_ind, col_ind])
            adj_pred = torch.sparse_coo_tensor(torch.tensor(pot_edges), trimmed_preds, (num_nodes, num_nodes)).to_dense()
            # Eliminate prediction that do not reach the threshold to fix zero nodes
            adj_pred = (adj_pred > self.threshold).float()
            # adj_pred = torch.round(adj_pred)
        else:
            pred_edges = pot_edges[:, torch.round(batch_pred).squeeze().bool()]
            # compute pred and ground truth adj matrices
            if torch.sum(pred_edges) > 0:
                adj_pred = pyg.utils.to_dense_adj(pred_edges, max_num_nodes=num_nodes).squeeze().cpu()
            else: # to avoid exception in to_dense_adj when there is no predicted edge
                adj_pred = torch.zeros((num_nodes, num_nodes)).squeeze().to(self.device).cpu()
        # compute f1 score on the adj matrices
        loss = F.binary_cross_entropy(adj_pred.float(), adj_target.float())
        val_fscore = self.val_f1_score.cpu()(adj_pred.flatten(), adj_target.flatten())
        self.log("val_loss", loss.item(), batch_size=1)
        self.log("val_fscore", val_fscore.item(), prog_bar=True, batch_size=1)
        self.log("val_precision", self.val_precision.cpu()(adj_pred.flatten(), adj_target.flatten()).item(), batch_size=1)
        self.log("val_recall", self.val_recall.cpu()(adj_pred.flatten(), adj_target.flatten()).item(), batch_size=1)
        self.log("val_la_score", self.val_linear_assignment.cpu()(pot_edges, batch_pred, target_edges, num_nodes).item(), batch_size=1)

        # TODO compute monophonic voice f1 score
        # y_pred, n_voc = voice_from_edges(pred_edges, num_nodes)
        # val_monf1 = self.val_monvoicef1(torch.tensor(y_pred, device=self.device), graph["note"].y, onset=torch.tensor(graph["note"].onset_div[0], device=self.device), duration=torch.tensor(graph["note"].duration_div[0], device=self.device))
        # self.log("val_monvoicef1", val_monf1.item(),prog_bar = True, on_epoch = True, batch_size=1)

    def test_metric_logging_step(self, batch_pred, pot_edges, target_edges, num_nodes):
        """Test logging only done once, similar to validation.
        See val_metric_logging_step for details."""
        batch_pred = batch_pred.cpu()
        pot_edges = pot_edges.cpu()
        target_edges = target_edges.cpu()

        adj_target = pyg.utils.to_dense_adj(target_edges, max_num_nodes=num_nodes).squeeze().long().cpu()

        # Solve with Hungarian Algorithm
        cost_matrix = torch.sparse_coo_tensor(
            pot_edges, batch_pred.squeeze(), (num_nodes, num_nodes)).to_dense().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
        potential_edges = np.vstack((row_ind, col_ind))
        trimmed_preds = torch.tensor(cost_matrix[row_ind, col_ind])
        adj_pred = torch.sparse_coo_tensor(torch.tensor(potential_edges), trimmed_preds, (num_nodes, num_nodes)).to_dense()
        # Eliminate prediction that do not reach the threshold to fix zero nodes
        adj_pred = (adj_pred > self.threshold).float()
        test_la = self.test_f1_allignment.cpu()(adj_pred.flatten(), adj_target.flatten())
        self.log("test_fscore_la", test_la.item(), prog_bar=True, batch_size=1)
        self.log("test_precision_la", self.test_precision_allignment.cpu()(adj_pred.flatten(), adj_target.flatten()).item(), batch_size=1)
        self.log("test_recall_la", self.test_recall_allignment.cpu()(adj_pred.flatten(), adj_target.flatten()).item(), batch_size=1)

        pred_edges = pot_edges[:, batch_pred.squeeze() > self.threshold]
        # compute pred and ground truth adj matrices
        if torch.sum(pred_edges) > 0:
            adj_pred = pyg.utils.to_dense_adj(pred_edges, max_num_nodes=num_nodes).squeeze().cpu()
        else: # to avoid exception in to_dense_adj when there is no predicted edge
            adj_pred = torch.zeros((num_nodes, num_nodes)).squeeze().to(self.device).cpu()
        # compute f1 score on the adj matrices
        test_fscore = self.test_f1_score.cpu()(adj_pred.flatten(), adj_target.flatten())
        self.log("test_fscore", test_fscore.item(), prog_bar=True, batch_size=1)
        self.log("test_precision", self.test_precision.cpu()(adj_pred.flatten(), adj_target.flatten()).item(), batch_size=1)
        self.log("test_recall", self.test_recall.cpu()(adj_pred.flatten(), adj_target.flatten()).item(), batch_size=1)
        self.log("test_la_score", self.test_linear_assignment.cpu()(pot_edges, batch_pred, target_edges, num_nodes).item(), batch_size=1)

    def predict_metric_step(self, batch_pred, pot_edges, target_edges, num_nodes):
        """Predict logging only done once, similar to validation.
        See val_metric_logging_step for details."""
        batch_pred = batch_pred.cpu()
        pot_edges = pot_edges.cpu()
        target_edges = target_edges.cpu()

        adj_target = pyg.utils.to_dense_adj(target_edges, max_num_nodes=num_nodes).squeeze().long().cpu()

        # Solve with Hungarian Algorithm
        cost_matrix = torch.sparse_coo_tensor(
            pot_edges, batch_pred.squeeze(), (num_nodes, num_nodes)).to_dense().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
        potential_edges = np.vstack((row_ind, col_ind))
        trimmed_preds = torch.tensor(cost_matrix[row_ind, col_ind])
        adj_pred = torch.sparse_coo_tensor(torch.tensor(potential_edges), trimmed_preds,
                                           (num_nodes, num_nodes)).to_dense()
        # Eliminate prediction that do not reach the threshold to fix zero nodes
        adj_pred = (adj_pred > self.threshold).float()
        fscore = self.test_f1_allignment.cpu()(adj_pred.flatten(), adj_target.flatten())
        return adj_pred, fscore

    def pitch_score(self, edge_index, mpitch):
        """Pitch score from midi to freq."""
        a = 440  # frequency of A (coomon value is 440Hz)
        fpitch = (a / 32) * (2 ** ((mpitch - 9) / 12))
        pscore = torch.pow(
            torch.div(torch.min(fpitch[edge_index], dim=0)[0], torch.max(fpitch[edge_index], dim=0)[0]), self.alpha)
        return pscore.unsqueeze(1)

    def onset_score(self, edge_index, onset, duration, onset_beat, duration_beat, ts_beats):
        offset = onset + duration
        offset_beat = onset_beat + duration_beat
        note_distance_beat = onset_beat[edge_index[1]] - offset_beat[edge_index[0]]
        ts_beats_edges = ts_beats[edge_index[1]]
        oscore = 1 - (1 / (1 + torch.exp(-2 * (note_distance_beat / ts_beats_edges))) - 0.5) * 2
        one_hot_pitch_score = (onset[edge_index[1]] == offset[edge_index[0]]).float()
        oscore = torch.cat((oscore.unsqueeze(1), one_hot_pitch_score.unsqueeze(1)), dim=1)
        return oscore
