import torch.nn as nn
import torch
from torch_geometric.utils import coalesce
from typing import Tuple
from torch import Tensor
from piano_svsep.models.models import UnpoolInfo

class PostProcessPooling(nn.Module):
    """
    A PyTorch module for post-processing pooling operations in a graph neural network.

    Args:
        threshold (float): The threshold value for filtering chord edges.
    """
    def __init__(self, threshold=0.5):
        """
        Initializes the PostProcessPooling module with a given threshold.

        Args:
            threshold (float): The threshold value for filtering chord edges.
        """
        super().__init__()
        self.threshold = threshold

    def forward(self, edge_index, edge_probs, chord_edge_index, chord_edge_score, batch, num_nodes):
        """
        Forward pass for the post-processing pooling operation.

        Args:
            edge_index (Tensor): The edge indices of the graph.
            edge_probs (Tensor): The edge probabilities of the graph.
            chord_edge_index (Tensor): The edge indices for chord connections.
            chord_edge_score (Tensor): The scores for chord edges.
            batch (Tensor): The batch indices for the nodes.
            num_nodes (int): The number of nodes in the graph.

        Returns:
            Tuple[Tensor, Tensor, UnpoolInfo, int]: The new edge indices, new edge probabilities, unpool information, and reduced number of nodes.
        """
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

        # Coalesce the edges and edge probabilities based on the new clusters
        new_edge_index, new_edge_probs = coalesce(cluster[edge_index], edge_probs, num_nodes=reduced_num_nodes, reduce="mean")
        new_batch = batch.new_empty(reduced_num_nodes, dtype=torch.long)
        new_batch = new_batch.scatter_(0, cluster, batch)

        unpool_info = UnpoolInfo(edge_index=edge_index, cluster=cluster, batch=batch, new_edge_score=edge_score)

        return new_edge_index, new_edge_probs, unpool_info, reduced_num_nodes

    def unpool(self, edge_index, num_nodes, unpool_info: UnpoolInfo) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Unpooling operation to reconstruct the original graph from the pooled representation.

        Args:
            edge_index (Tensor): The edge indices of the pooled graph.
            num_nodes (int): The number of nodes in the pooled graph.
            unpool_info (UnpoolInfo): The information required for unpooling.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: The new edge indices after unpooling.
        """
        x = torch.arange(num_nodes, device=edge_index.device)
        new_x = x[unpool_info.cluster]
        new_edge_index = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

        # Reconstruct the edges by creating all possible connections between the nodes in the same cluster
        for edge_idx in range(edge_index.shape[-1]):
            row = edge_index[0, edge_idx]
            col = edge_index[1, edge_idx]
            multiple_rows = torch.where(new_x == row)[0]
            multiple_cols = torch.where(new_x == col)[0]
            new_edges = torch.cartesian_prod(multiple_rows, multiple_cols).T
            new_edge_index = torch.cat((new_edge_index, new_edges), dim=-1)

        return new_edge_index