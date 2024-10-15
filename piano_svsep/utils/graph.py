import numpy as np
from numpy.lib import recfunctions as rfn
import torch
import torch_geometric as pyg
import pickle
import os
import warnings
import random
import string
from typing import Union, List, Tuple, Dict, Any


class HeteroScoreGraph(object):
    """
    A class to represent a heterogeneous score graph for musical notes.

    Attributes:
        node_features (list): List of node feature names.
        features (np.ndarray): Array of node features.
        x (torch.Tensor): Tensor of node features.
        etypes (dict): Dictionary mapping edge types to indices.
        note_array (np.ndarray): Array of note information.
        edge_type (torch.Tensor): Tensor of edge types.
        edge_index (torch.Tensor): Tensor of edge indices.
        edge_weights (torch.Tensor): Tensor of edge weights.
        name (str): Name of the graph.
        y (torch.Tensor): Tensor of labels.
    """

    def __init__(self, note_features, edges, etypes=["onset", "consecutive", "during", "rest"], name=None, note_array=None, edge_weights=None, labels=None):
        """
        Initializes the HeteroScoreGraph with given features, edges, and other attributes.

        Args:
            note_features (np.ndarray): Array of note features.
            edges (np.ndarray): Array of edges.
            etypes (list): List of edge types.
            name (str, optional): Name of the graph.
            note_array (np.ndarray, optional): Array of note information.
            edge_weights (np.ndarray, optional): Array of edge weights.
            labels (np.ndarray, optional): Array of labels.
        """
        self.node_features = note_features.dtype.names if note_features.dtype.names else []
        self.features = note_features
        # Filter out string fields of structured array.
        if self.node_features:
            self.node_features = [feat for feat in self.node_features if note_features.dtype.fields[feat][0] != np.dtype('U256')]
            self.features = self.features[self.node_features]
        self.x = torch.tensor(np.asarray(rfn.structured_to_unstructured(self.features)) if self.node_features else self.features , dtype = torch.float32)
        assert etypes is not None
        self.etypes = {t: i for i, t in enumerate(etypes)}
        self.note_array = note_array
        self.edge_type = torch.tensor(edges[-1], dtype=torch.long)
        self.edge_index = torch.tensor(edges[:2], dtype=torch.long )
        self.edge_weights = torch.ones(len(self.edge_index[0])) if edge_weights is None else torch.tensor(edge_weights)
        self.name = name
        self.y = labels if labels is None else torch.tensor(labels)

    def adj(self, weighted=False):
        """
        Returns the adjacency matrix of the graph.

        Args:
            weighted (bool): If True, returns a weighted adjacency matrix.

        Returns:
            torch.sparse_coo_tensor: The adjacency matrix.
        """
        if weighted:
            return torch.sparse_coo_tensor(self.edge_index, self.edge_weights, (len(self.x), len(self.x)))
        ones = torch.ones(len(self.edge_index[0]))
        matrix = torch.sparse_coo_tensor(self.edge_index, ones, (len(self.x), len(self.x)))
        return matrix

    def add_measure_nodes(self, measures):
        """
        Adds virtual nodes for every measure.

        Args:
            measures (np.ndarray): Array of measure start and end times.
        """
        assert "onset_div" in self.note_array.dtype.names, "Note array must have 'onset_div' field to add measure nodes."
        if not isinstance(measures, np.ndarray):
            measures = np.array([[m.start.t, m.end.t] for m in measures])
        nodes = np.arange(len(measures))
        edges = []
        for i in range(len(measures)):
            idx = np.where((self.note_array["onset_div"] >= measures[i,0]) & (self.note_array["onset_div"] < measures[i,1]))[0]
            if idx.size:
                edges.append(np.vstack((idx, np.full(idx.size, i))))
        self.measure_nodes = nodes
        self.measure_edges = np.hstack(edges)
        if self.measure_edges.size == 0:
            warnings.warn(f"No edges found for measure nodes. Check that the note array has the 'onset_div' field on score {self.name}.")

    def add_beat_nodes(self):
        """
        Adds virtual nodes for every beat.
        """
        assert "onset_beat" in self.note_array.dtype.names, "Note array must have 'onset_beat' field to add measure nodes."
        nodes = np.arange(int(self.note_array["onset_beat"].max()))
        edges = []
        for b in nodes:
            idx = np.where((self.note_array["onset_beat"] >= b) & (self.note_array["onset_beat"] < b + 1))[0]
            if idx.size:
                edges.append(np.vstack((idx, np.full(idx.size, b))))
        self.beat_nodes = nodes
        self.beat_edges = np.hstack(edges)

    def assign_typed_weight(self, weight_dict:dict):
        """
        Assigns weights to edges based on their types.

        Args:
            weight_dict (dict): Dictionary mapping edge types to weights.
        """
        assert weight_dict.keys() == self.etypes.keys()
        for k, v in weight_dict.items():
            etype = self.etypes[k]
            self.edge_weights[self.edge_type == etype] = v

    def get_edges_of_type(self, etype):
        """
        Returns the edges of a specific type.

        Args:
            etype (str): The edge type.

        Returns:
            torch.Tensor: The edges of the specified type.
        """
        assert etype in self.etypes.keys()
        etype = self.etypes[etype]
        return self.edge_index[:, self.edge_type == etype]

    def save(self, save_dir):
        """
        Saves the graph to a directory.

        Args:
            save_dir (str): The directory to save the graph.
        """
        save_name = self.name if self.name else ''.join(random.choice(string.ascii_letters) for i in range(10))
        (os.makedirs(os.path.join(save_dir, save_name)) if not os.path.exists(os.path.join(save_dir, save_name)) else None)
        object_properties = vars(self)
        with open(os.path.join(save_dir, save_name, "x.npy"), "wb") as f:
            np.save(f, self.x.numpy())
        del object_properties['x']
        with open(os.path.join(save_dir, save_name, "edge_index.npy"), "wb") as f:
            np.save(f, torch.cat((self.edge_index, self.edge_type.unsqueeze(0))).numpy())
        del object_properties['edge_index']
        del object_properties['edge_type']
        if isinstance(self.y, torch.Tensor):
            with open(os.path.join(save_dir, save_name, "y.npy"), "wb") as f:
                np.save(f, self.y.numpy())
            del object_properties['y']
        if isinstance(self.edge_weights, torch.Tensor):
            np.save(open(os.path.join(save_dir, save_name, "edge_weights.npy"), "wb"), self.edge_weights.numpy())
            del object_properties['edge_weights']
        if isinstance(self.note_array, np.ndarray):
            np.save(open(os.path.join(save_dir, save_name, "note_array.npy"), "wb"), self.note_array)
            del object_properties['note_array']
        with open(os.path.join(save_dir, save_name, 'graph_info.pkl'), 'wb') as handle:
            pickle.dump(object_properties, handle, protocol=pickle.HIGHEST_PROTOCOL)


def hetero_graph_from_note_array(note_array, rest_array=None, norm2bar=False, pot_edge_dist=3):
    '''Turn note_array to homogeneous graph dictionary.

    Parameters
    ----------
    note_array : structured array
        The partitura note_array object. Every entry has 5 attributes, i.e. onset_time, note duration, note velocity, voice, id.
    rest_array : structured array
        A structured rest array similar to the note array but for rests.
    t_sig : list
        A list of time signature in the piece.
    '''

    edg_src = list()
    edg_dst = list()
    etype = list()
    pot_edges = list()
    start_rest_index = len(note_array)
    for i, x in enumerate(note_array):
        for j in np.where(note_array["onset_div"] == x["onset_div"])[0]:
            if i != j:
                edg_src.append(i)
                edg_dst.append(j)
                etype.append(0)
        if pot_edge_dist:
            for j in np.where(
                    (note_array["onset_div"] > x["onset_div"]+x["duration_div"]) &
                    (note_array["onset_beat"] <= x["onset_beat"] + x["duration_beat"] + pot_edge_dist*x["ts_beats"])
            )[0]:
                pot_edges.append([i, j])
        for j in np.where(note_array["onset_div"] == x["onset_div"] + x["duration_div"])[0]:
            edg_src.append(i)
            edg_dst.append(j)
            etype.append(1)

        if isinstance(rest_array, np.ndarray) and rest_array.size > 0:
            for j in np.where(rest_array["onset_div"] == x["onset_div"] + x["duration_div"])[0]:
                edg_src.append(i)
                edg_dst.append(j + start_rest_index)
                etype.append(1)

        for j in np.where(
                (x["onset_div"] < note_array["onset_div"]) & (x["onset_div"] + x["duration_div"] > note_array["onset_div"]))[0]:
            edg_src.append(i)
            edg_dst.append(j)
            etype.append(2)

    if isinstance(rest_array, np.ndarray) and rest_array.size > 0:
        for i, r in enumerate(rest_array):
            for j in np.where(np.isclose(note_array["onset_div"], r["onset_div"] + r["duration_div"], rtol=1e-04, atol=1e-04) == True)[0]:
                edg_src.append(start_rest_index + i)
                edg_dst.append(j)
                etype.append(1)

        feature_fn = [dname for dname in note_array.dtype.names if dname not in rest_array.dtype.names]
        if feature_fn:
            rest_feature_zeros = np.zeros((len(rest_array), len(feature_fn)))
            rest_feature_zeros = rfn.unstructured_to_structured(rest_feature_zeros, dtype=list(map(lambda x: (x, '<4f'), feature_fn)))
            rest_array = rfn.merge_arrays((rest_array, rest_feature_zeros))
    else:
        end_times = note_array["onset_div"] + note_array["duration_div"]
        for et in np.sort(np.unique(end_times))[:-1]:
            if et not in note_array["onset_div"]:
                scr = np.where(end_times == et)[0]
                diffs = note_array["onset_div"] - et
                tmp = np.where(diffs > 0, diffs, np.inf)
                dst = np.where(tmp == tmp.min())[0]
                for i in scr:
                    for j in dst:
                        edg_src.append(i)
                        edg_dst.append(j)
                        etype.append(3)


    edges = np.array([edg_src, edg_dst, etype])

    # Resize Onset Beat to bar
    if norm2bar:
        note_array["onset_beat"] = np.mod(note_array["onset_beat"], note_array["ts_beats"])
        if isinstance(rest_array, np.ndarray) and rest_array.size > 0:
            rest_array["onset_beat"] = np.mod(rest_array["onset_beat"], rest_array["ts_beats"])

    nodes = np.hstack((note_array, rest_array))
    if pot_edge_dist:
        pot_edges = np.hstack((np.array(pot_edges).T, edges[:, edges[2] == 1][:2]))
        return nodes, edges, pot_edges
    return nodes, edges


def score_graph_to_pyg(score_graph: HeteroScoreGraph):
    """
    Converts a ScoreGraph to a PyTorch Geometric graph.
    Parameters
    ----------
    score_graph : ScoreGraph
        The ScoreGraph to convert
    """
    if isinstance(score_graph, HeteroScoreGraph):
        data = pyg.data.HeteroData()
        data["note"].x = score_graph.x.clone()
        # data["note"].y = y
        # add edges
        for e_type in score_graph.etypes.keys():
            data["note", e_type, "note"].edge_index = score_graph.get_edges_of_type(
                e_type
            ).clone().long()
        # add potential edges
        if hasattr(score_graph, "pot_edges"):
            data["note", "potential", "note"].edge_index = score_graph.pot_edges.clone().long()
        # add truth edges
        if hasattr(score_graph, "truth_edges"):
            data["note", "truth", "note"].edge_index = score_graph.truth_edges.clone().long()
        # add chord potential edges
        if hasattr(score_graph, "pot_chord_edges"):
            data["note", "chord_potential", "note"].edge_index = score_graph.pot_chord_edges.clone().long()
        # add chord truth edges
        if hasattr(score_graph, "truth_chord_edges"):
            data["note", "chord_truth", "note"].edge_index = score_graph.truth_chord_edges.clone().long()
        # add pitch, onset, offset info in divs that is necessary for evaluation
        data["note"].pitch = torch.tensor(score_graph.note_array["pitch"].copy())
        data["note"].onset_div = torch.tensor(score_graph.note_array["onset_div"].copy())
        data["note"].duration_div = torch.tensor(score_graph.note_array["duration_div"].copy())
        data["note"].onset_beat = torch.tensor(score_graph.note_array["onset_beat"].copy())
        data["note"].duration_beat = torch.tensor(score_graph.note_array["duration_beat"].copy())
        data["note"].ts_beats = torch.tensor(score_graph.note_array["ts_beats"].copy())
        # staff is shifted to be 0-1 instead of 1-2
        data["note"].staff = torch.tensor(score_graph.note_array["staff"].copy() -1)
        data["note"].voice = torch.tensor(score_graph.note_array["voice"].copy())
        assert (data["note"].staff).min() == 0
        # assert (data["note"].staff).max() == 1
        # add name
        data["name"] = score_graph.name
        # add collection
        if hasattr(score_graph, "collection"):
            data["collection"] = score_graph.collection
    else:
        raise ValueError("Only HeteroScoreGraph is supported for now")

    return data

def add_reverse_edges(graph, mode):
    """
    Adds reverse edges to the Score graph based on the specified mode.
    For new_type mode, it adds reverse to consecutive, rest and during edges.
    Onset edges are already bidirectional.

    Args:
        graph (HeteroScoreGraph or pyg.data.HeteroData): The Score graph to which reverse edges will be added.
        mode (str): The mode of adding reverse edges. Can be "new_type" or "undirected".

    Returns:
        HeteroScoreGraph or pyg.data.HeteroData: The graph with added reverse edges.

    Raises:
        NotImplementedError: If mode is "undirected" for HeteroScoreGraph.
        ValueError: If mode is not "new_type" or "undirected".
    """
    if isinstance(graph, HeteroScoreGraph):
        if mode == "new_type":
            # Add reverse During Edges
            graph.edge_index = torch.cat((graph.edge_index, graph.get_edges_of_type("during").flip(0)), dim=1)
            graph.edge_type = torch.cat((graph.edge_type, 2 + torch.zeros(graph.edge_index.shape[1] - graph.edge_type.shape[0], dtype=torch.long)), dim=0)
            # Add reverse Consecutive Edges
            graph.edge_index = torch.cat((graph.edge_index, graph.get_edges_of_type("consecutive").flip(0)), dim=1)
            graph.edge_type = torch.cat((graph.edge_type, 4 + torch.zeros(graph.edge_index.shape[1] - graph.edge_type.shape[0], dtype=torch.long)), dim=0)
            graph.etypes["consecutive_rev"] = 4
        else:
            graph.edge_index = torch.cat((graph.edge_index, graph.edge_index.flip(0)), dim=1)
            raise NotImplementedError("To undirected is not Implemented for HeteroScoreGraph.")
    else:
        if mode == "new_type":
            # Add reversed consecutive edges
            graph["note", "consecutive_rev", "note"].edge_index = graph["note", "consecutive", "note"].edge_index[[1, 0]]
            # Add reversed during edges
            graph["note", "during_rev", "note"].edge_index = graph["note", "during", "note"].edge_index[[1, 0]]
            # Add reversed rest edges
            graph["note", "rest_rev", "note"].edge_index = graph["note", "rest", "note"].edge_index[[1, 0]]
        elif mode == "undirected":
            graph = pyg.transforms.ToUndirected()(graph)
        else:
            raise ValueError("mode must be either 'new_type' or 'undirected'")
    return graph


def add_reverse_edges_from_edge_index(edge_index, edge_type, mode="new_type"):
    if mode == "new_type":
        unique_edge_types = torch.unique(edge_type)
        for type in unique_edge_types:
            if type == 0:
                continue
            edge_index = torch.cat((edge_index, edge_index[:, edge_type == type].flip(0)), dim=1)
            edge_type = torch.cat((edge_type, torch.max(edge_type) + torch.zeros(edge_index.shape[1] - edge_type.shape[0], dtype=torch.long).to(edge_type.device)), dim=0)
    else:
        edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)
        edge_type = torch.cat((edge_type, edge_type), dim=0)
    return edge_index, edge_type
