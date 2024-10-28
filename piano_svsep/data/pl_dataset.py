from sklearn.model_selection import train_test_split
import numpy as np
from torch_geometric.loader import DataLoader as PygDataLoader
import torch
from piano_svsep.data.utils import idx_tuple_to_dict, idx_dict_to_tuple
from piano_svsep.data.dataset import DCMLPianoCorporaPolyVoiceSeparationDataset, MusescoreJPopPolyVoiceSeparationDataset
from pytorch_lightning import LightningDataModule


class GraphPolyphonicVoiceSeparationDataModule(LightningDataModule):
    """
    A PyTorch Lightning DataModule for polyphonic voice separation using graph-based datasets.
    It is suppose to handle at once the dcml and jpop datasets.

    Attributes
    ----------
    batch_size : int
        The size of the batches.
    num_workers : int
        The number of workers for data loading.
    force_reload : bool
        Whether to force reload the datasets.
    test_dataset : list
        List of test collections.
    subgraph_size : int
        The size of the subgraphs.
    verbose : bool
        Whether to print verbose output.
    raw_dir : str, optional
        Directory for raw data (defaults to ~/.piano_svsep_data when None).
    randomize_test : bool
        Whether to randomize the test collections.
    train_datasets : list
        List of datasets to use for training.

    """

    def __init__(
            self, batch_size=50, num_workers=4, force_reload=False, test_dataset=None, subgraph_size=500, verbose=False,
            raw_dir=None, train_datasets=["dcml","jpop"]
    ):
        super(GraphPolyphonicVoiceSeparationDataModule, self).__init__()
        self.batch_size = batch_size
        self.subgraph_size = subgraph_size
        self.num_workers = num_workers
        self.force_reload = force_reload
        self.normalize_features = True
        self.randomize_test = False
        self.datasets = []
        # check if datasets input is valid
        if not all([d in ["dcml", "jpop"] for d in train_datasets]):
            raise Exception("Invalid dataset input. Please choose from 'dcml' or 'jpop'")
        if "dcml" in train_datasets:
            self.datasets.append(DCMLPianoCorporaPolyVoiceSeparationDataset(
                force_reload=force_reload, nprocs=num_workers, verbose=verbose, raw_dir=raw_dir),
        )
        if "jpop" in train_datasets:
            self.datasets.append(MusescoreJPopPolyVoiceSeparationDataset(
                force_reload=force_reload, nprocs=num_workers, verbose=verbose, raw_dir=raw_dir)
        )
        if not (all([d.features == self.datasets[0].features for d in self.datasets])):
            raise Exception("Input dataset has different features, Datasets {} with sizes: {}".format(
                " ".join([d.name for d in self.datasets]), " ".join([str(d.features) for d in self.datasets])))
        self.features = self.datasets[0].features
        self.test_dataset = test_dataset if isinstance(test_dataset, list) else [test_dataset]

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        """
        Sets up the datasets and splits them into training, validation, and test sets.
        """
        self.datasets_map = [(dataset_i, piece_i) for dataset_i, dataset in enumerate(self.datasets) for piece_i in
                             range(len(dataset))]
        idxs = np.arange(len(self.datasets_map))
        collections = np.array([self.datasets[self.datasets_map[i][0]].graphs[self.datasets_map[i][1]].collection for i
                                in idxs])
        if self.test_dataset is None or self.randomize_test:
            print("Randomizing test collections")
            trainval_idx, test_idx = train_test_split(idxs, test_size=0.2, stratify=collections, random_state=0)
            trainval_collections = collections[trainval_idx]
            train_idx, val_idx = train_test_split(trainval_idx, test_size=0.1, stratify=trainval_collections,
                                                  random_state=0)
        else:
            print("Using predefined splits for test collections")
            collections_mask = np.isin(np.array(collections), self.test_dataset)
            split_idxs = idxs[collections_mask]
            idx_dict = idx_tuple_to_dict(split_idxs, self.datasets_map)
            self.test_idx_dict = {}
            trainval_idx_dict = {}
            for k in idx_dict.keys():
                dataset_test_pieces = np.array(self.datasets[k].test_files)
                graphs = self.datasets[k].graphs
                graph_names = np.array([graphs[i].name for i in idx_dict[k]])
                test_mask = np.isin(graph_names, dataset_test_pieces)
                test_idx = np.array(idx_dict[k])[test_mask]
                self.test_idx_dict[k] = test_idx
                trainval_idx_dict[k] = np.array(idx_dict[k])[~test_mask]
            # exclude other test sets
            for k, v in idx_tuple_to_dict(idxs[~collections_mask], self.datasets_map).items():
                dataset_test_pieces = np.array(self.datasets[k].test_files)
                graphs = self.datasets[k].graphs
                graph_names = np.array([graphs[i].name for i in v])
                test_mask = np.isin(graph_names, dataset_test_pieces)
                trainval_idx_dict[k] = np.array(v)[~test_mask]


            trainval_tuples = idx_dict_to_tuple(trainval_idx_dict)
            trainval_idx = idxs[np.array([t in trainval_tuples for t in self.datasets_map])]
            # trainval_idx = np.concatenate([trainval_idx, idxs[~collections_mask]])
            trainval_collections = collections[trainval_idx]
            train_idx, val_idx = train_test_split(trainval_idx, test_size=0.1, stratify=trainval_collections,
                                                  random_state=0)


        self.train_idx_dict = idx_tuple_to_dict(train_idx, self.datasets_map)
        self.val_idx_dict = idx_tuple_to_dict(val_idx, self.datasets_map)

        # define the ratio between potential edges and real edges
        self.pot_real_ratio = sum([d.get_positive_weight() for d in self.datasets] ) /len(self.datasets)
        self.pot_real_ratio_chord = sum([d.get_positive_weight_chord() for d in self.datasets] ) /len(self.datasets)

        # create the datasets
        print("Running on all collections")
        print(
            f"Train size :{len(train_idx)}, Val size :{len(val_idx)}, Test size :{len(test_idx)}"
        )

    def train_dataloader(self):
        """
        Creates the DataLoader for the training set.

        Returns
        -------
        DataLoader: The DataLoader for the training set.
        """
        print(f"Creating train dataloader with subgraph size {self.subgraph_size} and batch size {self.batch_size}")
        # compute training graphs here to change the graphs that exceed max size.
        training_graphs = [graph[0] for k in self.train_idx_dict.keys() for graph in self.datasets[k][self.train_idx_dict[k]] ]
        graph_sizes = np.array([g.num_nodes  for g in training_graphs])
        # Compute the number of times each graph should be repeated based on size
        multiples = graph_sizes // self.subgraph_size + 1
        # Create a list of indices repeating each graph the appropriate number of times
        indices = np.concatenate([np.repeat(i, m) for i, m in enumerate(multiples)])
        # Create the dataset by subgraphing each graph to the base size
        dataset_train = list()
        for idx in indices:
            g = training_graphs[idx]
            graph_length = g.num_nodes
            if graph_length > self.subgraph_size: # subgraph only if the size is bigger than the subgraph size
                start = np.random.randint(0, graph_length - self.subgraph_size)
                sub_g = g.subgraph({"note": torch.arange(start, start + self.subgraph_size, dtype=torch.long)})
                dataset_train.append(sub_g)
            else: # otherwise insert the entire graph
                dataset_train.append(g)
        print(f"Passing {len(dataset_train)} subgraphs into the dataloader")

        return PygDataLoader(dataset_train, batch_size = self.batch_size, num_workers=0, shuffle=True)


    def val_dataloader(self):
        """
        Creates the Graph DataLoader for the validation set.

        Returns
        -------
        DataLoader: The DataLoader for the validation set.
        """
        dataset_val = sum([self.datasets[k][self.val_idx_dict[k]] for k in self.val_idx_dict.keys()], [])
        return PygDataLoader(
            dataset_val, batch_size=1, num_workers=0
        )

    def test_dataloader(self):
        """
        Creates the Graph DataLoader for the test set.

        Returns
        -------
        DataLoader: The DataLoader for the test set.
        """
        dataset_test = sum([self.datasets[k][self.test_idx_dict[k]] for k in self.test_idx_dict.keys()], [])
        return PygDataLoader(
            dataset_test, batch_size=1, num_workers=0
        )

    def predict_dataloader(self):
        """
        Creates the DataLoader for the prediction set.


        Returns
        -------
        DataLoader: The DataLoader for the prediction set same as the test set.
        """
        return self.test_dataloader()
