import os
import hashlib
import abc
import torch
import partitura
import numpy as np
from tqdm import tqdm
import gc
from pathlib import Path
from joblib import Parallel, delayed
from piano_svsep.utils import remove_ties_acros_barlines, sanitize_staff_voices, hetero_graph_from_note_array, get_vocsep_features, get_measurewise_truth_edges, get_measurewise_pot_edges, get_pot_chord_edges, get_truth_chords_edges, score_graph_to_pyg, HeteroScoreGraph
from piano_svsep.data.utils import makedirs, get_download_dir
from git import Repo


class BaseDataset(object):
    """The basic Struttura Dataset for creating various datasets.
    This class defines a basic template class for Struttura Dataset.
    The following steps will are executed automatically:

      1. Check whether there is a dataset cache on disk
         (already processed and stored on the disk) by
         invoking ``has_cache()``. If true, goto 5.
      2. Call ``download()`` to download the data.
      3. Call ``process()`` to process the data.
      4. Call ``save()`` to save the processed dataset on disk and goto 6.
      5. Call ``load()`` to load the processed dataset from disk.
      6. Done.

    Users can overwite these functions with their
    own data processing logic.

    Parameters
    ----------
    name : str
        Name of the dataset
    url : str
        Url to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.piano_svsep_data/
    save_dir : str
        Directory to save the processed dataset.
        Default: same as raw_dir
    hash_key : tuple
        A tuple of values as the input for the hash function.
        Users can distinguish instances (and their caches on the disk)
        from the same dataset class by comparing the hash values.
        Default: (), the corresponding hash value is ``'f9065fa7'``.
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information

    Attributes
    ----------
    url : str
        The URL to download the dataset
    name : str
        The dataset name
    raw_dir : str
        Raw file directory contains the input data folder
    raw_path : str
        Directory contains the input data files.
        Default : ``os.path.join(self.raw_dir, self.name)``
    save_dir : str
        Directory to save the processed dataset
    save_path : str
        File path to save the processed dataset
    verbose : bool
        Whether to print information
    hash : str
        Hash value for the dataset and the setting.
    """
    def __init__(self, name, features="all", url=None, raw_dir=None, save_dir=None,
                 hash_key=(), force_reload=False, verbose=False):
        self._name = name
        self._url = url
        self._features = features
        self._force_reload = force_reload
        self._verbose = verbose
        self._hash_key = hash_key
        self._hash = self._get_hash()

        # if no dir is provided, the default piano_svsep_data download dir is used.
        if raw_dir is None:
            self._raw_dir = get_download_dir()
        else:
            self._raw_dir = raw_dir

        if save_dir is None:
            self._save_dir = self._raw_dir
        else:
            self._save_dir = save_dir

        self._load()

    def download(self):
        """Overwite to realize your own logic of downloading data.

        It is recommended to download the to the :obj:`self.raw_dir`
        folder. Can be ignored if the dataset is
        already in :obj:`self.raw_dir`.
        """
        pass

    def save(self):
        pass

    def load(self):
        pass

    def process(self):
        raise NotImplementedError

    def has_cache(self):
        return False

    def _download(self):
        """Download dataset by calling ``self.download()`` if the dataset does not exists under ``self.raw_path``.
            By default ``self.raw_path = os.path.join(self.raw_dir, self.name)``
            One can overwrite ``raw_path()`` function to change the path.
        """
        if os.path.exists(self.raw_path):  # pragma: no cover
            return

        makedirs(self.raw_dir)
        self.download()

    def _load(self):
        load_flag = not self._force_reload and self.has_cache()

        if load_flag:
            try:
                self.load()
                if self.verbose:
                    print('Done loading data from cached files for {} Dataset.'.format(self._name))
            except KeyboardInterrupt:
                raise
            except:
                load_flag = False
                print('Loading from cache failed, re-processing.')

        if not load_flag:
            self._download()
            if self.verbose:
                print('Preprocessing data...')
            self.process()
            if self.verbose:
                print('Saving preprocessed data...')
            self.save()
            if self.verbose:
                print('Done saving data into cached files for {} dataset.'.format(self._name))

    def _get_hash(self):
        hash_func = hashlib.sha1()
        hash_func.update(str(self._hash_key).encode('utf-8'))
        return hash_func.hexdigest()[:8]

    @property
    def url(self):
        """Get url to download the raw dataset.
        """
        return self._url

    @property
    def name(self):
        r"""Name of the dataset.
        """
        return self._name

    @property
    def raw_dir(self):
        r"""Raw file directory contains the input data folder.
        """
        return self._raw_dir

    @property
    def raw_path(self):
        r"""Directory contains the input data files.
            By default raw_path = os.path.join(self.raw_dir, self.name)
        """
        return os.path.join(self.raw_dir, self.name)

    @property
    def save_dir(self):
        r"""Directory to save the processed dataset.
        """
        return self._save_dir

    @property
    def save_path(self):
        r"""Path to save the processed dataset.
        """
        return os.path.join(self._save_dir, self.name)

    @property
    def verbose(self):
        r"""Whether to print information.
        """
        return self._verbose

    @property
    def hash(self):
        r"""Hash value for the dataset and the setting.
        """
        return self._hash

    @abc.abstractmethod
    def __getitem__(self, idx):
        r"""Gets the data object at index.
        """
        pass

    @abc.abstractmethod
    def __len__(self):
        r"""The number of examples in the dataset."""
        pass


class BuiltinDataset(BaseDataset):
    """The Basic Builtin Dataset.

    Parameters
    ----------
    name : str
        Name of the dataset.
    url : str
        Url to download the raw dataset.
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.piano_svsep_data/
    hash_key : tuple
        A tuple of values as the input for the hash function.
        Users can distinguish instances (and their caches on the disk)
        from the same dataset class by comparing the hash values.
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
        Whether to print out progress information. Default: False
    is_zip : bool
    """
    def __init__(self, name, url, raw_dir=None, hash_key=(), force_reload=False, verbose=False, is_zip=False, clone=False, branch=None):
        self.is_zip = is_zip
        self.force_reload = force_reload
        self.clone = clone if not is_zip else False
        if self.clone:
            self.branch = "master" if branch is None else branch
        else:
            self.branch = None
        super(BuiltinDataset, self).__init__(
            name,
            url=url,
            raw_dir=raw_dir,
            save_dir=None,
            hash_key=hash_key,
            force_reload=force_reload,
            verbose=verbose)

    def download(self):
        if "https://github.com/" in self.url or self.clone or "https://gitlab." in self.url:
            repo_path = os.path.join(self.raw_dir, self.name)
            Repo.clone_from(self.url, repo_path, single_branch=True, b=self.branch, depth=1)
        else:
            raise ValueError("Unknown url: {}".format(self.url))



class DCMLPianoCorporaDataset(BuiltinDataset):
    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True):
        url = "https://github.com/fosfrancesco/piano_corpora_dcml.git"
        super(DCMLPianoCorporaDataset, self).__init__(
            name="DCMLPianoCorporaDataset",
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            is_zip=False)
        self.scores = []
        self.collections = []
        self.composers = []
        self.period = "Unknown"
        self.type = []
        self.process()

    def process(self):
        self.scores = []
        self.collections = []
        for fn in os.listdir(os.path.join(self.save_path, "scores")):
            for score_fn in os.listdir(os.path.join(self.save_path, "scores", fn)):
                if score_fn.endswith(".musicxml"):
                    self.scores.append(os.path.join(self.save_path, "scores", fn, score_fn))
                    self.collections.append("dcml")
                    self.composers.append(fn.split("_")[0])
                    self.type.append("_".join(fn.split("_")[1:]))

    def has_cache(self):
        if os.path.exists(self.save_path):
            return True
        return False


class GraphPolyphonicVoiceSeparationDataset(BaseDataset):
    def __init__(
            self, dataset_base, is_pyg=True, raw_dir=None, force_reload=False, verbose=True, nprocs=4, include_measures=False, max_size=500, subsample_size=200, prob_pieces=[]
    ):
        self.dataset_base = dataset_base
        self.prob_pieces = prob_pieces
        self.dataset_base.process()
        self.max_size = max_size
        self.stage = "validate"
        if verbose:
            print("Loaded {} Successfully, now processing...".format(dataset_base.name))
        self.graphs = list()
        self.n_jobs = nprocs
        self.dropped_notes = 0
        self.is_pyg = is_pyg
        self._force_reload = force_reload
        self.include_measures = include_measures
        name = self.dataset_base.name.split("Dataset")[0] + "PGGraphPolyVoiceSeparationDataset"
        super(GraphPolyphonicVoiceSeparationDataset, self).__init__(
            name=name,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
        )

    def process(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_score)(score, collection)
            for score, collection in tqdm(
                zip(self.dataset_base.scores, self.dataset_base.collections)
            )
        )
        self.load()

    def has_cache(self):
        if self.is_pyg:
            if all(
                    [os.path.exists(os.path.join(self.save_path, os.path.splitext(os.path.basename(path))[0] + ".pt" ,))
                     for path in self.dataset_base.scores]
            ):
                return True
        else:
            if all(
                    [os.path.exists(os.path.join(self.save_path, os.path.splitext(os.path.basename(path))[0]))
                     for path in self.dataset_base.scores]
            ):
                return True
        return False

    def load(self):
        for fn in os.listdir(self.save_path):
            path_graph = os.path.join(self.save_path, fn)
            graph = torch.load(path_graph)
            self.graphs.append(graph)

    def _process_score(self, score_fn, collection):
        parent_name = os.path.basename(os.path.dirname(score_fn))
        name = os.path.splitext(os.path.basename(score_fn))[0]
        name = f"{parent_name}_{name}" if parent_name != "musicxml" else name
        if self._force_reload or \
                not Path(self.save_path, f"{collection}_{name}.pt").exists():
            if os.path.splitext(os.path.basename(score_fn))[0] in self.prob_pieces:
                return
            try:
                score = partitura.load_score(score_fn)
                # remove the ties across measures in the score
                # NOTE: the removing of ties could need to be done after the input graph creation for other tasks
                remove_ties_acros_barlines(score)

                note_array = score.note_array(
                    include_time_signature=True,
                    include_grace_notes=True,
                    include_staff=True,
                )
                # sanitize the note array to have only one part and two staves
                sanitize_staff_voices(note_array)
            except:
                print("Failed to load score: ", score_fn)
                return

            # get the measure number for each note in the note array
            mn_map = score[np.array([p._quarter_durations[0] for p in score]).argmax()].measure_number_map
            note_measures = mn_map(note_array["onset_div"])

            # compute the input graph
            nodes, edges = hetero_graph_from_note_array(note_array, pot_edge_dist=0)
            note_features = get_vocsep_features(note_array)
            hg = HeteroScoreGraph(
                note_features,
                edges,
                name=f"{collection}_{name}",
                labels=None,
                note_array=note_array,
            )
            # Compute the output graph
            truth_edges = get_measurewise_truth_edges(note_array ,note_measures)
            pot_edges = get_measurewise_pot_edges(note_array ,note_measures)
            pot_chord_edges = get_pot_chord_edges(note_array, hg.get_edges_of_type("onset").numpy())
            truth_chords_edges = get_truth_chords_edges(note_array, pot_chord_edges)

            # save edges for output graph
            setattr(hg, "truth_edges", torch.tensor(truth_edges))
            setattr(hg, "pot_edges", torch.tensor(pot_edges))
            setattr(hg, "pot_chord_edges", torch.tensor(pot_chord_edges))
            setattr(hg, "truth_chord_edges", torch.tensor(truth_chords_edges))

            # Save collection as an attribute of the graph.
            setattr(hg, "collection", collection)

            # transform to pytorch geometric
            pg_graph = score_graph_to_pyg(hg)
            assert pg_graph.num_nodes != 0, f"0 note score graph for piece {pg_graph['name']}"
            file_path = Path(self.save_path, f"{pg_graph['name']}.pt")
            torch.save(pg_graph, file_path)
            del pg_graph
            del hg, note_array, truth_edges, nodes, edges, note_features, score
            gc.collect()
        return

    def set_split(self, stage):
        self.stage = stage

    def __getitem__(self, idx):
        if self.is_pyg:
            return [[self.graphs[i]] for i in idx]
        else:
            raise Exception("You should be using pytorch geometric.")

    def __len__(self):
        return len(self.graphs)

    @property
    def features(self):
        return self.graphs[0]["note"].x.shape[-1]

    @property
    def metadata(self):
        return self.graphs[0].metadata()

    def num_dropped_truth_edges(self):
        return sum([len(graph["dropped_truth_edges"]) for graph in self.graphs])

    def get_positive_weight(self):
        return sum \
            ([g["note" ,"potential" ,"note"].edge_index.shape[1 ] /g["note" ,"truth" ,"note"].edge_index.shape[1] for g in self.graphs] ) /len(self.graphs)

    def get_positive_weight_chord(self):
        return sum([g["note" ,"chord_potential" ,"note"].edge_index.shape[1 ] /max
            (g["note" ,"chord_truth" ,"note"].edge_index.shape[1] ,1) for g in self.graphs] ) /len(self.graphs)



class MusescorePopDataset(BuiltinDataset):
    def __init__(self, raw_dir=None, force_reload=False,
                 verbose=True):
        url = ""
        raise NotImplementedError("The MusescorePopDataset is private and not available for download. Please contact the authors for access.")
        super(MusescorePopDataset, self).__init__(
            name="MusescorePopDataset",
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            is_zip=True)
        self.scores = []
        self.collections = []
        self.composer = "Unknown"
        self.period = "Unknown"
        self.type = "Pop"
        self.process()

    def process(self):
        self.scores = []
        self.collections = []
        for score_fn in os.listdir(os.path.join(self.save_path, "musicxml")):
            if score_fn.endswith(".xml"):
                self.scores.append(os.path.join(self.save_path, "musicxml", score_fn))
                self.collections.append("musescore_pop")

    def has_cache(self):
        if os.path.exists(self.save_path):
            return True
        return False


class DCMLPianoCorporaPolyVoiceSeparationDataset(GraphPolyphonicVoiceSeparationDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, nprocs=4, max_size=5000):
        dataset_base = DCMLPianoCorporaDataset(raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)
        # These pieces have 3 staves. We discard them for now.
        prob_pieces = ["l000_soirs", "l099_cahier", "l111-01_images_cloches", "l111-02_images_lune", "l111-03_images_poissons", "l117-11_preludes_danse", "l123-01_preludes_brouillards", "l123-02_preludes_feuilles", "l123-03_preludes_puerta", "l123-04_preludes_fees", "l123-05_preludes_bruyeres", "l123-06_preludes_general", "l123-07_preludes_terrasse", "l123-08_preludes_ondine", "l123-09_preludes_hommage", "l123-10_preludes_canope", "l123-11_preludes_tierces", "l123-12_preludes_feux", "l136-10_etudes_sonorites", "op43n06", "op34n03", "op35n04"]
        super(DCMLPianoCorporaPolyVoiceSeparationDataset, self).__init__(dataset_base=dataset_base, force_reload=force_reload, nprocs=nprocs, max_size=max_size, prob_pieces=prob_pieces)
        # display some data on the number of pieces that was discarded
        print("DCMLPianoCorporaPolyVoiceSeparationDataset loaded.")
        print(f"Available pieces: {len(self.graphs)} from the original {len(self.dataset_base.scores)} scores")
        print(f"{len(self.dataset_base.scores) - len(self.graphs)} scores were discarded due to exceptions during loading.")
        print("Score discarded:")
        orig_names = [f"{collection}_{Path(score_fn).stem}" for score_fn,collection in zip(self.dataset_base.scores,self.dataset_base.collections)]
        proc_names = [g.name for g in self.graphs]
        discarded = [score for score in orig_names if score not in proc_names]
        self.test_files = [
            'liszt_pelerinage_160.08_Le_Mal_du_Pays_(Heimweh)', 'grieg_lyric_pieces_op12n05', 'grieg_lyric_pieces_op47n05',
            'grieg_lyric_pieces_op38n04', 'chopin_mazurkas_BI153-2op56-2',
            'beethoven_piano_sonatas_11-3', 'debussy_corpus_l066-02_arabesques_deuxieme', 'tchaikovsky_seasons_op37a05',
            'debussy_corpus_l133_page', 'grieg_lyric_pieces_op12n08', 'debussy_corpus_l110-02_images_hommage',
            'schumann_kinderszenen_n12', 'mozart_sonatas_K576-1', 'mozart_sonatas_K283-2', 'beethoven_piano_sonatas_11-2',
            'beethoven_piano_sonatas_21-2', 'debussy_corpus_l132_berceuse', 'beethoven_piano_sonatas_13-2',
            'chopin_mazurkas_BI85', 'dvorak_silhouettes_op08n08', 'mozart_sonatas_K282-2', 'debussy_corpus_l136-05_etudes_octaves',
            'chopin_mazurkas_BI105-3op30-3', 'beethoven_piano_sonatas_02-3', 'schumann_kinderszenen_n06',
            'beethoven_piano_sonatas_21-3', 'chopin_mazurkas_BI77-2op17-2', 'grieg_lyric_pieces_op68n05',
            'debussy_corpus_l095-02_pour_sarabande', 'grieg_lyric_pieces_op12n01', 'grieg_lyric_pieces_op38n07',
            'debussy_corpus_l082_nocturne', 'chopin_mazurkas_BI167op67-2', 'medtner_tales_op26n03', 'debussy_corpus_l108_morceau',
            'chopin_mazurkas_BI77-3op17-3', 'beethoven_piano_sonatas_03-3',
            'chopin_mazurkas_BI60-4op06-4', 'chopin_mazurkas_BI60-1op06-1', 'liszt_pelerinage_160.05_Orage',
            'grieg_lyric_pieces_op38n06', 'grieg_lyric_pieces_op57n02', 'mozart_sonatas_K310-1', 'grieg_lyric_pieces_op65n03',
            'mozart_sonatas_K545-1', 'schumann_kinderszenen_n04', 'mozart_sonatas_K279-1', 'beethoven_piano_sonatas_13-1',
            'beethoven_piano_sonatas_18-4', 'mozart_sonatas_K280-2',
            'beethoven_piano_sonatas_14-1', 'grieg_lyric_pieces_op38n08', 'grieg_lyric_pieces_op38n05',
            'liszt_pelerinage_160.09_Les_Cloches_de_Geneve_(Nocturne)', 'chopin_mazurkas_BI93-1op67-1',
            'tchaikovsky_seasons_op37a07', 'beethoven_piano_sonatas_04-3', 'medtner_tales_op48n02',
            'mozart_sonatas_K311-3', 'mozart_sonatas_K310-3', 'medtner_tales_op35n02', 'beethoven_piano_sonatas_03-4',
            'beethoven_piano_sonatas_31-1', 'beethoven_piano_sonatas_17-3', 'debussy_corpus_l100-01_estampes_pagode',
            'beethoven_piano_sonatas_30-2', 'mozart_sonatas_K331-1', 'chopin_mazurkas_BI162-3op63-3',
            'debussy_corpus_l121_plus', 'beethoven_piano_sonatas_09-1', 'beethoven_piano_sonatas_01-4', 'mozart_sonatas_K332-1',
            'beethoven_piano_sonatas_07-3', 'dvorak_silhouettes_op08n05', 'chopin_mazurkas_BI60-2op06-2',
            'grieg_lyric_pieces_op38n02', 'debussy_corpus_l136-06_etudes_huit']
        self.test_files = [f"dcml_{piece}" for piece in self.test_files]
        # Excluded from tests due to exceptional cases
        # ['liszt_pelerinage_162.01_Gondoliera', 'debussy_corpus_l095-01_pour_prelude', 'debussy_corpus_l100-03_estampes_jardins']
        print(discarded)



class MusescorePopPolyVoiceSeparationDataset(GraphPolyphonicVoiceSeparationDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=True, nprocs=4, max_size=5000):
        dataset_base = MusescorePopDataset(raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)
        prob_pieces = ['0858', '0749', '0634', '0654', '0362', '0409', '0517', '0461', '0404', '0850', '0339', '0595', '0822', '0471', '0431', '0643']
        super(MusescorePopPolyVoiceSeparationDataset, self).__init__(dataset_base=dataset_base, force_reload=force_reload, nprocs=nprocs, max_size=max_size, prob_pieces=prob_pieces)
        # display some data on the number of pieces that was discarded
        print("MusescorePopPolyVoiceSeparationDataset loaded.")
        print(f"Available pieces: {len(self.graphs)} from the original {len(self.dataset_base.scores)} scores")
        print(f"{len(self.dataset_base.scores) - len(self.graphs)} scores were discarded due to exceptions during loading.")
        print("Score discarded:")
        orig_names = [f"{collection}_{Path(score_fn).stem}" for score_fn,collection in zip(self.dataset_base.scores,self.dataset_base.collections)]
        proc_names = [g.name for g in self.graphs]
        discarded = [score for score in orig_names if score not in proc_names]
        self.test_files = [
            'musescore_pop_0227', 'musescore_pop_0626', 'musescore_pop_0256', 'musescore_pop_0053', 'musescore_pop_0693',
            'musescore_pop_0739', 'musescore_pop_0150', 'musescore_pop_0385', 'musescore_pop_0430', 'musescore_pop_0518',
            'musescore_pop_0781', 'musescore_pop_0694', 'musescore_pop_0483', 'musescore_pop_0575', 'musescore_pop_0730',
            'musescore_pop_0041', 'musescore_pop_0261', 'musescore_pop_0013', 'musescore_pop_0718', 'musescore_pop_0124',
            'musescore_pop_0621', 'musescore_pop_0488', 'musescore_pop_0489', 'musescore_pop_0025', 'musescore_pop_0361',
            'musescore_pop_0180', 'musescore_pop_0539', 'musescore_pop_0690', 'musescore_pop_0509', 'musescore_pop_0342',
            'musescore_pop_0200', 'musescore_pop_0836', 'musescore_pop_0802', 'musescore_pop_0282', 'musescore_pop_0117',
            'musescore_pop_0524', 'musescore_pop_0100', 'musescore_pop_0661', 'musescore_pop_0076', 'musescore_pop_0011',
            'musescore_pop_0748', 'musescore_pop_0035', 'musescore_pop_0116', 'musescore_pop_0666', 'musescore_pop_0787',
            'musescore_pop_0784', 'musescore_pop_0102', 'musescore_pop_0210', 'musescore_pop_0089', 'musescore_pop_0754',
            'musescore_pop_0835', 'musescore_pop_0307', 'musescore_pop_0580', 'musescore_pop_0552', 'musescore_pop_0560',
            'musescore_pop_0340', 'musescore_pop_0343', 'musescore_pop_0492', 'musescore_pop_0668', 'musescore_pop_0477',
            'musescore_pop_0558', 'musescore_pop_0283', 'musescore_pop_0573', 'musescore_pop_0175', 'musescore_pop_0162',
            'musescore_pop_0677', 'musescore_pop_0318', 'musescore_pop_0292', 'musescore_pop_0667', 'musescore_pop_0218',
            'musescore_pop_0325', 'musescore_pop_0310', 'musescore_pop_0583', 'musescore_pop_0711', 'musescore_pop_0502',
            'musescore_pop_0684', 'musescore_pop_0043', 'musescore_pop_0574', 'musescore_pop_0006', 'musescore_pop_0168',
            'musescore_pop_0632', 'musescore_pop_0589', 'musescore_pop_0462', 'musescore_pop_0496', 'musescore_pop_0516',
            'musescore_pop_0332', 'musescore_pop_0616', 'musescore_pop_0847', 'musescore_pop_0670', 'musescore_pop_0501',
            'musescore_pop_0448', 'musescore_pop_0284', 'musescore_pop_0796', 'musescore_pop_0533', 'musescore_pop_0188',
            'musescore_pop_0557', 'musescore_pop_0301', 'musescore_pop_0221', 'musescore_pop_0568', 'musescore_pop_0597',
            'musescore_pop_0274', 'musescore_pop_0660', 'musescore_pop_0242', 'musescore_pop_0799', 'musescore_pop_0045',
            'musescore_pop_0542', 'musescore_pop_0189', 'musescore_pop_0160', 'musescore_pop_0810', 'musescore_pop_0532',
            'musescore_pop_0442', 'musescore_pop_0186', 'musescore_pop_0449', 'musescore_pop_0618', 'musescore_pop_0764',
            'musescore_pop_0750', 'musescore_pop_0653', 'musescore_pop_0681', 'musescore_pop_0527', 'musescore_pop_0121',
            'musescore_pop_0295', 'musescore_pop_0798', 'musescore_pop_0423', 'musescore_pop_0098', 'musescore_pop_0398',
            'musescore_pop_0549', 'musescore_pop_0046', 'musescore_pop_0259', 'musescore_pop_0725', 'musescore_pop_0507',
            'musescore_pop_0130', 'musescore_pop_0040', 'musescore_pop_0028', 'musescore_pop_0265', 'musescore_pop_0440',
            'musescore_pop_0548', 'musescore_pop_0608', 'musescore_pop_0472', 'musescore_pop_0408', 'musescore_pop_0159',
            'musescore_pop_0233', 'musescore_pop_0226', 'musescore_pop_0758', 'musescore_pop_0248', 'musescore_pop_0581',
            'musescore_pop_0138', 'musescore_pop_0545', 'musescore_pop_0768', 'musescore_pop_0383', 'musescore_pop_0447',
            'musescore_pop_0024', 'musescore_pop_0021', 'musescore_pop_0467', 'musescore_pop_0357', 'musescore_pop_0133',
            'musescore_pop_0270', 'musescore_pop_0526', 'musescore_pop_0688', 'musescore_pop_0303']
        print(discarded)
