import torch
import numpy as np
import scipy as sp
import partitura
import torch_geometric as pyg
from typing import Tuple, List
import partitura as pt
import torch.nn as nn
from torch_scatter import scatter_add
from scipy.sparse import coo_matrix
from torchmetrics import Accuracy

from piano_svsep.postprocessing import PostProcessPooling
from scipy.optimize import linear_sum_assignment



def get_pc_one_hot(part, note_array):
    """Get one-hot encoding of pitch classes."""
    one_hot = np.zeros((len(note_array), 12))
    idx = (np.arange(len(note_array)),np.remainder(note_array["pitch"], 12))
    one_hot[idx] = 1
    return one_hot, ["pc_{:02d}".format(i) for i in range(12)]


def get_full_pitch_one_hot(part, note_array, piano_range = True):
    """Get one-hot encoding of all pitches."""
    one_hot = np.zeros((len(note_array), 127))
    idx = (np.arange(len(note_array)),note_array["pitch"])
    one_hot[idx] = 1
    if piano_range:
        one_hot = one_hot[:, 21:109]
    return one_hot, ["pc_{:02d}".format(i) for i in range(one_hot.shape[1])]


def get_octave_one_hot(part, note_array):
    """Get one-hot encoding of octaves."""
    one_hot = np.zeros((len(note_array), 10))
    idx = (np.arange(len(note_array)), np.floor_divide(note_array["pitch"], 12))
    one_hot[idx] = 1
    return one_hot, ["octave_{:02d}".format(i) for i in range(10)]


def get_vocsep_features(part, return_names=False) -> Tuple[np.ndarray, List]:
    """
    Computes Voice Detection features.

    Parameters
    ----------
    part: structured note array or partitura score part
        The partitura part or note array of the score.
    return_names: bool, optional
        Whether to return the feature names, by default False.

    Returns
    -------
    out : np.ndarray
    feature_fn : List
    """
    if isinstance(part, pt.performance.PerformedPart):
        perf_array = part.note_array()
        x = perf_array[["onset_sec", "duration_sec"]].astype([("onset_beat", "f4"), ("duration_beat", "f4")])
        note_array = np.lib.recfunctions.merge_arrays((perf_array, x))
    elif isinstance(part, np.ndarray):
        note_array = part
        part = None
    else:
        note_array = part.note_array(include_time_signature=True)

    octave_oh, octave_names = get_octave_one_hot(part, note_array)
    pc_oh, pc_names = get_pc_one_hot(part, note_array)
    duration_feature = np.expand_dims(1 - np.tanh(note_array["duration_beat"]/note_array["ts_beats"]), 1)
    dur_names = ["bar_exp_duration"]
    names = dur_names + pc_names + octave_names
    out = np.hstack((duration_feature, pc_oh, octave_oh))
    if return_names:
        return out, names
    return out


def get_mcma_potential_edges(hg, max_dist=16):
    """Get potential edges for the MCMADataset."""
    # Compute which edge to use for prediction.
    onset_edges = hg.get_edges_of_type("onset")
    during_edges = hg.get_edges_of_type("during")
    consecutive_edges = hg.get_edges_of_type("consecutive")
    consecutive_dense = torch.sparse_coo_tensor(consecutive_edges, torch.ones(consecutive_edges.shape[1]),
                                                size=(len(hg.x), len(hg.x))).to_dense()
    predsub_edges = torch.cat((onset_edges, during_edges), dim=1)
    trim_adj = torch.sparse_coo_tensor(predsub_edges, torch.ones(predsub_edges.shape[1]),
                                       size=(len(hg.x), len(hg.x)))
    trim_adj = trim_adj.to_dense()
    # Remove onset and during edges from a full adjacency matrix.
    trim_adj = torch.ones((len(hg.x), len(hg.x))) - trim_adj
    # Take only the upper triangular part of the adjacency matrix.
    # without the self loops (diagonal=1)
    trim_adj = torch.triu(trim_adj, diagonal=1)
    # remove indices that are further than x units apart.
    trim_adj = trim_adj - torch.triu(trim_adj, diagonal=max_dist)
    # readd consecutive edges if they were deleted
    trim_adj[consecutive_dense == 1] = 1
    # transform to edge index
    pot_edges = pyg.utils.sparse.dense_to_sparse(trim_adj)[0]
    return pot_edges


def get_mcma_truth_edges(note_array):
    """Get the ground truth edges for the MCMA dataset.
    Parameters
    ----------
    note_array : np.array
        The note array of the score.
    Returns
    -------
    truth_edges : torch.tensor (2, n)
        Ground truth edges.
    """
    part_ids = np.char.partition(note_array["id"], sep="_")[:, 0]
    truth_edges = list()
    # Append edges for consecutive notes in the same voice.
    for un in np.unique(part_ids):
        # Sort indices which are in the same voice.
        voc_inds = np.sort(np.where(part_ids == un)[0])
        # edge indices between consecutive notes in the same voice.
        truth_edges.append(np.vstack((voc_inds[:-1], voc_inds[1:])))
    truth_edges = np.hstack(truth_edges)
    return torch.tensor(truth_edges)


def get_pot_chord_edges(note_array, onset_edges):
    """Get edges connecting notes with same onset and duration"""
    # checking only the duration, the same onset is already checked in the onset edges
    same_duration_mask = note_array[onset_edges[0]]["duration_div"] == note_array[onset_edges[1]]["duration_div"]
    return onset_edges[:, same_duration_mask]


def get_truth_chords_edges(note_array, pot_chord_edges):
    """Get edges connecting notes with same onset and duration and voice and staff"""
    same_voice_mask = note_array[pot_chord_edges[0]]["voice"] == note_array[pot_chord_edges[1]]["voice"]
    same_staff_mask = note_array[pot_chord_edges[0]]["staff"] == note_array[pot_chord_edges[1]]["staff"]
    return pot_chord_edges[:, same_voice_mask & same_staff_mask]


def get_measurewise_pot_edges(note_array, measure_notes):
    """Get potential edges for the polyphonic voice separation dataset.
    Parameters
    ----------
    note_array : np.array
        The note array of the score.
    measure_notes : np.array
        The measure number of each note.
    Returns
    -------
    pot_edges : np.array (2, n)
        Potential edges.
    """
    pot_edges = list()
    for un in np.unique(measure_notes):
        # Sort indices which are in the same voice.
        voc_inds = np.sort(np.where(measure_notes == un)[0])
        # edge indices between all pairs of notes in the same measure (without self loops). size of (2, n)
        edges = np.vstack((np.repeat(voc_inds, len(voc_inds)), np.tile(voc_inds, len(voc_inds))))

        # remove edges whose end onset is before start offset
        not_after_mask = note_array["onset_div"][edges[1]] >= note_array["onset_div"][edges[0]] + \
                         note_array["duration_div"][edges[0]]
        # # remove edges that are not during the same note
        # during_mask = (note_array["onset_div"]+note_array["duration_div"])[edges[0]] < note_array["onset_div"][edges[1]]
        # apply all masks
        edges = edges[:, not_after_mask]
        pot_edges.append(edges)
    pot_edges = np.hstack(pot_edges)
    # remove self loops
    self_loop_mask = pot_edges[0] != pot_edges[1]
    pot_edges = pot_edges[:, self_loop_mask]
    return pot_edges


def sanitize_staff_voices(note_array):
    # case when there are two parts, and each one have only one staff.
    if len(np.unique(note_array["staff"])) == 1:
        # id is in the form "P01_something", extract the number before the underscore and after P
        staff_from_id = np.char.partition(np.char.partition(note_array["id"], sep="_")[:, 0], sep="P")[:, 2].astype(int)
        note_array["staff"] = staff_from_id + 1  # staff is 1-indexed
    # check if only two staves exist
    if len(np.unique(note_array["staff"])) != 2:
        raise Exception("After sanitizing, the score has", len(np.unique(note_array["staff"])),
                        "staves but it must have only 2.")
    # sometimes staff numbers are shifted. Shift them back to 1-2
    if note_array["staff"].min() != 1:
        note_array["staff"] = note_array["staff"] - note_array["staff"].min() + 1
    # check if they are between 1 and 2
    if note_array["staff"].min() != 1:
        raise Exception(f"After sanitizing, the minimum staff is {note_array['staff'].min()} but it should be 1")
    if note_array["staff"].max() != 2:
        raise Exception(f"After sanitizing, the maximum staff is {note_array['staff'].max()} but it should be 2")
    # check that there are no None voice values.
    if np.any(note_array["voice"] == None):
        raise Exception("Note array contains None voice values.")
    return note_array


def get_polyphonic_truth_edges(note_array, original_ids):
    """
    Extract ground truth voice edges for polyphonic voice separation.

    Add a tuple (ind_start,ind_end) to truth_edges list, where ind_start, ind_end
    are indices in the note_array, if the respective notes are consecutive with
    with the same voice and staff.
    Consecutive means that there is not another note with the same staff and voice whose onset is
    between the two notes.

    Parameters
    ----------
    note_array: structured array
        The note array of the score. Should contain staff, voice, onset_div, and duration_div.
    original_ids: list
        The original ids of the full note array. Necessary because this function is usualy called in a subset.

    Returns
    -------
    truth_edges: list
        Ground truth edges.
    """
    truth_edges = list()

    # build a square boolean matrix (n,n) to check the same voice
    voice_2dmask = note_array["voice"][:, np.newaxis] == note_array["voice"][np.newaxis, :]
    # the same for staff
    staff_2dmask = note_array["staff"][:, np.newaxis] == note_array["staff"][np.newaxis, :]
    # check if one note is after the other
    after_2dmask = (note_array["onset_div"] + note_array["duration_div"])[:, np.newaxis] <= note_array["onset_div"][
                                                                                            np.newaxis, :]
    # find the intersection of all masks
    voice_staff_after_2dmask = np.logical_and(np.logical_and(voice_2dmask, staff_2dmask), after_2dmask)

    # check if there is no other note before, i.e., there is not note with index ind_middle such as
    # note_array["onset"][ind_middle] < note_array["onset"][ind_end]
    # since the notes are only after by previous checks, this check that there are not notes in the middle
    for start_id, start_note in enumerate(note_array):
        # find the possible end notes left from the previous filtering
        possible_end_notes_idx = np.argwhere(voice_staff_after_2dmask[start_id])[:, 0]
        # find the notes with the smallest onset
        if len(possible_end_notes_idx) != 0:
            possible_end_notes_idx = possible_end_notes_idx[
                note_array[possible_end_notes_idx]["onset_div"] == note_array[possible_end_notes_idx][
                    "onset_div"].min()]
            # add the all couple (start_id, end_id) where end_id is in possible_end_notes_idx to truth_edges
            truth_edges.extend([(original_ids[start_id], original_ids[end_id]) for end_id in possible_end_notes_idx])

    return truth_edges


def get_edges_mask(subset_edges, total_edges, transpose=True, check_strict_subset=False):
    """Get a mask of edges to use for training.
    Parameters
    ----------
    subset_edges : np.array
        A subset of total_edges.
    total_edges : np.array
        Total edges.
    transpose : bool, optional.
        Whether to transpose the subset_edges, by default True.
        This is necessary if the input arrays are (2, n) instead of (n, 2)
    check_strict_subset : bool, optional
        Whether to check that the subset_edges are a strict subset of total_edges.
    Returns
    -------
    edges_mask : np.array
        Mask that identifies subset edges from total_edges.
    dropped_edges : np.array
        Truth edges that are not in potential edges.
        This is only returned if check_strict_subset is True.
    """
    # convert to numpy, custom types are not supported by torch
    total_edges = total_edges.numpy() if not isinstance(total_edges, np.ndarray) else total_edges
    subset_edges = subset_edges.numpy() if not isinstance(subset_edges, np.ndarray) else subset_edges
    # transpose if r; contiguous is required for the type conversion step later
    if transpose:
        total_edges = np.ascontiguousarray(total_edges.T)
        subset_edges = np.ascontiguousarray(subset_edges.T)
    # convert (n, 2) array to an n array of bytes, in order to use isin, that only works with 1d arrays
    # view_total = total_edges.view(np.dtype((np.void, total_edges.dtype.itemsize * total_edges.shape[-1])))
    # view_subset = subset_edges.view(np.dtype((np.void, subset_edges.dtype.itemsize * subset_edges.shape[-1])))
    view_total = np.char.array(total_edges.astype(str))
    view_subset = np.char.array(subset_edges.astype(str))
    view_total = view_total[:, 0] + "-" + view_total[:, 1]
    view_subset = view_subset[:, 0] + "-" + view_subset[:, 1]
    if check_strict_subset:
        dropped_edges = subset_edges[(~np.isin(view_subset, view_total))]
        if dropped_edges.shape[0] > 0:
            print(f"{dropped_edges.shape[0]} truth edges are not part of potential edges")
        return torch.tensor(np.isin(view_total, view_subset)).squeeze(), dropped_edges
    else:
        return torch.tensor(np.isin(view_total, view_subset)).squeeze()


def preprocess_na_to_monophonic(note_array, score_fn, drop_extra_voices=True, drop_chords=True):
    """Preprocess the note array to remove polyphonic artifacts.
    Parameters
    ----------
    note_array : np.array
        The note array of the score.
        score_fn : str
        The score filename.
    drop_extra_voices : bool, optional
        Whether to drop extra voices in parts, by default True.
    drop_chords : bool, optional
        Whether to drop chords all notes in chords except the highest, by default True.
        Returns
        -------
    note_array : np.array
        The preprocessed note array.
    """
    num_dropped = 0
    if drop_chords and not drop_extra_voices:
        raise ValueError("Drop chords work correctly only if drop_extra_voices is True.")
    if drop_extra_voices:
        # Check how many voices per part:
        num_voices_per_part = np.count_nonzero(note_array["voice"] > 1)
        if num_voices_per_part > 0:
            print("More than one voice on part of score: {}".format(score_fn))
            print("Dropping {} notes".format(num_voices_per_part))
            num_dropped += num_voices_per_part
            note_array = note_array[note_array["voice"] == 1]
    if drop_chords:
        ids_to_drop = []
        part_ids = np.char.partition(note_array["id"], sep="_")[:, 0]
        for id in np.unique(part_ids):
            part_na = note_array[part_ids == id]
            for onset in np.unique(part_na["onset_div"]):
                if len(part_na[part_na["onset_div"] == onset]) > 1:
                    to_drop = list(part_na[part_na["onset_div"] == onset]["id"][:-1])
                    num_dropped += len(to_drop)
                    ids_to_drop.extend(to_drop)
                    print("Dropping {} notes from chord in score: {}".format(len(to_drop), score_fn))
        return note_array[~np.isin(note_array["id"], ids_to_drop)], num_dropped


def get_measurewise_truth_edges(note_array, note_measure):
    """Create groundtruth edges for polyphonic music.

    The function creates edges between consecutive notes in the same voice.

    The groundtruth is restricted per bar (measure) by the score typesetting.
    Voices can switch between bars.

    Parameters
    ----------
    note_array : np.structured array
        The partitura note array

    measure_notes: np.array
        A array with the measure number for each note in the note_array

    Returns
    -------
    edges : np.array (2, n)
        Ground truth edges.
    """
    # bring all note arrays to a common format, with a single part and 2 staves
    note_array = sanitize_staff_voices(note_array)

    edges = list()
    # split note_array to measures where every onset_div is within range of measure start and end
    measurewise_na_list = np.split(note_array, np.where(np.diff(note_measure))[0] + 1)
    measurewise_original_ids = np.split(np.arange(len(note_array)), np.where(np.diff(note_measure))[0] + 1)
    for measurewise_na, original_id in zip(measurewise_na_list, measurewise_original_ids):
        bar_edges = get_polyphonic_truth_edges(measurewise_na, original_id)
        edges.extend(bar_edges)
    return np.array(edges).T


def remove_ties_acros_barlines(score, return_ids=True):
    """Remove ties that are across barlines.

    This function don't return anything since the score will be modified in place.
    """
    src_ids = []
    dst_ids = []
    for part in score.parts:
        measure_map = part.measure_number_map
        # iterate over all notes in the part to currate start time
        for note in part.iter_all(partitura.score.Note):
            if note.tie_next is not None:
                if measure_map(note.start.t) != measure_map(note.tie_next.start.t):
                    src = note.id
                    dst = note.tie_next.id
                    src_ids.append(src)
                    dst_ids.append(dst)
                    note.tie_next.tie_prev = None
                    note.tie_next = None
            if note.tie_prev is not None:
                if measure_map(note.start.t) != measure_map(note.tie_prev.start.t):
                    src = note.id
                    dst = note.tie_prev.id
                    src_ids.append(src)
                    dst_ids.append(dst)
                    note.tie_prev.tie_next = None
                    note.tie_prev = None
                if note.end.t != note.tie_prev.start.t:
                    note.tie_prev.tie_next = None
                    note.tie_prev = None

    if return_ids:
        return np.array([src_ids, dst_ids])
    return


def remove_ties_to_partial_chords(score):
    """Remove ties that are across barlines.

    This function don't return anything since the score will be modified in place.
    """
    for part in score.parts:
        for note in part.notes:
            if note.tie_next is not None:
                next_note = note.tie_next
                for pt_chord_note in part.iter_all(partitura.score.Note, note.start.t, note.start.t + 1):
                    if pt_chord_note != next_note:
                        if pt_chord_note.voice == next_note.voice and pt_chord_note.staff == next_note.staff:
                            next_note.tie_prev = None
                            note.tie_next = None
                            break


def assign_voices(part, predicted_voice_edges: torch.LongTensor, predicted_staff: torch.LongTensor):
    """
    Assign voices to the notes of a partitura part based on the predicted edges

    Parameters
    ----------
    part: partitura.Part
        Part to assign voices to
    predicted_voice_edges: torch.LongTensor
        Predicted voice edges of size (2, N) where N is the number of edges
    predicted_staff: torch.LongTensor
        Predicted staff labels (binary) of size (M,) where M is the number of notes.
    """
    
    predicted_staff = predicted_staff.detach().cpu().numpy().astype(int) + 1 # (make staff start from 1)
    note_array = part.note_array()
    assert len(part.notes_tied) == len(note_array)
    # sort the notes by the note.id to match the order of the note_array["id"] which was used as the input to the model
    for i, note in enumerate(part.notes_tied):
        note.staff = int(predicted_staff[np.where(note_array["id"] == note.id)[0][0]])
    
    # recompute note_array to include the now newly added staff
    note_array = part.note_array(include_staff=True)
    preds = predicted_voice_edges.detach().cpu().numpy()
    # build the adjacency matrix
    graph = sp.sparse.csr_matrix((np.ones(preds.shape[1]), (preds[0], preds[1])), shape=(len(note_array), len(note_array)))
    n_components, voice_assignment = sp.sparse.csgraph.connected_components(graph, directed=True, return_labels=True)
    voice_assignment = voice_assignment.astype(int)
    for measure_idx, measure in enumerate(part.measures):
        note_idx = np.where((note_array['onset_div'] >= measure.start.t) & (note_array['onset_div'] < measure.end.t))[0]       
        voices_per_measure = voice_assignment[note_idx]
        # Re-index voices to start from 1 and be consecutive for each measure
        unique_voices = np.unique(voices_per_measure)
        voices = np.zeros(n_components, dtype=int)
        voices[unique_voices] = np.arange(1, len(unique_voices) + 1, dtype=int)
        voices_per_measure = voices[voices_per_measure]
        # assert len(unique_voices) < 8, f"More than 8 voices detected in measure {measure_idx}"
        note_array_seg = note_array[note_idx]
        note_array_seg["voice"] = voices_per_measure
        note_array_ends = note_array_seg["onset_div"] + note_array_seg["duration_div"]
        # group note_array_seg by voice find the start and end of each voice
        unique_new_voices = np.unique(voices_per_measure)
        voice_start = np.zeros(len(unique_new_voices))
        voice_end = np.zeros(len(unique_new_voices))
        staff_for_grouping = np.zeros(len(unique_new_voices))

        for i in range(len(unique_new_voices)):
            vidxs = np.where(note_array_seg["voice"] == unique_new_voices[i])[0]
            voice_start[i] = note_array_seg["onset_div"][vidxs].min()
            voice_end[i] = note_array_ends[vidxs].max()
            staff_of_voice = note_array_seg["staff"][vidxs]
            # if there are more than one staff in the voice, assign the staff number to -1
            if len(np.unique(staff_of_voice)) > 1:
                staff_for_grouping[i] = -1
            else:
                staff_for_grouping[i] = staff_of_voice[0]

        # if for any two voices, voice_end - voice_start > 0, then the voices could be merged
        # if the staff of the two voices are the same
        for i in range(len(unique_new_voices)):
            for j in range(len(unique_new_voices)):
                if i == j:
                    continue
                if staff_for_grouping[i] == -1 or staff_for_grouping[j] == -1:
                    continue
                if staff_for_grouping[i] != staff_for_grouping[j]:
                    continue
                if voice_end[i] <= voice_start[j]:
                    # merge the two voices
                    note_array_seg["voice"][np.where(note_array_seg["voice"] == unique_new_voices[j])[0]] = unique_new_voices[i]
                    voice_end[i] = voice_end[j]
                    voice_start[j] = voice_start[i]

        voices_per_measure = note_array_seg["voice"]

        # group note_array_seg by voice and get the mean of the pitches
        unique_new_voices = np.unique(voices_per_measure)
        mean_pitch_per_voice = np.zeros(len(unique_new_voices))
        for i in range(len(unique_new_voices)):
            mean_pitch_per_voice[i] = note_array_seg["pitch"][np.where(note_array_seg["voice"] == unique_new_voices[i])[0]].mean()

        # re-assign the voice numbers based on the mean pitch
        voice_order = np.argsort(mean_pitch_per_voice)[::-1]
        old_voices_per_measure = voices_per_measure.copy()
        for i in range(len(unique_new_voices)):
            voices_per_measure[np.where(old_voices_per_measure == unique_new_voices[voice_order[i]])[0]] = i + 1

        # set the voice attribute of the notes
        for single_note_idx, est_voice in zip(note_idx, voices_per_measure):
            if part.notes_tied[single_note_idx].id == note_array["id"][single_note_idx]:
                part.notes_tied[single_note_idx].voice = est_voice
            else:
                for note in part.notes_tied:
                    if note.id == note_array["id"][single_note_idx]:
                        note.voice = est_voice
                        break


def infer_vocstaff_algorithm(graph, return_score=True, normalize=True, return_graph=False):
    """
    This function infers voice and staff assignments for a given musical graph.

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



def linear_assignment(edge_pred_mask_prob, pot_edges, num_notes, threshold=0.5):
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
    mask_over_potential = new_probs > threshold
    pred_edges = pot_edges[:, mask_over_potential]
    return pred_edges


def compute_voice_f1_score(pred_edges, truth_edges, num_nodes):
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


def isin_pairwise(element, test_elements, assume_unique=True):
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

