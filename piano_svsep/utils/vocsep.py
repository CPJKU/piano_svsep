import torch
import numpy as np
import partitura
import torch_geometric as pyg
from typing import Tuple, List
import partitura as pt


def get_pc_one_hot(part, note_array):
    one_hot = np.zeros((len(note_array), 12))
    idx = (np.arange(len(note_array)),np.remainder(note_array["pitch"], 12))
    one_hot[idx] = 1
    return one_hot, ["pc_{:02d}".format(i) for i in range(12)]


def get_full_pitch_one_hot(part, note_array, piano_range = True):
    one_hot = np.zeros((len(note_array), 127))
    idx = (np.arange(len(note_array)),note_array["pitch"])
    one_hot[idx] = 1
    if piano_range:
        one_hot = one_hot[:, 21:109]
    return one_hot, ["pc_{:02d}".format(i) for i in range(one_hot.shape[1])]


def get_octave_one_hot(part, note_array):
    one_hot = np.zeros((len(note_array), 10))
    idx = (np.arange(len(note_array)), np.floor_divide(note_array["pitch"], 12))
    one_hot[idx] = 1
    return one_hot, ["octave_{:02d}".format(i) for i in range(10)]


def get_vocsep_features(part, return_names=False) -> Tuple[np.ndarray, List]:
    """
    Returns features Voice Detection features.

    Parameters
    ----------
    part: structured note array or partitura score part

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

    # octave_oh, octave_names = get_octave_one_hot(part, note_array)
    # pc_oh, pc_names = get_pc_one_hot(part, note_array)
    # onset_feature = np.expand_dims(np.remainder(note_array["onset_beat"], note_array["ts_beats"]) / note_array["ts_beats"], 1)
    # on_feats, _ = pt.musicanalysis.note_features.onset_feature(note_array, part)
    # duration_feature = np.expand_dims(np.remainder(note_array["duration_beat"], note_array["ts_beats"]) / note_array["ts_beats"], 1)
    # # new attempt! To delete in case
    # # duration_feature = np.expand_dims(1- (1/(1+np.exp(-3*(note_array["duration_beat"]/note_array["ts_beats"])))-0.5)*2, 1)
    # pitch_norm = np.expand_dims(note_array["pitch"] / 127., 1)
    # on_names = ["barnorm_onset", "piecenorm_onset"]
    # dur_names = ["barnorm_duration"]
    # pitch_names = ["pitchnorm"]
    # names = on_names + dur_names + pitch_names + pc_names + octave_names
    # out = np.hstack((onset_feature, np.expand_dims(on_feats[:, 1], 1), duration_feature, pitch_norm, pc_oh, octave_oh))

    # octave_oh, octave_names = get_octave_one_hot(part, note_array)
    # pitch_oh, pitch_names = get_full_pitch_one_hot(part, note_array)
    # onset_feature = np.expand_dims(np.remainder(note_array["onset_beat"], note_array["ts_beats"]) / note_array["ts_beats"], 1)
    # on_feats, _ = pt.musicanalysis.note_features.onset_feature(note_array, part)
    octave_oh, octave_names = get_octave_one_hot(part, note_array)
    pc_oh, pc_names = get_pc_one_hot(part, note_array)
    # duration_feature = np.expand_dims(1- (1/(1+np.exp(-3*(note_array["duration_beat"]/note_array["ts_beats"])))-0.5)*2, 1)
    duration_feature = np.expand_dims(1 - np.tanh(note_array["duration_beat"]/note_array["ts_beats"]), 1)
    dur_names = ["bar_exp_duration"]
    # on_names = ["barnorm_onset", "piecenorm_onset"]
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
    return torch.from_numpy(truth_edges)


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
        # # remove self loops
        # self_loop_mask = edges[0] != edges[1]
        # remove edges whose end onset is before start offset
        not_after_mask = note_array["onset_div"][edges[1]] >= note_array["onset_div"][edges[0]] + \
                         note_array["duration_div"][edges[0]]
        # # remove edges that are not during the same note
        # during_mask = (note_array["onset_div"]+note_array["duration_div"])[edges[0]] < note_array["onset_div"][edges[1]]
        # apply all masks
        edges = edges[:, not_after_mask]
        pot_edges.append(edges)
    pot_edges = np.hstack(pot_edges)
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
        return torch.from_numpy(np.isin(view_total, view_subset)).squeeze(), dropped_edges
    else:
        return torch.from_numpy(np.isin(view_total, view_subset)).squeeze()


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
