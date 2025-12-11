import os.path
import partitura as pt
import partitura.score as spt
import numpy as np
import torch
import torch_geometric as pyg
from piano_svsep.models.pl_models import PLPianoSVSep, infer_vocstaff_algorithm
from piano_svsep.utils.visualization import save_pyg_graph_as_json
from piano_svsep.utils import (
    hetero_graph_from_note_array,
    get_vocsep_features,
    score_graph_to_pyg,
    HeteroScoreGraph,
    remove_ties_acros_barlines,
    get_measurewise_pot_edges,
    get_pot_chord_edges,
    get_truth_chords_edges,
    get_measurewise_truth_edges,
    assign_voices
    )
import argparse


def prepare_score(path_to_score, include_original=True):
    """
    Prepare the score for voice separation.

    Parameters
    ----------
    path_to_score: str
        Path to the score file. Partitura can handle different formats such as musicxml, mei, etc.
    include_original: bool, optional
        Whether to include the original voice and chord assignments in the graph. Defaults to True.
        Mostly used for visualization purposes.
    Returns
    -------
        pg_graph: torch_geometric.data.HeteroData
            PyG HeteroData object containing the score graph.
        score: partitura.score.Score
            Partitura Score object.
        tie_couples: np.ndarray
            Array of tied notes.
    """
    # Load the score
    score = pt.load_score(path_to_score, force_note_ids=True)
    if len(score) > 1:
        score = pt.score.Score(pt.score.merge_parts(score.parts))

    # Preprocess score for voice separation
    tie_couples = remove_ties_acros_barlines(score, return_ids=True)

    # Remove beams
    for part in score:
        beams = list(part.iter_all(pt.score.Beam))
        for beam in beams:
            beam_notes = beam.notes
            for note in beam_notes:
                note.beam = None
            part.remove(beam)

    # Remove rests
    for part in score:
        rests = list(part.iter_all(pt.score.Rest))
        for rest in rests:
            part.remove(rest)
        # Remove Tuplets that contain rests
        tuplets = list(part.iter_all(pt.score.Tuplet))
        for tuplet in tuplets:
            if isinstance(tuplet.start_note, pt.score.Rest) or isinstance(tuplet.end_note, pt.score.Rest):
                part.remove(tuplet)

    # Remove grace notes
    for part in score:
        grace_notes = list(part.iter_all(pt.score.GraceNote))
        for grace_note in grace_notes:
            part.remove(grace_note)

    # Create note array with necessary features
    note_array = score[0].note_array(
        include_time_signature=True,
        include_grace_notes=True, # this is just to check that there are not grace notes left
        include_staff=True,
    )

    # Get the measure number for each note in the note array
    mn_map = score[np.array([p._quarter_durations[0] for p in score]).argmax()].measure_number_map
    note_measures = mn_map(note_array["onset_div"])

    # Create heterogeneous graph from note array
    nodes, edges = hetero_graph_from_note_array(note_array, pot_edge_dist=0)
    note_features = get_vocsep_features(note_array)
    hg = HeteroScoreGraph(
        note_features,
        edges,
        name="test_graph",
        labels=None,
        note_array=note_array,
    )

    # Get potential edges
    pot_edges = get_measurewise_pot_edges(note_array, note_measures)
    pot_chord_edges = get_pot_chord_edges(note_array, hg.get_edges_of_type("onset").numpy())
    setattr(hg, "pot_edges", torch.tensor(pot_edges))
    setattr(hg, "pot_chord_edges", torch.tensor(pot_chord_edges))

    if include_original:
        # Get truth edges, also called truth when original voice assignment is wrong.
        truth_chords_edges = get_truth_chords_edges(note_array, pot_chord_edges)
        polyphonic_truth_edges = get_measurewise_truth_edges(note_array, note_measures)
        setattr(hg, "truth_chord_edges", torch.tensor(truth_chords_edges).long())
        setattr(hg, "truth_edges", torch.tensor(polyphonic_truth_edges).long())

    # Convert score graph to PyG graph
    pg_graph = score_graph_to_pyg(hg)

    return pg_graph, score, tie_couples


def predict_voice(path_to_model, path_to_score, save_path=None):
    """
    Predict the voice assignment for a given score using a pre-trained model.

    Parameters
    ----------
    path_to_model: str
        Path to the pre-trained model checkpoint.
    path_to_score: str
        Path to the score file. Partitura can handle different formats such as musicxml, mei, etc.
    save_path: str, optional
        Path to save the predicted score. If None, the predicted score will be saved in the same directory as the input score with '_pred' appended to the filename. Defaults to None.

    Returns
    -------
    None
        Updates are made to the score object and saved to the specified path.
    """
    # Load the model
    pl_model = PLPianoSVSep.load_from_checkpoint(path_to_model, map_location="cpu", strict=False, weights_only=False)
    # Prepare the score
    pg_graph, score, tied_notes = prepare_score(path_to_score)
    # Batch for compatibility
    pg_graph = pyg.data.Batch.from_data_list([pg_graph])
    # predict the voice assignment
    with torch.no_grad():
        pl_model.module.eval()
        pred_voices, pred_staff, pg_graph = pl_model.predict_step(pg_graph, return_graph=True)
    # Partitura processing for visualization
    part = score[0]
    save_path = save_path if save_path is not None else os.path.splitext(path_to_score)[0] + "_pred.mei"
    pg_graph.name = os.path.splitext(os.path.basename(save_path))[0]
    save_pyg_graph_as_json(pg_graph, ids=part.note_array()["id"], path=os.path.dirname(save_path))
    assign_voices(part, pred_voices, pred_staff)
    tie_notes_over_measures(part, tied_notes)
    spt.fill_rests(part, measurewise=True)
    spt.infer_beaming(part)
    print("Saving MEI score to", save_path)
    if save_path.endswith(".mei"):
        pt.save_mei(part,save_path)
    elif save_path.endswith(".musicxml") or save_path.endswith(".xml"):
        pt.save_musicxml(part, save_path)
    else:
        raise ValueError("Unsupported file format. Please use .mei or .musicxml/.xml")



def predict_voice_baseline(path_to_score, save_path=None):
    # Prepare the score
    pg_graph, score, tied_notes = prepare_score(path_to_score)
    # predict the voice assignment
    pred_voices, pred_staff, pg_graph = infer_vocstaff_algorithm(pg_graph, return_score=False, return_graph=True)
    # Partitura processing for visualization
    part = score[0]
    pg_graph.name = os.path.splitext(os.path.basename(save_path))[0]
    save_pyg_graph_as_json(pg_graph, ids=part.note_array()["id"], path=os.path.dirname(save_path))
    assign_voices(part, pred_voices, pred_staff)
    tie_notes_over_measures(part, tied_notes)
    spt.fill_rests(part, measurewise=True)
    spt.infer_beaming(part)
    save_path = save_path if save_path is not None else os.path.splitext(path_to_score)[0] + "_baseline_pred.mei"
    print("Saving MEI score to", save_path)
    pt.save_mei(part, save_path)


def tie_notes_over_measures(part, tied_notes):
    for src, dst in tied_notes.T:
        src_note = None
        dst_note = None
        for note in part.notes_tied:
            if note.id == dst:
                dst_note = note
                break
        for note in part.notes_tied:
            if note.id == src:
                src_note = note
                break
        if src_note is not None and dst_note is not None:
            src_note.tie_next = dst_note
            dst_note.tie_prev = src_note


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict voice assignment for a given score using a pre-trained model.")
    parser.add_argument("--model_path", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pretrained_models", "model.ckpt"), help="Path to the pre-trained model checkpoint.")
    parser.add_argument("--score_path", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "test_score.musicxml"), help="Path to the score file.")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the predicted score. If None, the predicted score will be saved in the same directory as the input score with '_pred' appended to the filename.")

    args = parser.parse_args()

    basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    score_name = os.path.splitext(os.path.basename(args.score_path))[0]
    save_path = args.save_path if args.save_path is not None else os.path.join(basepath, "artifacts", f"{score_name}_pred.mei")

    predict_voice(args.model_path, args.score_path, save_path)
    print("Done.")
