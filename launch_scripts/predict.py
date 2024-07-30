import os.path
import partitura as pt
import partitura.score as spt
import numpy as np
import torch
import torch_geometric as pyg
from piano_svsep.models.pl_models import PLPianoSVSep
from piano_svsep.utils import assign_voices, infer_vocstaff_algorithm
from piano_svsep.utils.visualization import save_pyg_graph_as_json
from piano_svsep.utils import (
    hetero_graph_from_note_array,
    get_vocsep_features,
    score_graph_to_pyg,
    HeteroScoreGraph,
    remove_ties_acros_barlines,
    get_measurewise_pot_edges,
    get_pot_chord_edges,
    )
from lxml import etree
import argparse


def prepare_score(path_to_score, exclude_grace=True):
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

    # Remove Grace_notes
    if exclude_grace:
        for part in score:
            grace_notes = list(part.iter_all(pt.score.GraceNote))
            for grace_note in grace_notes:
                part.remove(grace_note)

    note_array = score[0].note_array(
        include_time_signature=True,
        include_grace_notes=True, # this is just to check that there are not grace notes left
        include_staff=True,
    )
    # get the measure number for each note in the note array
    mn_map = score[np.array([p._quarter_durations[0] for p in score]).argmax()].measure_number_map
    note_measures = mn_map(note_array["onset_div"])
    nodes, edges = hetero_graph_from_note_array(note_array, pot_edge_dist=0)
    note_features = get_vocsep_features(note_array)
    hg = HeteroScoreGraph(
        note_features,
        edges,
        name="test_graph",
        labels=None,
        note_array=note_array,
    )
    pot_edges = get_measurewise_pot_edges(note_array, note_measures)
    pot_chord_edges = get_pot_chord_edges(note_array, hg.get_edges_of_type("onset").numpy())
    setattr(hg, "pot_edges", torch.tensor(pot_edges))
    setattr(hg, "pot_chord_edges", torch.tensor(pot_chord_edges))
    pg_graph = score_graph_to_pyg(hg)
    return pg_graph, score, tie_couples


def predict_voice(path_to_model, path_to_score, save_path=None):
    # Load the model
    pl_model = PLPianoSVSep.load_from_checkpoint(path_to_model, map_location="cpu")
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
    correct_and_save_mei(part,save_path)


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


def correct_and_save_mei(part,save_path):
    # Correct the mei file reassigning voices so each staff voice start from 1
    mei_string = pt.save_mei(part)
    # load the mei file
    mei = etree.fromstring(mei_string)
    # get namespace
    nsmap = mei.nsmap
    nsmap['def'] = nsmap.pop(None)
    # iterate over all measure element
    for measure in mei.xpath("//def:measure", namespaces=nsmap):
        # get all  layer elements inside all occurrencies of <staff n="2">
        staff2_layers = measure.xpath("def:staff[@n='2']/def:layer", namespaces=nsmap)
        # get the n attribute for all layers
        staff2_layers_n = [int(layer.get("n")) for layer in staff2_layers]
        # get the mimumum n attribute
        min_n = min(staff2_layers_n)
        # subtract the minimum n attribute from all layers in staff 2
        for layer in staff2_layers:
            layer.set("n", str(int(layer.get("n")) - min_n + 1))
    # save the corrected mei file
    with open(save_path, "w") as f:
        f.write(etree.tostring(mei, pretty_print=True).decode("utf-8"))



if __name__ == "__main__":
    basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(basepath, "pretrained_models", "model.ckpt")
    score_path = os.path.join(basepath, "artifacts", "test_score.musicxml")
    score_name = os.path.splitext(os.path.basename(score_path))[0]
    predict_voice(model_path, score_path, os.path.join(basepath, "artifacts", f"{score_name}_pred.mei"))
