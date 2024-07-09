import partitura
import numpy as np
import plotly.express as px
import verovio
from xml.etree import ElementTree as ET
import torch
from pathlib import Path
import json

def show_voice_pr(pitches, onsets, durations, voices, time_unit, time_div, return_figure= False, colors = None):
    unique_voices = np.unique(voices)
    # create the structured arrays
    struct_array = np.zeros(len(voices), dtype={'names':('pitch', 'onset_beat', 'duration_beat'),
                          'formats':('i4', 'f4', 'f4')})
    struct_array["pitch"] = pitches
    struct_array["onset_beat"] = onsets
    struct_array["duration_beat"] = durations

    # create a pianoroll where each voice has a different value
    piano_rolls = []
    end_time = onsets[-1] + durations[-1]
    for i,voice_n in enumerate(unique_voices):
        pr = partitura.utils.compute_pianoroll(struct_array[voices==voice_n],piano_range =True, time_unit = time_unit, time_div=time_div, remove_silence = False, end_time = float(end_time))
        piano_rolls.append(pr.multiply(voice_n).todense())
    # this takes the maximum, meaning that if two voices share the same note, only the highest voice will be shown
    mixed_pr = np.maximum.reduce(piano_rolls)
    # sum_pr = np.sum(piano_rolls)
    # different_bins = np.nonzero(mixed_pr!=sum_pr)
    # if len(different_bins)!=0:
    #     print(different_bins)
    #     for e in different_bins:
    #         print("Double voice in bins", e)

    # a list of accented colors in such an order to seems 
    # colors = ["red","green","cyan", "blue",  "orange", "purple", "magenta","yellow"]
    if colors is None:
        colors = px.colors.sample_colorscale("turbo", [voice_n/(np.max(unique_voices) -1) for voice_n in range(np.max(unique_voices))])

    separators = [voice_n/np.max(unique_voices) for voice_n in unique_voices]
    epsilon = 0.001
    background_color = "rgba(0,0,0,0.0)"
    color_scale = [(0.0,background_color),(0+epsilon,background_color)]
    last_value = 0
    for i,sep in enumerate(separators):
        color_scale.append((last_value+epsilon,colors[i]))
        color_scale.append((sep,colors[i]))
        last_value = sep

    fig = px.imshow( mixed_pr, origin="lower", color_continuous_scale = color_scale)
    fig.show()
    if return_figure:
        return fig


def save_pyg_graph_as_json(graph, ids, path="./"):
    """Save the graph as a json file.

    Args:
        graph (torch_geometric.data.HeteroData): the graph to save
        ids (list): the ids of the nodes
        path (str, optional): the path to save the file. Defaults to "./".
    """
    out_dict = {}
    # for k,v in graph.__dict__.items():
    #     if isinstance(v, (np.ndarray,torch.Tensor)):
    #         out_dict[k] = v.tolist()
    #     elif isinstance(v, str):
    #         out_dict[k] = v
    # export the input edges
    for k,v in graph.edge_index_dict.items():
        out_dict[k[1]] = v.tolist()

    # export the output edges
    # truth edges
    # out_dict["output_edges_dict"]["truth"] = graph["truth_edges"].tolist()
    # # potential edges
    # out_dict["output_edges_dict"]["potential"] = graph["pot_edges"].tolist()

    # export the nodes ids
    if "_" in ids[0]: # MEI with multiple parts, remove the Pxx_ prefix
        out_dict["id"] = [i.split("_")[1] for i in ids]
    else:
        out_dict["id"] = ids.tolist()
    
    with open(Path(path,graph.name + ".json"), "w") as f:
        print("Saving graph to", Path(path,graph.name + ".json"))
        json.dump(out_dict, f)