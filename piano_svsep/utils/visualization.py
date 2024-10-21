from pathlib import Path
import json


def save_pyg_graph_as_json(graph, ids, path="./"):
    """Save the graph as a json file.

    Args:
        graph (torch_geometric.data.HeteroData): the graph to save
        ids (list): the ids of the nodes
        path (str, optional): the path to save the file. Defaults to "./".
    """

    # some renaming for better readability
    # it will become useless when the graph edge types will be updated
    renaming_dict = {'potential':'voice_candidate',
                    'chord_potential':'chord_candidate',	
                    'truth':'voice_truth',
                    'chord_truth':'chord_truth',
                    'predicted':'voice_output',
                    'chord_predicted':'chord_output'
                    }

    out_dict = {}
    for k,v in graph.edge_index_dict.items():
        new_k = renaming_dict[k[1]] if k[1] in renaming_dict else k[1]
        out_dict[new_k] = v.tolist()

    # export the nodes ids
    if "_" in ids[0]: # MEI with multiple parts, remove the Pxx_ prefix
        out_dict["id"] = [i.split("_")[1] for i in ids]
    else:
        out_dict["id"] = ids.tolist()
    
    with open(Path(path,graph.name + ".json"), "w") as f:
        print("Saving graph to", Path(path,graph.name + ".json"))
        json.dump(out_dict, f)