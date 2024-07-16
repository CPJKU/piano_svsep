from pathlib import Path
import json


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