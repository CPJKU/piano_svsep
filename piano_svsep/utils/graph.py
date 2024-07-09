import numpy as np


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
