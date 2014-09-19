import hdfgraph

def import_graph(fname):
    """
    """
    vertices_df, edges_df = hdfgraph.frames_from_hdf(fname)
    return vertices_df, edges_df

def load_vertices(fname, start_stamp, stop_stamp, cols=None):

    vertices = hdfgraph.vertices_time_slice(fname, start_stamp, stop_stamp)
    if cols is not None:
        vertices = vertices[cols]
    return vertices.swaplevel(0, 1).sortlevel()

def load_edges(fname, start_stamp, stop_stamp, cols=None):

    edges = hdfgraph.edges_time_slice(fname, start_stamp, stop_stamp)
    if cols is not None:
        edges = edges[cols]
    return edges.swaplevel(0,1).swaplevel(1,2).sortlevel()
