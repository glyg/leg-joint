import sys
sys.path.append('/home/guillaume/Python/hdfgraph')
from graph_tool import load_graph
import hdfgraph

def import_graph(fname):
    """
    """
    vertices_df, edges_df = hdfgraph.frames_from_hdf(fname)
    return vertices_df, edges_df
