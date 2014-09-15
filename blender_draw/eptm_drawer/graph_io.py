import sys
sys.path.append('/usr/lib/python3/dist-packages/')
import graph_tool.all as gt
import hdfgraph

def import_graph(fname):
    """
    """

    return gt.load_graph(fname)
