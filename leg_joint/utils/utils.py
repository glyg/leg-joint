# -*- coding: utf-8 -*-

import numpy as np

from scipy.interpolate import splrep, splev


def to_xy(rho, theta):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def to_rhotheta(x, y):
    rho = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return rho, theta

def compute_distribution(prop_u, prop_v, bins, smth=0):

    hist_uv, bins_u, bins_v = np.histogram2d(prop_u,
                                             prop_v, bins=bins)
    bin_wu = bins_u[1] - bins_u[0]
    bin_wv = bins_v[1] - bins_v[0]

    regular_u = bins_u[:-1] + bin_wu/2.
    regular_v = bins_v[:-1] + bin_wv/2.

    Huv = hist_uv, bins_u, bins_v

    print('''bin width = %.3f''' % bin_wu)
    norm = hist_uv.sum(axis=1)
    mean_vfu = (regular_v * hist_uv).sum(axis=1) / hist_uv.sum(axis=1)
    tck_vfu = splrep(regular_u, mean_vfu, k=3, s=smth)
    return tck_vfu, Huv

def _local_subgraph(meth):
    def new_function(eptm, *args, **kwargs):
        from .epithelium import Epithelium
        from graph_tool import Graph, GraphView
        local_graph = Graph(GraphView(eptm.graph,
                                      vfilt=eptm.is_local_vert,
                                      efilt=eptm.is_local_edge),
                            prune=True)
        local_eptm = Epithelium(paramtree=eptm.paramtree,
                                graph=local_graph)
        # if local_eptm.at_boundary.fa.sum() > 0:
        #     local_eptm.rotate(np.pi)
        #     local_eptm.current_angle = np.pi
        #     print('rotated')
        out = meth(local_eptm, *args, **kwargs)
        # if not -1e-8 < local_eptm.current_angle < 1e-8:
        #     local_eptm.rotate(-local_eptm.current_angle)
        #     local_eptm.current_angle = 0.
        #     print( 'rotated back')
        eptm.graph.set_vertex_filter(eptm.is_local_vert)
        eptm.graph.set_edge_filter(eptm.is_local_edge)
        for key, val in local_eptm.graph.vertex_properties.items():
            eptm.graph.vertex_properties[key].fa = val.a
        for key, val in local_eptm.graph.edge_properties.items():
            eptm.graph.edge_properties[key].fa = val.a
        eptm.graph.set_vertex_filter(None)
        eptm.graph.set_edge_filter(None)
        eptm.reset_topology()

        return out
    return new_function


def local_subgraph(meth):
    def new_function(eptm, *args, **kwargs):
        eptm.graph.set_vertex_filter(eptm.is_local_vert)
        eptm.graph.set_edge_filter(eptm.is_local_edge)

        # if eptm.at_boundary.fa.sum() > 0:
        #     eptm.rotate(np.pi)
        #     eptm.current_angle = np.pi
        #     print('rotated')

        out = meth(eptm, *args, **kwargs)
        # if not -1e-8 < eptm.current_angle < 1e-8:
        #     eptm.rotate(-eptm.current_angle)
        #     eptm.current_angle = 0.
        #     print( 'rotated back')

        eptm.graph.set_vertex_filter(None)
        eptm.graph.set_edge_filter(None)

        return out
    return new_function

