# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import matplotlib.pyplot as plt
import json
import graph_tool.all as gt
import leg_joint as lj
import numpy as np

from numpy.testing import assert_almost_equal, assert_array_almost_equal
from leg_joint.objects import get_faces


import tempfile

tmp_dir = tempfile.mkdtemp(prefix='tmp_lj_')

def _get_cell(eptm, idx=2):
    return eptm.graph.vertex(idx)

def _get_jv(eptm, idx=100):
    return eptm.graph.vertex(idx)

def _get_je(eptm, tup=(100, 101)):
    return eptm.graph.edge(*tup)


def test_epithelium():
    eptm = lj.Epithelium(graphXMLfile=lj.data.small_xml(),
                         save_dir=tmp_dir,
                         identifier='test_epithelium',
                         copy=True)
    assert (56, 136) == (eptm.is_cell_vert.a.sum(), eptm.is_junction_edge.a.sum())

def test_cell_area():
    eptm = lj.Epithelium(graphXMLfile=lj.data.small_xml(),
                         save_dir=tmp_dir,
                         identifier='test_epithelium',
                         copy=True)
    cell = _get_cell(eptm)
    assert_almost_equal(eptm.cells.areas[cell],
                        12.396962310561445)
    eptm.isotropic_relax()
    assert_almost_equal(eptm.cells.areas[cell],
                        9.432559119945529)

def test_junction_lenght():
    eptm = lj.Epithelium(graphXMLfile=lj.data.small_xml(),
                         save_dir=tmp_dir,
                         identifier='test_epithelium',
                         copy=True)
    eptm.isotropic_relax()
    je =  _get_je(eptm)
    assert_almost_equal(eptm.edge_lengths[je],
                        1.9199463830814676)

def test_topology():
    eptm = lj.Epithelium(graphXMLfile=lj.data.small_xml(),
                         save_dir=tmp_dir,
                         identifier='test_epithelium',
                         copy=True)

    tri_graph = gt.Graph()
    ## the vertices
    _cell, _jv0, _jv1 = tri_graph.add_vertex(3)
    ## edges
    tri_graph.add_edge_list([(0, 1), (0, 2), (1, 2)])
    _cj0, _cj1, _je = (e for e in tri_graph.edges())
    triangles = gt.subgraph_isomorphism(tri_graph, eptm.graph)
    for tri in triangles:
        cell = eptm.graph.vertex(tri[_cell])
        jv0 = eptm.graph.vertex(tri[_jv0])
        jv1 = eptm.graph.vertex(tri[_jv1])
        assert eptm.is_cell_vert[cell]
        assert eptm.is_cell_vert[jv0] == 0
        assert eptm.is_cell_vert[jv1] == 0

        je = eptm.graph.edge(jv0, jv1)
        cj0 = eptm.graph.edge(cell, jv0)
        cj1 = eptm.graph.edge(cell, jv1)
        assert eptm.is_junction_edge[je]
        assert eptm.is_junction_edge[cj0] == 0
        assert eptm.is_junction_edge[cj1] == 0
