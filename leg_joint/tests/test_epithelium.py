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
