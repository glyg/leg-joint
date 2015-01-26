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

def _get_cell(eptm, idx=3):
    return eptm.graph.vertex(idx)

def _get_jv(eptm, idx=100):
    return eptm.graph.vertex(idx)

def _get_je(eptm, tup=(100, 101)):
    return eptm.graph.edge(*tup)

def grad_norm(eptm, jv):
    return (eptm.grad_ix[jv]**2 + eptm.grad_zed[jv]**2 + eptm.grad_wy[jv]**2)**0.5

def test_gradient():
    eptm = lj.Epithelium(graphXMLfile=lj.data.small_xml(),
                         save_dir=tmp_dir,
                         identifier='test_epithelium',
                         copy=True)
    eptm.isotropic_relax()
    mother_cell = _get_cell(eptm)
    jv0, jv1 = eptm.cells.junctions[mother_cell][0]

    area0 = eptm.cells.areas[mother_cell]
    growth_rate = 1.5
    eptm.cells.prefered_vol[mother_cell] *= growth_rate
    eptm.update_gradient()
    assert_almost_equal(grad_norm(eptm, jv0)/4845., 1, decimal=3)
    assert_almost_equal(grad_norm(eptm, jv1)/5080., 1, decimal=3)
