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
from scipy import optimize


from numpy.testing import assert_almost_equal, assert_array_almost_equal

import tempfile
tmp_dir = tempfile.mkdtemp(prefix='tmp_lj_')

def _get_cell(eptm, idx=3):
    return eptm.graph.vertex(idx)

def _get_jv(eptm, idx=100):
    return eptm.graph.vertex(idx)

def _get_je(eptm, tup=(100, 101)):
    return eptm.graph.edge(*tup)

def test_relax():

    eptm = lj.Epithelium(graphXMLfile=lj.data.small_xml(),
                         save_dir=tmp_dir,
                         identifier='test_epithelium',
                         copy=True)
    eptm.isotropic_relax()
    vfilt = eptm.is_cell_vert.copy()
    vfilt.a *= eptm.is_alive.a
    eptm.graph.set_vertex_filter(vfilt)

    area = eptm.cells.areas.fa.mean()
    height = eptm.rhos.fa.mean() - eptm.rho_lumen
    eptm.delta_o = eptm.find_grad_roots()
    ratio0 = np.sqrt(area / eptm.params["prefered_area"])
    ratio1 = height / eptm.params['prefered_height']
    assert_almost_equal(eptm.delta_o, ratio0)
    assert_almost_equal(eptm.delta_o, ratio1)

def test_gradient():
    eptm = lj.Epithelium(graphXMLfile=lj.data.small_xml(),
                         save_dir=tmp_dir,
                         identifier='test_epithelium',
                         copy=True)
    eptm.isotropic_relax()
    mother_cell = _get_cell(eptm)
    growth_rate = 1.5
    eptm.cells.prefered_vol[mother_cell] *= growth_rate
    eptm.set_local_mask(None)
    eptm.set_local_mask(mother_cell)
    grad_err = lj.optimizers.check_local_grad(eptm)
    print('gradient_error: {}'.format(grad_err))
    assert abs(grad_err) < 1 #TODO gradient is badly dimensioned


def test_fmin():
    eptm = lj.Epithelium(graphXMLfile=lj.data.small_xml(),
                         save_dir=tmp_dir,
                         identifier='test_epithelium',
                         copy=True)
    eptm.isotropic_relax()
    mother_cell = _get_cell(eptm)
    area0 = eptm.cells.areas[mother_cell]
    growth_rate = 1.5
    eptm.cells.prefered_vol[mother_cell] *= growth_rate
    eptm.set_local_mask(None)
    eptm.set_local_mask(mother_cell, wider=True)
    pos0, pos1 = lj.find_energy_min(eptm)
    print(eptm.cells.areas[mother_cell] / area0)
    assert_almost_equal(eptm.cells.areas[mother_cell] / area0,
                       1.5664, decimal=3)
