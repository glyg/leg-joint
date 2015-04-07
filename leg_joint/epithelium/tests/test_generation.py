# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from leg_joint.epithelium.generation import cylindrical
from leg_joint.epithelium.epithelium import Epithelium


vertex_data = {
    ## Coordinates
    'x': (0., np.float),
    'y': (0., np.float),
    'z': (0., np.float),
    'rho': (0., np.float),
    'theta': (0., np.float),
    'height': (0., np.float),
    ## Geometry
    'perimeter': (0., np.float),
    'area': (0., np.float),
    'vol': (0., np.float),
    ## Topology
    'is_cell_vert': (0, np.bool),
    'is_alive': (1, np.bool),
    'is_active': (1, np.bool),
    'num_sides': (1, np.uint16),
    ## Dynamical
    'contractility': (0.014, np.float),
    'vol_elasticity': (0.014, np.float)}
edge_data = {
    ## Coordinates
    'dx': (0., np.float),
    'dy': (0., np.float),
    'dz': (0., np.float),
    'edge_length': (0., np.float),
    ## Gradients
    'gx': (0., np.float),
    'gy': (0., np.float),
    'gz': (0., np.float),
    ## Topological
    'is_junction_edge': (0, np.bool),
    ## Dynamical parameters
    'line_tension': (0., np.float),
    'radial_tension': (0., np.float)}
face_data = {
    ## Normal
    'ux': (0., np.float),
    'uy': (0., np.float),
    'uz': (0., np.float),
    ## Geometry
    'sub_area': (0., np.float),
    'ell_ij': (0., np.float),
    'height': (0., np.float)}



def test_cylindrical():

    n_circum, n_length = 7, 8
    graph, vertex_df, edge_df = cylindrical(
        n_circum, n_length, 1., 1.)

    degrees = np.bincount(graph.degree_property_map('out').a)

    assert degrees[4] == 2 * n_circum
    assert degrees[6] == (n_length - 2) * n_circum



def test_new_eptm():

    eptm = Epithelium()
    n_zeds = eptm.params['n_zeds']
    n_sigmas = eptm.params['n_sigmas']
    assert eptm.is_cell_vert.a.sum() == n_zeds * n_sigmas
