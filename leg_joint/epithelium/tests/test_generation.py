# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from leg_joint.epithelium.generation import cylindrical
from leg_joint.epithelium.epithelium import Epithelium


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
