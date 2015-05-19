# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from leg_joint.epithelium.generation import base_grid, cylindrical

from leg_joint.epithelium.epithelium import Epithelium



def test_base_grid():

    pos, is_cell_vert = base_grid(3, 3, delta_x=1, delta_y=1)
    assert pos.shape == (27, 2)
    assert is_cell_vert.shape == (27,)

    np.testing.assert_array_almost_equal(
        pos[:3, :],
        np.array([[ 0. ,  0.5],
                  [ 0. ,  1.5],
                  [ 0. ,  2.5]]))

    np.testing.assert_array_almost_equal(
        is_cell_vert[:3],
        np.array([1, 0, 0]))

    np.testing.assert_array_equal(
        is_cell_vert[:3],
        np.array([ True, False, False], dtype=bool))

def test_cylindrical():

    n_circum, n_length = 7, 9
    graph = cylindrical(n_length, n_circum, 1., 1.)
    graph.set_edge_filter(graph.ep['is_junction_edge'], inverted=True)
    degrees = np.bincount(graph.degree_property_map('out').fa)
    graph.clear_filters()
    assert degrees[0] == graph.num_vertices() - degrees[4] - degrees[6]
    assert degrees[4] == 2 * n_circum
    assert degrees[6] == (n_length - 2) * n_circum
