# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from leg_joint.epithelium import generation

def test_cell_generation():

    n_circum, n_length = 7, 8
    pos, cells_graph = generation.generate_cells(n_circum, n_length,
                                                 l_0=1, h_0=1)
    degrees = np.bincount(cells_graph.degree_property_map('out').a)

    assert degrees[4] == 2 * n_circum
    assert degrees[6] == (n_length - 2) * n_circum
