# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd
import graph_tool.all as gt


def generate_cells(n_circum, n_length, l_0, h_0):

    rho_c = n_circum * l_0 / (2 * np.pi)
    rho_lumen = rho_c - h_0

    delta_theta = 2 * np.pi / n_circum
    delta_z = delta_theta * rho_c * np.sqrt(3)/2.

    zt_grid = np.mgrid[:n_length, :n_circum]
    thetas = zt_grid[1].astype('float')
    thetas[::2, ...] += 0.5
    thetas *= delta_theta
    thetas = thetas.flatten()
    zeds = zt_grid[0].astype('float')
    zeds *= delta_z
    zeds -= zeds.max() / 2
    zeds = zeds.flatten()

    pos = np.empty((n_circum * n_length, 3))

    pos[:, 0] = np.cos(thetas) * rho_c
    pos[:, 1] = np.sin(thetas) * rho_c
    pos[:, 2] = zeds
    cells_graph, pos_vp = gt.geometric_graph(pos, l_0*1.1)

    pos = pd.DataFrame(pos, columns=['x', 'y', 'z'])
    pos.index.name = 'cell'

    return pos, cells_graph