# -*- coding: utf-8 -*-



import numpy as np
# Import mpltlib before graph-tool to
# avoid Gtk seg fault (don't ask why)
import matplotlib.pyplot as plt
import graph_tool.all as gt
import time
from datetime import datetime

import leg_joint as lj
import random


# eptm = lj.Epithelium(graphXMLfile='../saved_graphs/xml/initial_graph.xml',
#                      paramfile='../default/params.xml')

def explore_delta_h_2D(eptm, param1, values1, param2, values2):

    delta_hs = values1.copy()
    for n, (value1, value2) in enumerate(zip(values1, values2)):
        param_dict = {param1: value1, param2: value2}
        delta_hs[n] = one_division_height(eptm, **param_dict)
    return delta_hs


def explore_delta_h_1D(eptm, param, values):

    delta_hs = values.copy()
    for n, value in enumerate(values):
        param_dict = {param: value}
        delta_hs[n] = one_division_height(eptm, **param_dict)
    return delta_hs

def one_division_height(eptm, **param_dict):
    
    eptm2 = lj.Epithelium(graph=eptm.graph.copy(), **param_dict)
    try:
        eptm2.isotropic_relax()
    except ValueError:
        print('invalid parameter')
        return np.nan
    cell_num  = eptm2.params['n_zeds'] // 2
    mother_cell = eptm2.graph.vertex(cell_num)
    height = eptm2.rhos[mother_cell] - eptm2.rho_lumen
    eptm2.set_local_mask(None)
    eptm2.set_local_mask(mother_cell)
    eptm2.cells.prefered_vol[mother_cell] *= 1.8
    pos0, pos1 = lj.find_energy_min(eptm2, tol=1e-5)
    eptm2.isotropic_relax()
    j = lj.cell_division(eptm2, mother_cell,
                         verbose=False)
    if j is not None:
        pos0, pos1 = lj.find_energy_min(eptm2, tol=1e-5)
        eptm2.isotropic_relax()
        delta_h = (eptm2.rhos[mother_cell] - eptm2.rho_lumen) / height
    else:
        delta_h = np.nan

    return delta_h