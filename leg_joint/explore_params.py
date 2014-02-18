# -*- coding: utf-8 -*-



import numpy as np
# Import mpltlib before graph-tool to
# avoid Gtk seg fault (don't ask why)
import matplotlib.pyplot as plt
import graph_tool.all as gt
import time
from datetime import datetime
import datetime
import os
import json




from .optimizers import find_energy_min
from .topology import cell_division
from .epithelium import Epithelium

import random


'''
Parameter management utilities
==============================
'''


def get_param(key, params, index):
    '''
    Returns the corresponding paramter value at position `index`
    if the valuie is a list or the value itself
    
    Parameters:
    ===========
    
    key: a dictionary key (usually a string)
    params: a dictionnary of dictonnaries
    index: int: the position of the value
    '''
    for sub_params in params.values():
        if not key in sub_params:
            continue
        try:
            return sub_params[key][index]
        except TypeError:
            return sub_params[key]
    
def get_grid_indices(params):

    grid_params = {}
    for sub_params in params.values():
        grid_params.update({key: val
                            for key, val in sub_params.items()
                            if type(val) is list
                            or type(val) is np.ndarray})
    if not len(grid_params):
        return None
    grid_indices = np.meshgrid(*(np.arange(len(values))
                                 for values in grid_params.values()),
                               indexing='ij') ## Avoids the two first axes to be swaped
    grid_indices = {key: indices.ravel() 
                    for key, indices
                    in zip(grid_params.keys(), grid_indices)}
    return grid_indices

def get_kwargs(index, params):

    grid_indices = get_grid_indices(params)
    if grid_indices is None:
        return params
    def get_sub_dict(sub_params):
        target = {}
        for key in sub_params.keys():
            try:
                print(key, grid_indices[key][index])
                param_index = grid_indices[key][index]
            except KeyError:
                param_index = 0
            target[key] = get_param(key, params,
                                    index=param_index)
        return target

    return {sub_key: get_sub_dict(sub_params)
            for sub_key, sub_params in params.items()}

def dump_json(all_kwds, index, save_dir):
    
    now = datetime.datetime.isoformat(
        datetime.datetime.utcnow())
    time_tag = '_'.join(now.split(':'))
    identifier = '%05i_%s' % (index, time_tag)
    json_name = os.path.join(save_dir,
                             'params_%s.json'
                              % identifier)
    with open(json_name, 'w+') as json_file:
        json.dump(all_kwds, json_file, sort_keys=True)
        print('Wrote %s' % os.path.abspath(json_name))
    return identifier





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
    
    eptm2 = Epithelium(graph=eptm.graph.copy(), **param_dict)
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
    pos0, pos1 = find_energy_min(eptm2, tol=1e-5)
    eptm2.isotropic_relax()
    j = cell_division(eptm2, mother_cell,
                      verbose=False)
    if j is not None:
        pos0, pos1 = find_energy_min(eptm2, tol=1e-5)
        eptm2.isotropic_relax()
        delta_h = (eptm2.rhos[mother_cell] - eptm2.rho_lumen) / height
    else:
        delta_h = np.nan

    return delta_h