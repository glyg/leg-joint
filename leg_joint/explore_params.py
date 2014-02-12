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


def get_param(key, fixed_params, grid_params, index, mode='fixed'):
    '''
    Returns the corresponding paramter value in either `fixed_params` 
    or `grid_params` dictionaries. If it's from the grid parameters,
    the value  at position `index` is returned
    
    If mode is set to 'fixed', `fixed_params` will be
    searched first (the default), it it'set to grid, the opposite will happen.
    
    Parameters:
    ===========
    
    key: a dictionary key (usually a string)
    fixed_params: a dictionnary containing a single value per key
    grid_params: a dictionnary containing sequences of values per key
    mode: {'fixed' | 'grid'} default 'fixed', dictionnary to search first
    
    '''
    if mode == 'fixed':
        if fixed_params.get(key) is None:
            values = grid_params.get(key)
            if values is not None:
                return values[index]
        return fixed_params.get(key)
    elif mode == 'grid':
        values = grid_params.get(key)
        if values is None:
            return fixed_params.get(key)
        else:
            return values[index]
    
def get_grid_indices(grid_params):
    if not len(grid_params):
        return None
    grid_indices = np.meshgrid(*(np.arange(len(values))
                                 for values in grid_params.values()),
                               indexing='ij') ## Avoids the two first axes to be swaped
    grid_indices = {key: indices.ravel() 
                    for key, indices
                    in zip(grid_params.keys(), grid_indices)}
    return grid_indices

def get_all_kwds(index, fixed_params, grid_params,
                 grid_indices, seq_keys,
                 apopto_keys, post_keys):

    if grid_indices is  None:
        seq_kwds = {key: fixed_params[key]
                    for key in seq_keys}
        apopto_kwds = {key: fixed_params[key]
                    for key in apopto_keys}
        post_kwds = {key: fixed_params[key]
                    for key in post_keys}
        return seq_kwds, apopto_kwds, post_kwds

    def get_keys(keys):
        target = {}
        for key in keys:
            try:
                param_index = grid_indices[key][index]
            except KeyError:
                param_index = 0
            target[key] = get_param(key,
                                    fixed_params,
                                    grid_params,
                                    index=param_index, mode='fixed')
        return target
    seq_kwds = get_keys(seq_keys)
    apopto_kwds = get_keys(apopto_keys)
    post_kwds = get_keys(post_keys)

    return seq_kwds, apopto_kwds, post_kwds

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