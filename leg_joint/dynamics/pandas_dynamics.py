# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import pandas as pd
import numpy as np

from ..utils import _to_3d

def opt_energy(pos, trgles, norm_factor):

    _pos = pos.reshape((pos.size//3, 3))
    trgles.vertex_df.loc[trgles.uix_active, trgles.coords] = _pos
    trgles.geometry()
    energy = compute_energy(trgles)
    return energy/norm_factor

def opt_gradient(pos, trgles, norm_factor, rho_lumen):
    # _pos = pos.reshape((pos.size//3, 3))
    # trgles.vertex_df.loc[trgles.uix_active, trgles.coords] = _pos
    grad = trgles.gradient()
    return grad.values.flatten()/norm_factor

def compute_energy(trgles, full_output=False):

    junction_data = trgles.udf_itoj[['line_tensions', 'edge_lengths']]
    E_t = junction_data['line_tensions'] * junction_data['edge_lengths']

    cell_data = trgles.udf_cell[['vol_elasticities', 'vols', 'prefered_vol',
                               'contractilities', 'perimeters']]
    E_v =  0.5 * (cell_data['vol_elasticities']
                  * (cell_data['vols']
                     - cell_data['prefered_vol'])**2)
    E_c = 0.5 * (cell_data['contractilities']
                 * cell_data['perimeters']**2)
    if full_output:
        return E_t, E_c, E_v
    else:
        return E_t.sum()+(E_c+E_v).sum()

def compute_gradient(trgles, components=False):
    '''
    If components is True, returns the individual terms
    (grad_t, grad_c, grad_v)
    '''
    trgles.grad_i_lij = - trgles.udf_itoj[trgles.dcoords] / _to_3d(trgles.udf_itoj['edge_lengths'])
    trgles.grad_i_lij.index = pd.MultiIndex.from_tuples(trgles.uix_ij, names=('jv_i', 'jv_j'))

    grad_t = trgles.tension_grad()
    grad_c = trgles.contractile_grad()
    grad_v = trgles.volume_grad()

    grad_i = grad_t + grad_c + grad_v
    if components:
        return grad_i, grad_t, grad_c, grad_v
    return grad_i

def tension_grad(trgles):

    grad_t = trgles.grad_i.copy()
    grad_t[:] = 0

    tensions = trgles.udf_itoj['line_tensions']
    tensions.index.names = ('jv_i', 'jv_j')

    _grad_t = trgles.grad_i_lij * _to_3d(tensions)
    grad_t.loc[trgles.uix_active_i] = _grad_t.sum(level='jv_i').loc[trgles.uix_active_i].values
    grad_t.loc[trgles.uix_active_j] -= _grad_t.sum(level='jv_j').loc[trgles.uix_active_j].values
    return grad_t

def contractile_grad(trgles):

    grad_c = trgles.grad_i.copy()
    grad_c[:] = 0

    contract = trgles.udf_cell['contractilities']
    contract.index.name = 'cell'
    perimeters = trgles.udf_cell['perimeters']

    gamma_L = contract * perimeters
    gamma_L = gamma_L.loc[trgles.tix_a]
    gamma_L.index = trgles.tix_aij

    # area_term = gamma_L.groupby(level='jv_i').apply(
    #     lambda df: df.sum(level='jv_j'))
    area_term = gamma_L.groupby(level=('jv_i', 'jv_j')).sum()

    _grad_c = trgles.grad_i_lij.loc[trgles.uix_ij] * _to_3d(area_term.loc[trgles.uix_ij])
    grad_c.loc[trgles.uix_active_i] = _grad_c.sum(level='jv_i').loc[trgles.uix_active_i].values
    grad_c.loc[trgles.uix_active_j] -= _grad_c.sum(level='jv_j').loc[trgles.uix_active_j].values
    return grad_c

def volume_grad(trgles):
    '''
    Computes :math:`\sum_\alpha\nabla_i \left(K (V_\alpha - V_0)^2\right)`
    '''
    grad_v = trgles.grad_i.copy()
    grad_v[:] = 0

    elasticity = trgles.udf_cell['vol_elasticities']
    pref_V = trgles.udf_cell['prefered_vol']
    V = trgles.udf_cell['vols']
    KV_V0 = elasticity * (V - pref_V)
    tri_KV_V0 = KV_V0.loc[trgles.tix_a]
    tri_KV_V0.index = trgles.tix_aij

    r_ijs = trgles.tdf_itoj[trgles.dcoords]
    cross_ur = pd.DataFrame(np.cross(trgles.faces[trgles.normal_coords], r_ijs),
                            index=trgles.tix_aij, columns=trgles.coords)

    h_nu = trgles.udf_cell['heights'] / (2 * trgles.udf_cell['num_sides'])

    grad_i_V_cell = cross_ur.sum(level='cell') * _to_3d(KV_V0 * h_nu)

    cell_term_i = grad_i_V_cell.loc[trgles.tix_a].set_index(trgles.tix_ai)
    cell_term_j = grad_i_V_cell.loc[trgles.tix_a].set_index(trgles.tix_aj)

    grad_v.loc[trgles.uix_active_i] += cell_term_i.loc[trgles.uix_ai].sum(
        level='jv_i').loc[trgles.uix_active_i].values/2
    grad_v.loc[trgles.uix_active_j] += cell_term_j.loc[trgles.uix_aj].sum(
        level='jv_j').loc[trgles.uix_active_j].values/2

    _r_to_rho_i = trgles.udf_jv_i[trgles.coords] / _to_3d(trgles.udf_jv_i['rhos'])
    _r_to_rho_j = trgles.udf_jv_j[trgles.coords] / _to_3d(trgles.udf_jv_j['rhos'])
    r_to_rho_i = _r_to_rho_i.loc[trgles.tix_i].set_index(trgles.tix_aij)
    r_to_rho_j = _r_to_rho_j.loc[trgles.tix_j].set_index(trgles.tix_aij)
    r_ai = trgles.tdf_atoi[trgles.dcoords]
    r_aj = trgles.tdf_atoj[trgles.dcoords]
    normals = trgles.faces[trgles.normal_coords]
    cross_ai = pd.DataFrame(np.cross(normals, r_ai),
                            index=trgles.tix_aij, columns=trgles.coords)
    cross_aj = pd.DataFrame(np.cross(normals, r_aj),
                            index=trgles.tix_aij, columns=trgles.coords)

    tri_heights = trgles.tdf_cell['heights']
    tri_heights.index = trgles.tix_aij
    sub_areas = trgles.faces['sub_areas']

    _ij_term = _to_3d(tri_KV_V0) *(_to_3d(sub_areas / 2) * r_to_rho_i
                                   - _to_3d(tri_heights / 2) * cross_aj)
    _jk_term = _to_3d(tri_KV_V0) *(_to_3d(sub_areas / 2) * r_to_rho_j
                                   + _to_3d(tri_heights / 2) * cross_ai)

    #ij_term = _ij_term.groupby(level=('jv_i', 'jv_j')).sum()
    #jk_term = _jk_term.groupby(level=('jv_j', 'jv_i')).sum()

    grad_v.loc[trgles.uix_active_i] += _ij_term.sum(level='jv_i').loc[trgles.uix_active_i].values
    grad_v.loc[trgles.uix_active_j] += _jk_term.sum(level='jv_j').loc[trgles.uix_active_j].values

    return grad_v
