#!/usr/bin/env python
# -*- coding: utf-8 -*-
from graph_tool import Graph, GraphView
import numpy as np
from ..utils import deprecated

def local_slice(eptm, theta_c=0, theta_amp=np.pi/24,
                zed_c=0, zed_amp=None):
    ''' Mark as local a 'slice' of the epithelium, with
    zed = zed_c +/- zed_amp and theta = theta_c +/- theta_amp
    '''
    if theta_amp is None:
        theta_min = eptm.thetas.a.min()
        theta_max = eptm.thetas.a.max()
    else:
        theta_min = theta_c - theta_amp
        theta_max = theta_c + theta_amp
    if zed_amp is None:
        zed_min = eptm.zeds.a.min()
        zed_max = eptm.zeds.a.max()
    else:
        zed_min = zed_c - zed_amp
        zed_max = zed_c + zed_amp

    slice_cells = [cell for cell in eptm.cells
                   if (theta_min < eptm.thetas[cell] < theta_max)
                   and (zed_min < eptm.zeds[cell] < zed_max)]
    eptm.set_local_mask(None)
    for cell in slice_cells:
        eptm.set_local_mask(cell)

def focus_on_cell(eptm, cell, radius):

    theta_c = eptm.thetas[cell]
    theta_amp = radius / eptm.rhos[cell]
    zed_c = eptm.zeds[cell]
    zed_amp = radius
    local_slice(eptm, theta_c, theta_amp,
                zed_c, zed_amp)

#####  Decorators

def local(meth):
    def new_function(self, *args, **kwargs):
        prev_vstate, prev_inverted_v = self.graph.get_vertex_filter()
        prev_estate, prev_inverted_e = self.graph.get_edge_filter()
        if self.__verbose__ : print('filter local')
        self.set_vertex_state([(self.is_local_vert, False),])
        self.set_edge_state([(self.is_local_edge, False)])
        out = meth(self, *args, **kwargs)
        if self.__verbose__ : print('restore from local filter')
        self.graph.set_vertex_filter(prev_vstate, prev_inverted_v)
        self.graph.set_edge_filter(prev_estate, prev_inverted_e)
        return out
    return new_function


def active(meth):
    def new_function(self, *args, **kwargs):
        prev_vstate, prev_inverted_v = self.graph.get_vertex_filter()
        prev_estate, prev_inverted_e = self.graph.get_edge_filter()
        if self.__verbose__ : print('filter active')
        self.set_vertex_state([(self.is_active_vert, False),])
        self.set_edge_state([(self.is_active_edge, False)])
        out = meth(self, *args, **kwargs)
        if self.__verbose__ : print('restore from active filter')
        self.graph.set_vertex_filter(prev_vstate, prev_inverted_v)
        self.graph.set_edge_filter(prev_estate, prev_inverted_e)
        return out
    return new_function


def no_filter(meth):
    def new_function(self, *args, **kwargs):
        if self.__verbose__ : print('no filter')
        prev_vstate, prev_inverted_v = self.graph.get_vertex_filter()
        prev_estate, prev_inverted_e = self.graph.get_edge_filter()
        self.graph.set_vertex_filter(None)
        self.graph.set_edge_filter(None)
        out = meth(self, *args, **kwargs)
        if self.__verbose__ : print('restore from no filter')
        self.graph.set_vertex_filter(prev_vstate, prev_inverted_v)
        self.graph.set_edge_filter(prev_estate, prev_inverted_e)
        return out
    return new_function

def cells_in(meth):
    def new_function(self, *args, **kwargs):
        prev_vstate, prev_inverted = self.graph.get_vertex_filter()
        if self.__verbose__ : print('filter cells in')
        #self.graph.set_vertex_filter(self.is_cell_vert)
        self.set_vertex_state([(self.is_cell_vert, False),
                                (self.is_alive, False)])
        out = meth(self, *args, **kwargs)
        if self.__verbose__ : print('restore from cells in')
        self.graph.set_vertex_filter(prev_vstate, prev_inverted)
        return out
    return new_function

def cells_out(meth):
    def new_function(self, *args, **kwargs):
        if self.__verbose__ : print('cells out')
        prev_vstate, prev_inverted = self.graph.get_vertex_filter()
        self.set_vertex_state([(self.is_cell_vert, True),
                               (self.is_alive, False)])
        out = meth(self, *args, **kwargs)
        if self.__verbose__ : print('restore cells from cells out')
        self.graph.set_vertex_filter(prev_vstate, prev_inverted)
        return out
    return new_function

def j_edges_in(meth):
    def new_function(self, *args, **kwargs):
        prev_estate, prev_inverted = self.graph.get_edge_filter()
        if self.__verbose__ : print('junction edges in')
        self.set_edge_state([(self.is_junction_edge, False),])
        out = meth(self, *args, **kwargs)
        self.graph.set_edge_filter(prev_estate, prev_inverted)
        if self.__verbose__ : print('restore from junctions in')
        return out
    return new_function

def ctoj_in(meth):
    def new_function(self, *args, **kwargs):
        prev_estate, prev_inverted = self.graph.get_edge_filter()
        if self.__verbose__ : print('cell to junctions edges in')
        self.set_edge_state([(self.is_ctoj_edge, False),])
        out = meth(self, *args, **kwargs)
        if self.__verbose__ : print('restore from cell to junctions edges in')
        self.graph.set_edge_filter(prev_estate, prev_inverted)
        return out
    return new_function

def deads_in(meth):
    def new_function(self, *args, **kwargs):
        if self.__verbose__ : print('dead vertices in')
        prev_vstate, prev_inverted = self.graph.get_vertex_filter()
        self.set_vertex_state([(self.is_alive, True),
                               (self.is_alive, False)])
        out = meth(self, *args, **kwargs)
        if self.__verbose__ : print('restore from deads in')
        self.graph.set_vertex_filter(prev_vstate, prev_inverted)
        return out
    return new_function
