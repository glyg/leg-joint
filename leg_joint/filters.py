#!/usr/bin/env python -*- coding: utf-8 -*-
from graph_tool import Graph, GraphView

import numpy as np
import logging
log = logging.getLogger(__name__)



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
        log.debug('filter local')
        self.graph.set_vertex_filter(self.is_local_vert)
        self.graph.set_edge_filter(self.is_local_edge)
        out = meth(self, *args, **kwargs)
        return out
    return new_function


def active(meth):
    def new_function(self, *args, **kwargs):
        log.debug('filter active')
        self.graph.set_vertex_filter(self.is_active_vert)
        self.graph.set_edge_filter(self.is_active_edge)
        out = meth(self, *args, **kwargs)
        return out
    return new_function

def no_filter(meth):
    def new_function(self, *args, **kwargs):
        log.debug('no filter')
        self.graph.set_vertex_filter(None)
        self.graph.set_edge_filter(None)
        out = meth(self, *args, **kwargs)
        return out
    return new_function


def cells_in(meth):
    def new_function(self, *args, **kwargs):
        log.warning('This is deprecated')
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
        log.warning('This is deprecated')
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
        log.debug('cell to junctions edges in')
        self.set_edge_state([(self.is_ctoj_edge, False),])
        out = meth(self, *args, **kwargs)
        log.debug('restore from cell to junctions edges in')
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


class EpitheliumFilters(object):

    def __init__(self):
        if self.new:
            self._init_edge_filters()
            self._init_vertex_filters()
        else:
            self._get_edge_filters()
            self._get_vertex_filters()

    def _init_edge_filters(self):
        self.is_junction_edge = self.graph.new_edge_property('bool')
        self.is_junction_edge.a[:] = 0
        self.graph.edge_properties["is_junction_edge"
                                   ] = self.is_junction_edge

        self.is_ctoj_edge = self.graph.new_edge_property('bool')
        self.is_ctoj_edge.a[:] = 0
        self.graph.edge_properties["is_ctoj_edge"] = self.is_ctoj_edge
        self.is_local_edge = self.graph.new_edge_property('bool')
        self.is_local_edge.a[:] = 0
        self.graph.edge_properties["is_local_edge"] = self.is_local_edge
        self.is_active_edge = self.graph.new_edge_property('bool')
        self.is_active_edge.a[:] = 0
        self.graph.edge_properties["is_active_edge"] = self.is_active_edge

    def _get_edge_filters(self):
        self.is_junction_edge = self.graph.edge_properties["is_junction_edge"]
        self.is_ctoj_edge = self.graph.edge_properties["is_ctoj_edge"]
        self.is_local_edge = self.graph.edge_properties["is_local_edge"]
        self.is_active_edge = self.graph.edge_properties["is_active_edge"]

    def _init_vertex_filters(self):
        # Is a cell
        self.is_cell_vert = self.graph.new_vertex_property('bool')
        self.graph.vertex_properties["is_cell_vert"] = self.is_cell_vert
        self.is_alive = self.graph.new_vertex_property('bool')
        self.is_alive.a[:] = 1
        self.graph.vertex_properties["is_alive"] = self.is_alive
        # Locality
        self.is_local_vert = self.graph.new_vertex_property('bool')
        self.is_local_vert.a[:] = 0
        self.graph.vertex_properties["is_local_vert"] = self.is_local_vert
        # Active verts can move
        self.is_active_vert = self.graph.new_vertex_property('bool')
        self.is_active_vert.a[:] = 0
        self.graph.vertex_properties["is_active_vert"] = self.is_active_vert

    def _get_vertex_filters(self):
        # Is a cell
        self.is_cell_vert = self.graph.vertex_properties["is_cell_vert"]
        self.is_alive = self.graph.vertex_properties["is_alive"]
        # Locality
        self.is_local_vert = self.graph.vertex_properties["is_local_vert"]
        # Active verts can move
        self.is_active_vert = self.graph.vertex_properties["is_active_vert"]

    def new_ctoj_edge(self, cell, jv):
        if self.graph.edge(cell, jv) is None:
            cj = self.graph.add_edge(cell, jv)
            self.is_ctoj_edge[cj] = 1
            self.is_junction_edge[cj] = 0
            self.at_boundary[cj] = 0
            self.is_new_edge[cj] = 1
            return cj
        else:
            return self.graph.edge(cell, jv)

    def new_j_edge(self, jv0, jv1):
        if self.any_edge(jv0, jv1) is None:
            j_edge = self.graph.add_edge(jv0, jv1)
            self.is_ctoj_edge[j_edge] = 0
            self.is_junction_edge[j_edge] = 1
            self.at_boundary[j_edge] = 0
            self.is_new_edge[j_edge] = 1
            return j_edge
        else:
            return self.any_edge(jv0, jv1)

    def set_vertex_state(self,  properties=[]):
        if len(properties) == 0:
            self.graph.set_vertex_filter(None)
            return
        vstate = self.graph.get_vertex_filter()
        if vstate[0] is not None:
            cur_vfilt, inverted = vstate
            tmp_vfilt  = cur_vfilt.copy()
            if inverted: tmp_vfilt.a = 1 - cur_vfilt.a
        else:
            tmp_vfilt = self.graph.new_vertex_property('bool')
            tmp_vfilt.a[:] = 1
        for prop, inverted in properties:
            if inverted:
                tmp_vfilt.a *= (1 - prop.a)
            else:
                tmp_vfilt.a *= prop.a
        log.debug('%i vertices filtered in' % tmp_vfilt.a.sum())
        self.graph.set_vertex_filter(tmp_vfilt)

    def set_edge_state(self,  properties=[]):
        if len(properties) == 0:
            self.graph.set_edge_filter(None)
            return
        estate = self.graph.get_edge_filter()
        if estate[0] is not None:
            cur_efilt, inverted = estate
            tmp_efilt  = cur_efilt.copy()
        else:
            tmp_efilt = self.graph.new_edge_property('bool')
            tmp_efilt.a[:] = 1
        for prop, inverted in properties:
            tmp_efilt.a *= (1 - prop.a) if inverted else prop.a

        log.debug('%i edges filtered in'
                  % tmp_efilt.a.sum())
        self.graph.set_edge_filter(tmp_efilt)

    #### Local Masks
    @no_filter
    def set_local_mask(self, cell, wider=False):
        if cell is None:
            self.is_local_edge.a[:] = 0
            self.is_local_vert.a[:] = 0
            self.is_active_edge.a[:] = 0
            self.is_active_vert.a[:] = 0
            return
        if isinstance(cell, int):
            cell = self.graph.vertex(cell)
        self.is_local_vert[cell] = 1
        for je in self.cells.junctions[cell]:
            self.is_local_edge[je] = 1
        for j_vert in cell.out_neighbours():
            self.is_local_vert[j_vert] = 1
            self.is_active_vert[j_vert] = 1
            ctoj = self.graph.edge(cell, j_vert)
            self.is_local_edge[ctoj] = 1
            for neighb_cell in j_vert.all_neighbours():
                if self.is_cell_vert[neighb_cell] and neighb_cell != cell:
                    self.is_local_vert[neighb_cell] = 1
                    for jv in neighb_cell.out_neighbours():
                        self.is_local_vert[jv] = 1
                        self.is_local_edge[self.graph.edge(neighb_cell,
                                                           jv)] = 1
                    for je in self.cells.junctions[neighb_cell]:
                        self.is_local_edge[je] = 1

            for edge in j_vert.all_edges():
                self.is_active_edge[edge] = 1
        if wider:
            for n_cell in self.cells.get_neighbor_cells(cell):
                self.set_local_mask(n_cell, wider=False)


    @no_filter
    def remove_local_mask(self, cell):
        self.is_local_vert[cell] = 0
        for j_vert in cell.out_neighbours():
            self.is_local_vert[j_vert] = 0
            self.is_active_vert[j_vert] = 0
            for edge in j_vert.all_edges():
                self.is_local_edge[edge] = 0
                self.is_active_edge[edge] = 0
                if self.is_junction_edge[edge]:
                    cell0, cell1 = self.junctions.adjacent_cells[edge]
                    adj_cell = cell0 if cell1 == cell else cell1
                    self.is_local_vert[adj_cell] = 0
                    for jv in adj_cell.out_neighbours():
                        self.is_local_vert[jv] = 0
                        self.is_local_edge[self.graph.edge(adj_cell, jv)] = 0
                    for je in self.cell.junctions[adj_cell]:
                        self.is_local_edge[je] = 0
