#!/usr/bin/env python
# -*- coding: utf-8 -*-

#####  Decorators
def filter_local(meth):
    def new_function(self, *args, **kwargs):
        prev_vstate, prev_inverted_v = self.graph.get_vertex_filter()
        prev_estate, prev_inverted_e = self.graph.get_edge_filter()
        if __debug__ : print 'filter local'
        self.set_vertex_state([(self.is_local_vert, False),])
        self.set_edge_state([(self.is_local_edge, False)])
        out = meth(self, *args, **kwargs)
        if __debug__ : print 'retore from local filter'
        self.graph.set_vertex_filter(prev_vstate, prev_inverted_v)
        self.graph.set_edge_filter(prev_estate, prev_inverted_e)
        return out
    return new_function

def filter_active(meth):
    def new_function(self, *args, **kwargs):
        prev_vstate, prev_inverted_v = self.graph.get_vertex_filter()
        prev_estate, prev_inverted_e = self.graph.get_edge_filter()
        if __debug__ : print 'filter active'
        self.set_vertex_state([(self.is_active_vert, False),])
        self.set_edge_state([(self.is_active_edge, False)])
        out = meth(self, *args, **kwargs)
        if __debug__ : print 'retore from active filter'
        self.graph.set_vertex_filter(prev_vstate, prev_inverted_v)
        self.graph.set_edge_filter(prev_estate, prev_inverted_e)
        return out
    return new_function


def no_filter(meth):
    def new_function(self, *args, **kwargs):
        if __debug__ : print 'no filter'
        prev_vstate, prev_inverted_v = self.graph.get_vertex_filter()
        prev_estate, prev_inverted_e = self.graph.get_edge_filter()
        self.graph.set_vertex_filter(None)
        self.graph.set_edge_filter(None)
        out = meth(self, *args, **kwargs)
        if __debug__ : print 'restore from no filter'
        self.graph.set_vertex_filter(prev_vstate, prev_inverted_v)
        self.graph.set_edge_filter(prev_estate, prev_inverted_e)
        return out
    return new_function

def filter_cells_in(meth):
    def new_function(self, *args, **kwargs):
        prev_vstate, prev_inverted = self.graph.get_vertex_filter()
        if __debug__ : print 'filter cells in'
        self.set_vertex_state([(self.is_cell_vert, False),
                               (self.is_alive, False)])
        out = meth(self, *args, **kwargs)
        if __debug__ : print 'restore from cells in'
        self.graph.set_vertex_filter(prev_vstate, prev_inverted)
        return out
    return new_function

def filter_cells_out(meth):
    def new_function(self, *args, **kwargs):
        if __debug__ : print 'cells out'
        prev_vstate, prev_inverted = self.graph.get_vertex_filter()
        self.set_vertex_state([(self.is_cell_vert, True),
                               (self.is_alive, False)])
        out = meth(self, *args, **kwargs)
        if __debug__ : print 'restore cells from cells out'
        self.graph.set_vertex_filter(prev_vstate, prev_inverted)
        return out
    return new_function

def filter_j_edges_in(meth):
    def new_function(self, *args, **kwargs):
        prev_estate, prev_inverted = self.graph.get_edge_filter()
        if __debug__ : print 'junction edges in'
        self.set_edge_state([(self.is_junction_edge, False),])
        out = meth(self, *args, **kwargs)
        self.graph.set_edge_filter(prev_estate, prev_inverted)
        if __debug__ : print 'restore from junctions in'
        return out
    return new_function

def filter_ctoj_in(meth):
    def new_function(self, *args, **kwargs):
        prev_estate, prev_inverted = self.graph.get_edge_filter()
        if __debug__ : print 'cell to junctions edges in'
        self.set_edge_state([(self.is_ctoj_edge, False),])
        out = meth(self, *args, **kwargs)
        if __debug__ : print 'restore from cell to junctions edges in'
        self.graph.set_edge_filter(prev_estate, prev_inverted)
        return out
    return new_function

def filter_deads_in(meth):
    def new_function(self, *args, **kwargs):
        if __debug__ : print 'cells out'
        prev_vstate, prev_inverted = self.graph.get_vertex_filter()
        self.set_vertex_state([(self.is_alive, True)])
        
        out = meth(self, *args, **kwargs)
        if __debug__ : print 'restore cells from cells out'
        self.graph.set_vertex_filter(prev_vstate, prev_inverted)
        return out
    return new_function

class EpitheliumFilters(object):

    def __init__(self):
        if self.new:
            self._init_edge_filters()
            self._init_vertex_filters()

    def _init_edge_filters(self):
        is_junction_edge = self.graph.new_edge_property('bool')
        self.graph.edge_properties["is_junction_edge"
                                   ] = is_junction_edge
        is_ctoj_edge = self.graph.new_edge_property('bool')
        self.graph.edge_properties["is_ctoj_edge"] = is_ctoj_edge
        is_local_edge = self.graph.new_edge_property('bool')
        is_local_edge.a[:] = 0
        self.graph.edge_properties["is_local_edge"] = is_local_edge
        is_active_edge = self.graph.new_edge_property('bool')
        self.graph.edge_properties["is_active_edge"] = is_active_edge

    @property
    def is_local_edge(self):
        '''boolean edge property'''
        return self.graph.edge_properties["is_local_edge"]
    @property
    def is_active_edge(self):
        return self.graph.edge_properties["is_active_edge"]
    @property
    def is_junction_edge(self):
        '''boolean edge property '''
        return self.graph.edge_properties["is_junction_edge"]
    @property
    def is_ctoj_edge(self):
        '''boolean edge property '''
        return self.graph.edge_properties["is_ctoj_edge"]

    def _init_vertex_filters(self):
        # Is a cell
        is_cell_vert = self.graph.new_vertex_property('bool')
        self.graph.vertex_properties["is_cell_vert"] = is_cell_vert
        is_alive = self.graph.new_vertex_property('bool')
        is_alive.a[:] = 1
        self.graph.vertex_properties["is_alive"] = is_alive
        # Locality
        is_local_vert = self.graph.new_vertex_property('bool')
        is_local_vert.a[:] = 0
        self.graph.vertex_properties["is_local_vert"] = is_local_vert
        # Active verts can change
        is_active_vert = self.graph.new_vertex_property('bool')
        is_active_vert.a[:] = 0
        self.graph.vertex_properties["is_active_vert"] = is_active_vert

    @property
    def is_alive(self):
        return self.graph.vertex_properties["is_alive"]
    @property
    def is_local_vert(self):
        return self.graph.vertex_properties["is_local_vert"]
    @property
    def is_local_vert(self):
        return self.graph.vertex_properties["is_local_vert"]
    @property
    def is_active_vert(self):
        return self.graph.vertex_properties["is_active_vert"]
    @property
    def is_cell_vert(self):
        return self.graph.vertex_properties["is_cell_vert"]

    def new_ctoj_edge(self, cell, jv):
        cj = self.graph.add_edge(cell, jv)
        self.is_ctoj_edge[cj] = 1
        self.is_junction_edge[cj] = 0
        self.at_boundary[cj] = 0
        self.is_new_edge[cj] = 1
        return cj

    def new_j_edge(self, jv0, jv1):
        
        j_edge_out = self.graph.edge(jv0, jv1)
        j_edge_in = self.graph.edge(jv1, jv0)
        if self.any_edge(jv0, jv1) is None:
            j_edge = self.graph.add_edge(jv0, jv1)
            self.is_ctoj_edge[j_edge] = 0
            self.is_junction_edge[j_edge] = 1
            self.at_boundary[j_edge] = 0
            self.is_new_edge[j_edge] = 1
            return j_edge
        else:
            return None

            
    def set_vertex_state(self,  properties=[]):
        if len(properties) == 0:
            self.graph.set_vertex_filter(None)
            return (None, False)
        vstate = self.graph.get_vertex_filter()
        if vstate[0] is not None:
            cur_vfilt, inverted = vstate
            tmp_vfilt  = cur_vfilt.copy()
            if inverted: tmp_vfilt.a = 1 - cur_vfilt.a
        else:
            tmp_vfilt = self.graph.new_vertex_property('bool')
            tmp_vfilt.a[:] = 1
        
        for prop, inverted in properties:
            if prop.key_type() == 'v' : 
                if inverted:
                    tmp_vfilt.a *= (1 - prop.a)
                else: 
                    tmp_vfilt.a *= prop.a
        if __debug__: print '%i vertices filtered in' % tmp_vfilt.a.sum()        
        self.graph.set_vertex_filter(tmp_vfilt)

    def set_edge_state(self,  properties=[]):
        if len(properties) == 0:
            self.graph.set_edge_filter(None)
            return (None, False)
        estate = self.graph.get_edge_filter()
        if estate[0] is not None:
            cur_efilt, inverted = estate
            tmp_efilt  = cur_efilt.copy()
        else:
            tmp_efilt = self.graph.new_edge_property('bool')
            tmp_efilt.a[:] = 1
        for prop, inverted in properties:
            if prop.key_type() == 'e' : 
                if inverted:
                    tmp_efilt.a *= (1 - prop.a)
                else: 
                    tmp_efilt.a *= prop.a

        if __debug__: print '%i edges filtered in' % tmp_efilt.a.sum()        
        self.graph.set_edge_filter(tmp_efilt)

    #### Local Masks
    @filter_deads_in
    def set_local_mask(self, cell):
        if cell is None:
            self.is_local_edge.a[:] = 0
            self.is_local_vert.a[:] = 0
            return
        cell = self.graph.vertex(cell)
        self.is_local_vert[cell] = 1
        for neighbour in cell.out_neighbours():
            self.is_local_vert[neighbour] = 1
            self.is_active_vert[neighbour] = 1
            for edge in neighbour.all_edges():
                self.is_local_edge[edge] = 1
                self.is_active_edge[edge] = 1
                cell0, cell1 = self.adjacent_cells(edge)
                adj_cell = cell0 if cell1 == cell else cell1
                for jv in adj_cell.out_neighbours():
                    self.is_local_vert[jv] = 1
                    self.is_local_edge[self.graph.edge(adj_cell, jv)] = 1
                for je in self.cell_junctions(adj_cell):
                    self.is_local_edge[je] = 1

    def remove_local_mask(self, cell):
        self.is_local_vert[cell] = 0
        for neighbour in cell.all_neighbours():
            self.is_local_vert[neighbour] = 0
            for edge in neighbour.all_edges():
                self.is_local_edge[edge] = 0
                for cell in self.adjacent_cells(edge):
                    self.is_local_vert[cell] = 0            
