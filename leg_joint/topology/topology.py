# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import numpy as np
import graph_tool.all as gt

from ..utils import deprecated

import logging
log = logging.getLogger(__name__)



class Topology:

    def any_edge(self, v0, v1):
        '''
        Returns the edge between vertices v0 and v1 if it exists,
        whether it goes from v0 to v1 or from v1 to v0 and None otherwize
        '''
        e = self.graph.edge(v0, v1)
        if e is None:
            e = self.graph.edge(v1, v0)
        return e

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

    def new_edge(self, vertex0, vertex1, source_edge):
        '''Adds an edge between vertex0 and vertex1 and copies the properties
        of source_edge to this new edge
        '''
        if self.any_edge(vertex0, vertex1) is None:
            new_edge = self.graph.add_edge(vertex0, vertex1)
            for prop in self.graph.edge_properties.values():
                prop[new_edge] = prop[source_edge]
            return new_edge
        else:
            return self.any_edge(vertex0, vertex1)

    def new_vertex(self, source_vertex):
        new_v = self.graph.add_vertex()
        for prop in self.graph.vertex_properties.values():
            prop[new_v] = prop[source_vertex]
        return new_v

    def reset_topology(self):
        raise NotImplementedError

    def add_junction(self, j_verta, j_vertb, cell0, cell1):
        '''Adds a junction to the epithelium, creating a junction edge
        between `j_verta` and `j_vertb`, cell to junction edges
        between `cell0` and `j_edgea`, `cell0` and `j_edgeb`, `cell1`
        and `j_edgea` and `cell1` and `j_edgeb`
        '''
        ##### TODO: This block should go in a decorator
        valid = np.array([obj.is_valid() for obj in
                          (cell0, j_verta, j_vertb)])
        if not valid.all():
            raise ValueError("invalid elements in the argument list"
                             "with cell0 => %s"
                             "     vertex a => %s "
                             "     vertex b => %s " % (str(v) for v in valid)
                             )
        ####
        j_edgeab = self.graph.edge(j_verta, j_vertb)
        if j_edgeab is not None:
            self.log.debug('Previous {} to {} '
                           'edge is re-created.'.format(str(j_verta),
                                                        str(j_vertb)))
            self.graph.remove_edge(j_edgeab)
        j_edgeab = self.graph.add_edge(j_verta, j_vertb)
        j_edge_old = self.cells.junctions[cell0][0]
        for e_prop in self.graph.edge_properties.values():
            e_prop[j_edgeab] = e_prop[j_edge_old]
        self.junctions.adjacent_cells[j_edgeab] = cell0, cell1

        self.cells.junctions[cell0].append(j_edgeab)

        if self.cells.junctions[cell1] is None:
            self.cells.junctions[cell1] = [j_edgeab,]
        else:
            self.cells.junctions[cell1].append(j_edgeab)

        prev_ctojs = [ctoj for ctoj in cell0.out_edges()]
        ctoj_old = prev_ctojs[0]

        ctoj_0a = self.graph.edge(cell0, j_verta)
        if ctoj_0a is not None:
            self.graph.remove_edge(ctoj_0a)
        ctoj_0a = self.graph.add_edge(cell0, j_verta)

        ctoj_0b = self.graph.edge(cell0, j_vertb)
        if ctoj_0b is not None:
            self.graph.remove_edge(ctoj_0b)
        ctoj_0b = self.graph.add_edge(cell0, j_vertb)

        for e_prop in self.graph.edge_properties.values():
            e_prop[ctoj_0a] = e_prop[ctoj_old]
            e_prop[ctoj_0b] = e_prop[ctoj_old]
        self._set_cell_pos(cell0)

        if cell1 is not None:
            ctoj_1a = self.graph.edge(cell1, j_verta)
            if ctoj_1a is not None:
                self.graph.remove_edge(ctoj_1a)
            ctoj_1a = self.graph.add_edge(cell1, j_verta)

            ctoj_1b = self.graph.edge(cell1, j_vertb)
            if ctoj_1b is not None:
                self.graph.remove_edge(ctoj_1b)
            ctoj_1b = self.graph.add_edge(cell1, j_vertb)
            for e_prop in self.graph.edge_properties.values():
                e_prop[ctoj_1a] = e_prop[ctoj_old]
                e_prop[ctoj_1b] = e_prop[ctoj_old]

            self._set_cell_pos(cell1)
        return j_verta, j_vertb, cell0, cell1

    def remove_junction(self, j_verta, j_vertb, cell0, cell1):
        '''Removes junction between `j_edgea` and `j_edgeb`, and the
        corresponding cell to junction edges for `cell0` and
        `cell1`
        '''

        #This block should go in a decorator
        valid = np.array([element.is_valid() for element in
                          (cell0, j_verta, j_vertb)])
        if not valid.all():
            raise ValueError("invalid elements in the argument list"
                             "with cell0 => %s"
                             "     vertex a => %s "
                             "     vertex b => %s " % (str(v) for v in valid))
        ####
        j_edgeab = self.graph.edge(j_verta, j_vertb)
        if j_edgeab is None:
            j_edgeab = self.graph.edge(j_vertb, j_verta)
        if j_edgeab is None:
            self.log.debug("Junction from {} to {} doesn't exist".format(
                           j_edgeab.source(), j_edgeab.target()))
            return

        self.graph.remove_edge(j_edgeab)
        self.cells.update_junctions(cell0)
        self.cells.update_junctions(cell1)
        #self.junctions.update_adjacent()
        ctoj_0a = self.graph.edge(cell0, j_verta)
        if ctoj_0a is not None:
            self.graph.remove_edge(ctoj_0a)
        ctoj_0b = self.graph.edge(cell0, j_vertb)
        if ctoj_0b is not None:
            self.graph.remove_edge(ctoj_0b)
        if cell1 is not None:
            ctoj_1a = self.graph.edge(cell1, j_verta)
            if ctoj_1a is not None:
                self.graph.remove_edge(ctoj_1a)
            ctoj_1b = self.graph.edge(cell1, j_vertb)
            if ctoj_1b is not None:
                self.graph.remove_edge(ctoj_1b)

    def merge_j_verts(self, jv0, jv1):
        '''Merge junction vertices `jv0` and `jv1`. Raises an error if
        those vertices are not connected
        '''
        vertex_trash = []
        edge_trash = []
        je = self.any_edge(jv0, jv1)
        if je is None:
            raise ValueError('Can only merge connected vertices')

        edge_trash.append(je)
        for vert in jv1.out_neighbours():
            old_edge = self.graph.edge(jv1, vert)
            if vert != jv0:
                new_edge = self.graph.add_edge(jv0, vert)
            for prop in self.graph.edge_properties.values():
                prop[new_edge] = prop[old_edge]
            edge_trash.append(old_edge)
        for vert in jv1.in_neighbours():
            old_edge = self.graph.edge(vert, jv1)
            if vert != jv0:
                new_edge = self.graph.add_edge(vert, jv0)
            for prop in self.graph.edge_properties.values():
                prop[new_edge] = prop[old_edge]
            edge_trash.append(old_edge)
        vertex_trash.append(jv1)
        return vertex_trash, edge_trash

    #### Local Masks
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

    @deprecated
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
        if self.__verbose__:
            print('%i vertices filtered in' % tmp_vfilt.a.sum())
        self.graph.set_vertex_filter(tmp_vfilt)

    @deprecated
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

        if self.__verbose__: print('%i edges filtered in'
                                   % tmp_efilt.a.sum())
        self.graph.set_edge_filter(tmp_efilt)

    def remove_cell(self, cell):

        if isinstance(cell, int):
            cell = self.graph.vertex(cell)

        self.set_local_mask(None)
        self.graph.set_vertex_filter(None)
        self.graph.set_edge_filter(None)
        vertex_trash = []
        new_ctojs = []
        new_jes =[]
        ctojs = [ctoj for ctoj in cell.out_edges()]
        cell_jes = self.cells.junctions[cell]
        jvs = [jv for jv in cell.out_neighbours()]
        if not len(jvs):
            self.log.error('No neighbours for cell %s' %cell)
            self.is_alive[cell] = 0
            self.is_cell_vert[cell] = 0
            return
        edge_trash = ctojs
        edge_trash.extend(cell_jes)
        new_jv = self.new_vertex(jvs[0])
        self.log.info('new vertex %s' % str(new_jv))
        self.is_local_vert[new_jv] = 1
        self.x[new_jv] = self.x[cell]
        self.y[new_jv] = self.y[cell]
        self.z[new_jv] = self.z[cell]
        adjacent_cells = []
        for jv in jvs:
            vertex_trash.append(jv)
            for edge in jv.all_edges():
                if edge in edge_trash:
                    continue
                edge_trash.append(edge)
                if self.is_ctoj_edge[edge]:
                    adj_cell = edge.source()
                    adjacent_cells.append(adj_cell)
                    new_ctojs.append((adj_cell, new_jv))
                elif self.is_junction_edge[edge]:
                    jv0, jv1 = edge
                    opposite = jv0 if jv1 == jv else jv1
                    new_jes.append((opposite, new_jv))

        for neighb_cell, jv in new_ctojs:
            ctoj = self.new_edge(neighb_cell, jv, ctojs[0])
            self.set_local_mask(neighb_cell)
        for jv0, jv1 in new_jes:
            je = self.new_edge(jv0, jv1, cell_jes[0])
            self.is_local_vert[jv0] = 1
            self.is_local_vert[jv1] = 1
            self.is_local_edge[je] = 1

        self.is_alive[cell] = 0
        self.is_cell_vert[cell] = 0
        for v in vertex_trash:
            self.is_alive[v] = 0
            self.is_cell_vert[v] = 0
        for e in edge_trash:
            try:
                self.graph.remove_edge(e)
            except ValueError:
                self.log.error('edge already destroyed')
        self.reset_topology()
        self.update_geometry()
        self.set_local_mask(None)
        return new_jv
