#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, time
import numpy as np
from scipy import optimize
import graph_tool.all as gt
from xml_handler import ParamTree
from scipy import weave

from objects import  AbstractRTZGraph, Cells, AppicalJunctions

CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(CURRENT_DIR)
PARAMFILE = os.path.join(ROOT_DIR, 'default', 'params.xml')
ELEMENT_TYPES=('cells', 'j_vertices', 'j_edge')
    
class Epithelium(AbstractRTZGraph):
    
    def __init__(self, paramtree=None, paramfile=PARAMFILE):
        # Parametrisation
        if paramtree == None:
            self.paramtree = ParamTree(paramfile)
        else:
            self.paramtree = paramtree
        self.params = self.paramtree.absolute_dic
        # Graph instanciation
        self.graph = gt.Graph(directed=True)
        # All the properties are packed here
        AbstractRTZGraph.__init__(self)

        # Cells and junctions graphviews initialisation
        self.cells = Cells(self)

        self.update_deltas()
        self.update_edge_lengths()
        self.junctions = AppicalJunctions(self)

        # self.relax_rhos()
        self.update_thetas()
        self.update_deltas()
        self.update_edge_lengths()
        self.cells.update_energy_grad()
                



    @property
    def is_cell_vert(self):
        return self.graph.vertex_properties["is_cell_vert"]
    @property
    def is_junction_edge(self):
        return self.graph.edge_properties["is_junction_edge"]
    @property
    def is_ctoj_edge(self):
        return self.graph.edge_properties["is_ctoj_edge"]
    @property
    def ctojgraph(self):
        return gt.GraphView(self.graph,
                            efilt=self.is_ctoj_edge)
    @property
    def is_local_cell(self):
        return self.graph.vertex_properties["is_local_cell"]
    @property
    def is_local_junction(self):
        return self.graph.vertex_properties["is_local_junction"]
    @property
    def is_local_both(self):
        return self.graph.vertex_properties["is_local_cell"]

    def find_energy_min(self):
        self.graph.set_vertex_filter(self.is_local_junction)
        j_vertices = [j_vert for j_vert in self.graph.vertices()]
        pos0 = np.array([self.sz_pos[j_vert] for j_vert
                         in self.graph.vertices()])
        self.junctions.graph.set_vertex_filter(None)
        output = optimize.fmin_ncg(self.local_energy, pos0.flatten(),
                                   fprime=self.local_gradient,
                                   avextol=0.1,
                                   retall=True,
                                   args=(j_vertices,))
        return pos0, output

    def set_new_pos(self, new_sz_pos, vertices):
        new_sz_pos = new_sz_pos.flatten()
        for n, vert in enumerate(vertices):
            self.sigmas[vert] = new_sz_pos[2 * n]
            self.zeds[vert] = new_sz_pos[2 * n + 1]

    def local_energy(self, new_sz_pos, j_vertices):
        self.set_new_pos(new_sz_pos, j_vertices)
        return self.calc_energy(local=True)

    def local_gradient(self, new_sz_pos, j_vertices):
        gradient = np.zeros(new_sz_pos.shape)
        self.set_new_pos(new_sz_pos, j_vertices)
        self.update_apical_geometry(local=True)
        j_vertices = [self.graph.vertex(jv)
                      for jv in j_vertices]
        self.graph.set_vertex_filter(None)
        for n, j_vertex in enumerate(j_vertices):
            for edge in j_vertex.in_edges():
                #unitary vectors
                u_sg = - self.u_dsigmas[edge]
                u_zg = - self.u_dzeds[edge]
                cell = edge.source()
                cell_grad = self.cells.energy_grad[
                    cell] * self.is_cell_vert[cell]
                gradient[2 * n] += cell_grad * u_sg
                gradient[2 * n + 1] += cell_grad * u_zg
                junc_grad = self.junctions.line_tensions[
                    edge] * self.is_junction_edge[edge]
                gradient[2 * n] += junc_grad * u_sg
                gradient[2 * n + 1] += junc_grad * u_zg

            for j_edge in j_vertex.out_edges():
                junc_grad = self.junctions.line_tensions[
                    j_edge] * self.is_junction_edge[j_edge]
                u_sg = - self.u_dsigmas[j_edge]
                u_zg = - self.u_dzeds[j_edge]
                gradient[2 * n] += junc_grad * u_sg
                gradient[2 * n + 1] += junc_grad * u_zg
        return gradient
    
    def calc_cell_positions(self):
        for cell in self.cells.graph.vertices():
            cell_link = self.ctojgraph.vertex[cell]
            all_jpos = np.array([(self.sigmas[jv], self.zeds[jv])
                                for jv in cell_link.out_neighbours()])
            self.sigmas[cell] = all_jpos[:, 0].mean()
            self.zeds[cell] = all_jpos[:, 1].mean()

    def update_apical_geometry(self, local=False):
        """
        The area is approximated as the sum
        of the areas of the triangles
        formed by the cell position and each junction
        """
        self.update_deltas()
        self.update_edge_lengths()
        if local:
            self.graph.set_vertex_filter(self.is_local_cell)
        else:
            self.graph.set_vertex_filter(self.is_cell_vert)
        cells = [c for c in self.graph.vertices()]
        self.graph.set_vertex_filter(None)
        for cell in cells:
            area = 0.
            perimeter = 0.
            for j_edge in self.cell_junctions(cell):
                perimeter += self.edge_lengths[j_edge]
                ctoj0 = self.graph.edge(cell, j_edge.source())
                ctoj1 = self.graph.edge(cell, j_edge.target())
                area += np.abs(self.dsigmas[ctoj0] * self.dzeds[ctoj1]
                               -  self.dsigmas[ctoj1] * self.dzeds[ctoj0])/2.
            self.cells.areas[cell] = area
            self.cells.perimeters[cell] = perimeter
        self.cells.graph.set_vertex_filter(None)

    def adjacent_cells(self, j_edge):
        jv0 = j_edge.source()
        jv1 = j_edge.target()
        cells_a = [cell for cell in self.c2jgraph.vertex(jv0).in_neighbours()]
        cells_b = [cell for cell in self.c2jgraph.vertex(jv1).in_neighbours()]
        common_cells = [cell for cell in cells_a if cell in cells_b]
        return common_cells

    def cell_junctions(self, cell):
        self.graph.set_directed(False)
        j_verts = np.array([
            jv for jv in self.ordered_neighbours(cell)
            if self.is_cell_vert[jv] == 0])
        
        e0 = self.graph.edge(j_verts[-1], j_verts[0])
        if e0 is None:
            e0 = self.graph.edge(j_verts[0], j_verts[-1])
        j_edges = [e0] if e0 is not None else []
        for jv0, jv1 in zip(j_verts[:-1], j_verts[1:]):
            e = self.graph.edge(jv0, jv1)
            if e is None:
                e = self.graph.edge(jv1,jv0)
            if e is not None: j_edges.append(e)
        self.graph.set_directed(True)
        return j_edges
        
    def calc_energy(self, local=False):

        self.update_apical_geometry(local)
        if local:
            self.graph.set_vertex_filter(self.is_local_cell)
        else:
            self.graph.set_vertex_filter(self.is_cell_vert)
                        
        elastic_term = 0.5 * self.cells.elasticities.fa * (
            self.cells.areas.fa - self.cells.prefered_area.fa)**2
        contractile_term = 0.5 * self.cells.contractilities.fa * \
                           self.cells.perimeters.fa**2
        if local:
            self.graph.set_edge_filter(self.is_local_junction)
        else:
            self.graph.set_edge_filter(self.is_junction_edge)
        tension_term = self.junctions.line_tensions.fa * \
                       self.junctions.edge_lengths.fa
        print self.junctions.edge_lengths.fa.size
        self.graph.set_vertex_filter(None)
        return elastic_term.sum() + (
            contractile_term.sum() + tension_term.sum())

    def set_local_mask(self, cell):
        cell = self.graph.vertex(cell)
        self.is_local_cell[cell] = 1
        for neighbour in cell.all_neighbours():
            self.is_local_both[neighbour] = 1
            if not self.is_cell_vert[neighbour]:
                self.is_local_junction[neighbour] = 1
            else:
                self.is_local_cell[neighbour] = 1
        for edge in self.cell_junctions(cell):
            self.is_local_j_edge

            
    def remove_local_mask(self, cell):
        self.is_local_cell[cell] = 0
        for neighbour in cell.all_neighbours():
            self.is_local_cell[neighbour] = 0
            self.is_local_junction[neighbour] = 0
            self.is_local_both[neighbour] = 0
            
    def type1_transition(self, elements, element_type='cells'):
        """Type one transition (see the definition in
        Farhadifar et al. Curr Biol. 2007 Dec 18;17(24):2095-104.
        Suppplementary figure S1)
        
        In ASCII art (letters represent junctions and number represent cells):

        e 2 d                  
         \ /         e  d        e  2  d  
          b           \/          \   /
        3 | 1  ---->  ab  ----> 3  a-b  1  
          a           /\          /   \    
         / \         f  c        f  4  c 
        f 4 c                     
        """
        if not element_type in ELEMENT_TYPES:
            raise AttributeError( "element_type should be in:"
                                  "{'cells', 'j_vertices', 'j_edge'} ")

        if element_type == 'cells':
            cell1 = elements[0]
            cell3 = elements[1]
            j_edges1 =  self.cell_junctions(cell1)
            j_edges3 =  self.cell_junctions(cell3)
            try:
                j_edgeab = [je for je in j_edges1 if je in j_edges3][0]
            except IndexError:
                print ("No valid junction found"
                       "beetween %s and %s cells" % (cell1, cell3))
                return
            j_verta = j_edgeab.source()
            j_vertb = j_edgeab.target()
        elif element_type == 'j_vertices':
            j_verta, j_vertb = elements
            j_edgeab = self.graph.edge(elements)
            if j_edgeab is None:
                print "Invalid junction %s" % str(j_edgeab)
                return
            try:
                cell1, cell3 = self.adjacent_cells(j_edgeab)
            except ValueError:
                print ("No adgacent cells found"
                       "for junction %s" % str(j_edgeab)
                       )
                return
        elif element_type == 'j_edge':
            j_edgeab = elements
            j_verta, j_vertb = j_edgeab.source(), j_edgeab.target()

            try:
                cell1, cell3 = self.junctions.adjacent_cells(j_edgeab)
            except ValueError:
                print ("No adgacent cells found"
                       "for junction %s" % str(j_edgeab))
                return
        try:
            vecinos_a = [jv for jv in self.ordered_neighbours(j_verta)
                         if not self.is_cell_vert[jv]]
            vecinos_b = [jv for jv in self.ordered_neighbours(j_vertb)
                         if not self.is_cell_vert[jv]]
            j_vertc, j_vertf = vecinos_a[vecinos_a != j_vertb]
            j_vertd, j_verte = vecinos_b[vecinos_b != j_verta]
        except ValueError:
            print "Valid only for 3-way junctions"
            return
        
        j_edgeac = self.junctions_graph.edge(j_verta, j_vertc)
        j_edgebe = self.junctions_graph.edge(j_vertb, j_verte)
        if j_edgebe is None or j_edgeac is None:
            print "Invalid geometry"
            return
        if not cell1 in self.junctions.adjacent_cells(j_edgeac):
            j_vertc, j_vertf = j_vertf, j_vertc
        if not cell3 in self.junctions.adjacent_cells(j_edgebe):
            j_verte, j_vertd = j_verte, j_vertd
        cell2 = self.junctions.adjacent_cells(j_edgebe)[1]
        if cell2 == cell3:
            cell2 = self.junctions.adjacent_cells(j_edgebe)[0]
        cell4 = self.junctions.adjacent_cells(j_edgeac)[1]
        if cell4 == cell1:
            cell4 = self.junctions.adjacent_cells(j_edgeac)[0]

        self.remove_junction(j_verta, j_vertb, cell1, cell3)
        self.remove_junction(j_verta, j_vertc, cell1, cell4)
        self.remove_junction( j_vertb, j_verte, cell2, cell3)
        self.add_junction(j_verta, j_vertb, cell2, cell4)
        self.add_junction(j_verta, j_verte, cell2, cell3)
        self.add_junction( j_vertb, j_vertc, cell1, cell4)

        # pi/2 rotation of junction vertices a and b around their center
        center = (self.junctions.sz_pos[j_verta].a +
                  self.junctions.sz_pos[j_vertb].a)/2.
        j_verta_sz = self.junctions.sz_pos[j_verta].a - center
        j_vertb_sz = self.junctions.sz_pos[j_vertb].a - center

        self.sigmas[j_verta] = j_verta_sz[1] + center[0]
        self.zeds[j_verta] = - j_verta_sz[0]  + center[1]
        self.sigmas[j_vertb] = j_vertb_sz[1] + center[0]
        self.zeds[j_vertb] = - j_vertb_sz[0]  + center[1]
        self.upadte_apical_geometry(local=True)

        modified_cells = [cell1, cell2, cell3, cell4]
        modified_jverts = [j_verta, j_vertb, j_vertc,
                           j_vertd, j_verte, j_vertf]
        return modified_cells, modified_jverts

    # def cell_division(self, mother_cell):

    #     j_verts = self.cells.junctions_vertices[mother_cell]

    #     zeta_division = np.random.random_sample() * 2 * np.pi
    #     sigmas = [self.junctions.sigmas[j] - self.cells.sigmas[mother_cell]
    #               for j in j_verts]
    #     zeds = [self.junctions.zeds[j] - self.cells.zeds[mother_cell] 
    #             for j in j_verts]
    #     zetas = np.arctan2(sigmas, zeds) + np.pi
    #     zetas_rel = (zetas - zeta_division) % (2 * np.pi)
    #     right_of = [zetas_rel <= np.pi]
    #     right_verts = [jv for jv, right in zip(j_verts, right_of)
    #                    if right]
    #     left_verts =  [jv for jv, right in zip(j_verts, right_of)
    #                    if not right]
        
    #     daughter_cell = self.cells.graph.add_vertex()
    #     for j_edge in  self.cells.junction_edges[mother_cell]:
    #         if (j_edge.source() in right_verts
    #             ) and (j_edge.target() in right_verts):

    #             self.cells.junction_edges[mother_cell]
                
        
    def add_junction(self, j_verta, j_vertb, cell0, cell1):

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
            print "Warning: previous j_verta to j_vertb edge is re-created."
            self.graph.remove_edge(j_verta, j_vertb)
        j_edgeab = self.graph.add_edge(j_verta, j_vertb)
        self.is_junction_edge[j_edgeab] = 1
        self.is_ctoj_edge[j_edgeab] = 0
        line_tension0 = self.params['line_tension']
        self.junctions.line_tensions[j_edgeab] = line_tension0

        ctoj_0a = self.graph.edge(cell0, j_verta)
        if ctoj_0a is not None:
            print "Warning: previous cell0 to j_verta edge is re-created."
            self.graph.remove(ctoj_0a)
        ctoj_0a = self.graph.add_edge(cell0, j_verta)
        self.is_junction_edge[ctoj_0a] = 0
        self.is_ctoj_edge[ctoj_0a] = 1

        ctoj_0b = self.graph.edge(cell0, j_vertb)
        if ctoj_0b is not None:
            print "Warning: previous cell0 to j_vertb edge is re-created."
            self.graph.remove(ctoj_0b)
        ctoj_0b = self.graph.add_edge(cell0, j_vertb)
        self.is_junction_edge[ctoj_0b] = 0
        self.is_ctoj_edge[ctoj_0b] = 1

        if cell1 is not None:
            ce01 = self.graph.edge(cell0, cell1)
            if ce01 is not None:
                print "Warning: previous cell0 to cell1 edge is re-created."
                self.graph.remove_edge(ce01)
            ce01 = self.graph.add_edge(cell0, cell1)
            self.is_junction_edge[ce01] = 0
            self.is_ctoj_edge[ce01] = 0

            ctoj_1b = self.graph.edge(cell0, j_verta)
            if ctoj_1b is not None:
                print "Warning: previous cell0 to j_verta edge is re-created."
                self.graph.remove(ctoj_1b)
            ctoj_1b = self.graph.add_edge(cell0, j_verta)
            self.is_junction_edge[ctoj_1b] = 0
            self.is_ctoj_edge[ctoj_1b] = 1

            ctoj_1b = self.graph.edge(cell0, j_vertb)
            if ctoj_1b is not None:
                print "Warning: previous cell0 to j_vertb edge is re-created."
                self.graph.remove(ctoj_1b)
            ctoj_1b = self.graph.add_edge(cell0, j_vertb)
            self.is_junction_edge[ctoj_1b] = 0
            self.is_ctoj_edge[ctoj_1b] = 1

    def remove_junction(self, j_verta, j_vertb, cell0, cell1):
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
            print "Warning: junction %s doesn't exist" % str(j_edgeab)
            return
        self.graph.remove_edge(j_edgeab)
        ctoj_0a = self.c2jgraph.edge(cell0, j_verta)
        self.graph.remove_edge(ctoj_0a)
        ctoj_0b = self.c2jgraph.edge(cell0, j_vertb)
        self.graph.remove_edge(ctoj_1b)
        if cell1 is not None:
            ctoj_1a = self.c2jgraph.edge(cell1, j_verta)
            self.graph.remove_edge(ctoj_1a)
            ctoj_1b = self.c2jgraph.edge(cell1, j_vertb)
            self.graph.remove_edge(ctoj_1b)

            ce01 = self.cells.graph.edge(cell0, cell1)
            self.graph.remove_edge(ce01)


def triangle_geometry(sz0, sz1, sz2):
    c_code = """
    double s0 = sz0[0];
    double z0 = sz0[1];
    double s1 = sz1[0];
    double z1 = sz1[1];
    double s2 = sz2[0];
    double z2 = sz2[1];


    double d01 = sqrt((s0-s1) * (s0-s1) + (z1-z0) * (z1-z0));
    double d02 = sqrt((s0-s2) * (s0-s2) + (z2-z0) * (z2-z0));
    double d12 = sqrt((s1-s2) * (s1-s2) + (z2-z1) * (z2-z1));
    double area012 = fabs((s1-s0) * (z2-z0) - (s2-s0) * (z1-z0));

    py::tuple results(4);
    results[0] = d01;
    results[1] = d02;
    results[2] = d12;
    results[3] = area012;
    return_val = results;
    
    """
    return weave.inline(c_code,
                        arg_names=['sz0', 'sz1', 'sz2'],
                        headers=['<math.h>'])

