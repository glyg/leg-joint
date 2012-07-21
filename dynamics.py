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

        #Cell graph initialisation
        cells = Cells(self, subgraph=None)
        
        # Adding the cells to the (stil empty) epithelium graph
        subgraph = cells.subgraph.copy()


        rhos = subgraph.vertex_properties['rhos'].a
        thetas = subgraph.vertex_properties['thetas'].a
        zeds = subgraph.vertex_properties['zeds'].a
        AbstractRTZGraph.__init__(self, rhos,
                                  thetas,
                                  zeds,
                                  subgraph)

        self.junctions = AppicalJunctions(self, cells=cells, subgraph=None)        # Note that from next line on we only work on filtered views
        # and the AbstractRTZGraph methods are only valid applied directly
        # to `eptm.graph`. i.e. `eptm.cells.rhos` is not guaranteed to
        # be up to date.
        

    @property
    def is_cell_vert(self):
        return self.subgraph.vertex_properties["is_cell_vert"]
    @property
    def is_junction_edge(self):
        return self.subgraph.edge_properties["is_junction_edge"]
 
    @property
    def is_ctoj_edge(self):
        return self.subgraph.edge_properties["is_ctoj_edge"]

    @property
    def cells_subgraph(self):
        return gt.GraphView(self.subgraph,
                            vfilt=self.is_cell_vert)
    @property
    def junctions_subgraph(self):
        return gt.GraphView(self.subgraph,
                            efilt=self.is_junction_edge)
    def init_cells(self):
        return Cells(self, subgraph=self.cells_subgraph)

    def init_junctions(self):
        return AppicalJunctions(self, subgraph=self.junctions_subgraph)

    @property
    def is_local(self):
        return self.subgraph.edge_properties["is_local"]
    
    def find_energy_min(self):
        
        #vfilt = 
        self.junctions_subgraph.set_vertex_filter(self.is_local)
        j_vertices = [j_vert for j_vert in
                      self.junctions_subgraph.vertices()]
        pos0 = self.sz_pos

        np.array([self.sz_pos[j_vert]
                  for j_vert in self.junctions_subgraph.vertices()])
        self.junctions_subgraph.set_vertex_filter(None)
        output = optimize.fmin_ncg(self.local_energy, pos0.flatten(),
                                   fprime=self.local_gradient,
                                   avextol=0.1,
                                   retall=True,
                                   args=(j_vertices,))
        return pos0, output

    def set_new_pos(self, new_sz_pos, vertices, filter):
        for n, j_vert in enumerate(j_vertices):
            self.junctions.sigmas[j_vert] = new_sz_pos[2 * n]
            rho = self.junctions.rhos[j_vert]
            self.junctions.thetas[j_vert] = (new_sz_pos[2 * n]
                                             / rho) % (2 * np.pi)
            self.junctions.zeds[j_vert] = new_sz_pos[2 * n + 1]
        self.junctions.rtz_group()

    def local_energy(self, new_sz_pos, j_vertices):
        self.set_new_pos(new_sz_pos, j_vertices)
        return self.calc_energy(local=True)

    def local_gradient(self, new_sz_pos, j_vertices):
        gradient = np.zeros(new_sz_pos.shape)
        self.set_new_pos(new_sz_pos, j_vertices)        
        self.calc_apical_geometry(local=True)
        for n, j_vertex in enumerate(j_vertices):
            sn, zn = new_sz_pos[2 * n],  new_sz_pos[2 * n + 1]
            for cell in self.junctions.cells_vertices[j_vertex]:
                s0, z0 = self.cells.sz_pos[cell]
                d0n = np.hypot(s0 - sn, z0 - zn)
                sg = (s0 - sn) / d0n
                zg = (z0 - zn) / d0n
                elastic_term = - self.cells.elasticities[cell] * (
                    self.cells.areas[cell] - self.cells.prefered_area[cell])
                contractile_term = - self.cells.contractilities[cell] * (
                    self.cells.perimeters[cell])
                gradient[2 * n] += (elastic_term + contractile_term) * sg
                gradient[2 * n + 1] += (elastic_term + contractile_term) * zg
            for j_edge in j_vertex.out_edges():
                s1, z1 = self.junctions.sz_pos[j_edge.target()]
                d1n = np.hypot(s1 - sn, z1 - zn)
                sg = (s1 - sn) / d1n
                zg = (z1 - zn) / d1n
                gradient[2*n] +=  -self.junctions.line_tensions[j_edge] * sg
                gradient[2*n + 1] +=  -self.junctions.line_tensions[j_edge] * zg
        return gradient
    
    def calc_cell_positions(self):
        self.junctions.rtz_group()
        for cell in self.cells.subgraph.vertices():
            all_pos = np.array([self.junctions.sz_pos[jv].a
                                for jv in
                                self.cells.junctions_vertices[cell]])
            self.cells.sigmas[cell] = all_pos[:, 0].mean()
            self.cells.zeds[cell] = all_pos[:, 0].mean()
        self.cells.rtz_group()

    def calc_apical_geometry(self, local=False):
        """
        The area is approximated as the sum
        of the areas of the triangles
        formed by the cell position and each junction
        """
        if local:
            self.cells.subgraph.set_vertex_filter(self.cells.is_local)
        for cell in self.cells.subgraph.vertices():
            if not self.cells.is_alive[cell]: continue
            area = 0.
            perimeter = 0.
            sz0 = self.cells.sz_pos[cell]
            t0 = self.cells.thetas[cell]
            for edge in self.cells.junctions_edges[cell]:
                rtz1 = self.junctions.rtz_pos[edge.source()]
                rtz2 = self.junctions.rtz_pos[edge.target()]
                t1 = self.junctions.periodic_theta(edge.source(), t0)
                t2 = self.junctions.periodic_theta(edge.target(), t0)
                sz1 = [rtz1[0] * t1, rtz1[2]]
                sz2 = [rtz2[0] * t2, rtz2[2]]

                d01, d02, d12, area012 = triangle_geometry(sz0, sz1, sz2)
                area += area012
                self.junctions.edge_lengths[edge] = d12
                perimeter += d12
                self.cells.areas[cell] = area
                self.cells.perimeters[cell] = perimeter
        if local:
            self.cells.subgraph.set_vertex_filter(None)
        
    def calc_energy(self, local=False):
        
        self.calc_apical_geometry(local)
        elastic_term = 0.5 * self.cells.elasticities.a * (
            self.cells.areas.a - self.cells.prefered_area.a)**2
        contractile_term = 0.5 * self.cells.contractilities.a * (
            self.cells.perimeters.a**2)     
        tension_term = self.junctions.line_tensions.a * (
            self.junctions.edge_lengths.a)
        return elastic_term.sum() + (
            contractile_term.sum() + tension_term.sum())

    def set_local_mask(self, cell):

        self.cells.is_local[cell] = 1
        for neighbour in cell.all_neighbours():
            self.cells.is_local[neighbour] = 1
        for j_vert in self.cells.junctions_vertices[cell]:
            self.junctions.is_local[j_vert] = 1

    def remove_local_mask(self, cell):

        self.cells.is_local[cell] = 0
        for neighbour in cell.all_neighbours():
            self.cells.is_local[neighbour] = 0
        for j_vert in self.cells.junctions_vertices[cell]:
            self.junctions.is_local[j_vert] = 0

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
            for je in self.cells.junctions_edges[cell1]:
                if je in self.cells.junctions_edges[cell3]:
                    j_edgeab = je
            j_verta = j_edgeab.source()
            j_vertb = j_edgeab.target()
        elif element_type == 'j_vertices':
            j_verta, j_vertb = elements
            j_edgeab = self.junctions_subgraph.edge(elements)
            cell1, cell3 = self.junctions.adjacent_cells[j_edgeab]
        elif element_type == 'j_edge':
            j_edgeab = elements
            j_verta, j_vertb = j_edgeab.source(), j_edgeab.target()
            cell1, cell3 = self.junctions.adjacent_cells[j_edgeab]
        try:
            vecinos_a = self.junctions.all_vecinos(j_verta)
            vecinos_b = self.junctions.all_vecinos(j_vertb)
            j_vertc, j_vertf = vecinos_a[vecinos_a != j_vertb]
            j_vertd, j_verte = vecinos_b[vecinos_b != j_verta]
        except ValueError:
            raise ValueError("Valid only for 3-way junctions")

        j_edgeac = self.junctions_subgraph.edge(j_verta, j_vertc)
        j_edgebe = self.junctions_subgraph.edge(j_vertb, j_verte)
        if j_edgebe is None or j_edgeac is None:
            raise ValueError

        if not cell1 in self.junctions.adjacent_cells[j_edgeac]:
            j_vertc, j_vertf = j_vertf, j_vertc
        if not cell3 in self.junctions.adjacent_cells[j_edgebe]:
            j_verte, j_vertd = j_verte, j_vertd
        cell2 = self.junctions.adjacent_cells[j_edgebe][1]
        if cell2 == cell3:
            cell2 = self.junctions.adjacent_cells[j_edgebe][0]
        cell4 = self.junctions.adjacent_cells[j_edgeac][1]
        if cell4 == cell1:
            cell4 = self.junctions.adjacent_cells[j_edgeac][0]

        self.remove_junction(cell1, cell3, j_verta, j_vertb)
        self.remove_junction(cell1, cell4, j_verta, j_vertc)
        self.remove_junction(cell2, cell3, j_vertb, j_verte)
        self.add_junction(cell2, cell4, j_verta, j_vertb)
        self.add_junction(cell2, cell3, j_verta, j_verte)
        self.add_junction(cell1, cell4, j_vertb, j_vertc)

        # pi/2 rotation of junction vertices a and b around their center
        self.junctions.rtz_group()
        center = (self.junctions.sz_pos[j_verta].a +
                  self.junctions.sz_pos[j_vertb].a)/2.
        j_verta_sz = self.junctions.sz_pos[j_verta].a - center
        j_vertb_sz = self.junctions.sz_pos[j_vertb].a - center

        self.junctions.sigmas[j_verta] = j_verta_sz[1] + center[0]
        self.junctions.zeds[j_verta] = - j_verta_sz[0]  + center[1]
        self.junctions.thetas[j_verta] = (
            self.junctions.sigmas[j_verta]
            / self.junctions.rhos[j_verta]
            ) % (2 * np.pi)
        self.junctions.sigmas[j_vertb] = j_vertb_sz[1] + center[0]
        self.junctions.zeds[j_vertb] = - j_vertb_sz[0]  + center[1]
        self.junctions.thetas[j_vertb] = (
            self.junctions.sigmas[j_vertb]
            / self.junctions.rhos[j_vertb]
            ) % (2 * np.pi)
        self.junctions.rtz_group()
        self.calc_apical_geometry(local=True)

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
        
    #     daughter_cell = self.cells.subgraph.add_vertex()
    #     for j_edge in  self.cells.junction_edges[mother_cell]:
    #         if (j_edge.source() in right_verts
    #             ) and (j_edge.target() in right_verts):

    #             self.cells.junction_edges[mother_cell]
                
        
    def add_junction(self, cell0, cell1, j_verta, j_vertb):

        ##### This block should go in a decorator
        valid = np.array([obj.is_valid() for obj in
                          (cell0, cell1, j_verta, j_vertb)])
        if not valid.all():
            raise ValueError("invalid elements in the argument list"
                             "with cell0 => %s"
                             "     cell1 => %s"
                             "     vertex a => %s "
                             "     vertex b => %s " % (str(v) for v in valid)
                             )
        ####
        ce01 = self.cells.subgraph.edge(cell0, cell1)
        if ce01 is not None:
            print "Warning: previous cell0 to cell1 edge is re-created."
            self.cells.subgraph.remove_edge(ce01)
        ce01 = self.cells.subgraph.add_edge(cell0, cell1)

        j_edgeab = self.junctions_subgraph.edge(j_verta, j_vertb)
        if j_edgeab is not None:
            print "Warning: previous j_verta to j_vertb edge is re-created."
            self.junctions_subgraph.remove_edge(j_verta, j_vertb)
        j_edgeab = self.junctions_subgraph.add_edge(j_verta, j_vertb)

        self.cells.junctions_edges[cell0].append(j_edgeab)
        self.cells.junctions_vertices[cell0].extend((j_verta, j_vertb))
        self.cells.junctions_edges[cell1].append(j_edgeab)
        self.cells.junctions_vertices[cell1].extend((j_verta, j_vertb))
        
        self.junctions.adjacent_cells[j_edgeab] = (cell0, cell1)
        self.junctions.cells_vertices[j_verta].extend((cell0, cell1))
        self.junctions.cells_vertices[j_vertb].extend((cell0, cell1))
        line_tension0 = self.params['line_tension']
        self.junctions.line_tensions[j_edgeab] = line_tension0

    def remove_junction(self, cell0, cell1, j_verta, j_vertb):
        #This block should go in a decorator
        valid = np.array([element.is_valid() for element in
                          (cell0, cell1, j_verta, j_vertb)])
        if not valid.all():
            raise ValueError("invalid elements in the argument list"
                             "with cell0 => %s"
                             "     cell1 => %s"
                             "     vertex a => %s "
                             "     vertex b => %s " % (str(v) for v in valid))
        ####
        j_edgeab = self.junctions_subgraph.edge(j_verta, j_vertb)
        if j_edgeab is None:
            print "Warning: junction %s doesn't exist" % str(j_edgeab)
            return
        try:
            self.cells.junctions_edges[cell0].remove(j_edgeab)
            self.cells.junctions_edges[cell1].remove(j_edgeab)
            self.junctions.cells_vertices[j_verta].remove(cell0)
            self.junctions.cells_vertices[j_verta].remove(cell1)
            self.junctions.cells_vertices[j_vertb].remove(cell0)
            self.junctions.cells_vertices[j_vertb].remove(cell1)
            self.cells.junctions_vertices[cell0].remove(j_verta)
            self.cells.junctions_vertices[cell1].remove(j_verta)
            self.cells.junctions_vertices[cell0].remove(j_vertb)
            self.cells.junctions_vertices[cell1].remove(j_vertb)
        except ValueError:
            print("redundant call for cells %s, %s "
                  "and vertices %s, %s " 
                  % (cell0, cell1, j_verta, j_vertb))
            pass
        ce01 = self.cells.subgraph.edge(cell0, cell1)
        self.cells.subgraph.remove_edge(ce01)
        self.junctions_subgraph.remove_edge(j_edgeab)


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

