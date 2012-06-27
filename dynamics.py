#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, time
import numpy as np
from scipy import optimize
import graph_tool.all as gt
from xml_handler import ParamTree
from scipy import weave

from objects import CellGraph, AppicalJunctions

CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(CURRENT_DIR)
PARAMFILE = os.path.join(ROOT_DIR, 'default', 'params.xml')
ELEMENT_TYPES=('cells', 'j_vertices', 'j_edge')


def find_energy_min(epithelium):

    cells = epithelium.cells
    junctions = epithelium.junctions
    junctions.graph.set_vertex_filter(junctions.is_local)
    j_vertices = [j_vert for j_vert in junctions.graph.vertices()]
    pos0 = np.array([junctions.rtz_pos[j_vert]
                     for j_vert in junctions.graph.vertices()]).flatten()
    junctions.graph.set_vertex_filter(None)
    cells.graph.set_vertex_filter(cells.is_local)
    output = optimize.anneal(epithelium.local_energy, pos0,
                                args=(j_vertices,))
    return pos0, output

    
class Epithelium():
    
    def __init__(self, paramtree=None, paramfile=PARAMFILE):
        if paramtree == None:
            self.paramtree = ParamTree(paramfile)
        else:
            self.paramtree = paramtree
        self.params = self.paramtree.absolute_dic
        self.cells = CellGraph(self)
        self.junctions = AppicalJunctions(self)

    def local_energy(self, new_rtz_pos, j_vertices):
        
        for n, j_vert in enumerate(j_vertices):
            self.junctions.rhos[j_vert] = new_rtz_pos[3 * n]
            self.junctions.thetas[j_vert] = new_rtz_pos[3 * n + 1]
            self.junctions.zeds[j_vert] = new_rtz_pos[3 * n + 2]
            
        return self.calc_total_energy()

    def calc_apical_geometry(self):
        """
        The area is approximated as the sum
        of the areas of the triangles
        formed by the cell position and each junction
        """
        self.cells.rtz_group()
        self.junctions.rtz_group()

        for cell in self.cells.graph.vertices():
            area = 0.
            perimeter = 0.
            all_pos = np.array([self.junctions.rtz_pos[jv].a
                                for jv in
                                self.cells.junctions_vertices[cell]])

            xc = all_pos[:,0] * np.cos(all_pos[:,1])
            xc = xc.mean()
            yc = all_pos[:,0] * np.sin(all_pos[:,1])
            yc = yc.mean()
            zc = all_pos[:,2].mean()

            rhoc = np.sqrt(xc**2 + yc**2)
            thetac = np.arctan2(yc, xc)

            self.cells.rhos[cell] = rhoc
            self.cells.thetas[cell] = thetac
            self.cells.zeds[cell] = zc            
            self.cells.rtz_pos[cell].a = [rhoc, thetac, zc]
            rtz0 = self.cells.rtz_pos[cell]
            for edge in self.cells.junctions_edges[cell]:
                rtz1 = self.junctions.rtz_pos[edge.source()]
                rtz2 = self.junctions.rtz_pos[edge.target()]

                d01, d02, d12, area012 = triangle_geometry(rtz0, rtz1, rtz2)
                area += area012
                self.junctions.edge_lengths[edge] = d12
                perimeter += d12
                self.cells.areas[cell] = area
                self.cells.perimeters[cell] = perimeter
    
    def set_local_mask(self, cell):

        self.cells.is_local[cell] = 1
        for neighbour in cell.all_neighbours():
            self.cells.is_local[neighbour] = 1
        for j_vert in self.cells.junctions_vertices[cell]:
            self.junctions.is_local[j_vert] = 1

    def calc_total_energy(self):
        
        self.calc_apical_geometry()
        self.cells.graph.set_vertex_filter(None)
        elastic_term = 0.5 * self.cells.elasticities.a * (
            self.cells.areas.a - self.cells.prefered_area.a)**2
        contractile_term = 0.5 * self.cells.contractilities.a * (
            self.cells.perimeters.a**2)     
        tension_term = self.junctions.line_tensions.a * (
            self.junctions.edge_lengths.a)
        return elastic_term.sum() + (
            contractile_term.sum() + tension_term.sum())

    def type1_transition(self, elements, element_type='cells'):
        """Type one transition (see e.g 
        Farhadifar et al. Curr Biol. 2007 Dec 18;17(24):2095-104.
        Suppplementary figure S1
        
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
        
        cells = self.cells
        junctions = self.junctions
        if element_type == 'j_vertices':
            j_verta, j_vertb = elements
            j_edgeab = junctions.graph.edge(elements)
            cell1, cell3 = junctions.adjacent_cells[j_edgeab]
        elif element_type == 'j_edge':
            j_edgeab = elements
            j_verta, j_vertb = j_edgeab.source(), j_edgeab.target()
            cell1, cell3 = junctions.adjacent_cells[j_edgeab]
        elif element_type == 'cells':
            cell1 = elements[0]
            cell3 = elements[1]
            for je in cells.junctions_edges[cell1]:
                if je in cells.junctions_edges[cell3]:
                    j_edgeab = je
            j_verta = j_edgeab.source()
            j_vertb = j_edgeab.target()
        try:
            vecinos_a = junctions.all_vecinos(j_verta)
            vecinos_b = junctions.all_vecinos(j_vertb)
            j_vertc, j_vertf = vecinos_a[vecinos_a != j_vertb]
            j_vertd, j_verte = vecinos_b[vecinos_b != j_verta]
        except ValueError:
            print "Valid only for 3-way junctions"
            return

        j_edgeac = junctions.graph.edge(j_verta, j_vertc)
        j_edgeaf = junctions.graph.edge(j_verta, j_vertf)
        j_edgebd = junctions.graph.edge(j_vertb, j_vertd)
        j_edgeeb = junctions.graph.edge(j_verte, j_vertb)

        cell2 = junctions.adjacent_cells[j_edgebd][1]
        if cell2 == cell1:
            cell2 = junctions.adjacent_cells[j_edgebd][0]
        cell4 = junctions.adjacent_cells[j_edgeaf][1]
        if cell4 == cell3:
            cell4 = junctions.adjacent_cells[j_edgeac][0]
            
        # pi/2 rotation of junctions a and b around their center
        center = (junctions.sz_pos[j_verta].a +
                  junctions.sz_pos[j_vertb].a)/2.
        j_verta_sz = junctions.sz_pos[j_verta].a - center
        j_vertb_sz = junctions.sz_pos[j_vertb].a - center
        
        junctions.sigmas[j_verta] = j_verta_sz[1] + center[0]
        junctions.zeds[j_verta] = - j_verta_sz[0]  + center[1]
        junctions.thetas[j_verta] = (junctions.sigmas[j_verta]
                                     / junctions.rhos[j_verta]) % (2 * np.pi)

        junctions.sigmas[j_verta] = j_verta_sz[1] + center[0]
        junctions.zeds[j_verta] = - j_verta_sz[0]  + center[1]
        junctions.thetas[j_verta] = (junctions.sigmas[j_verta]
                                     / junctions.rhos[j_verta]) % (2 * np.pi)

        # redistributing junctions
        cells.junctions_vertices[cell1].remove(j_vertb)
        cells.junctions_vertices[cell3].remove(j_verta)
        cells.junctions_vertices[cell2].append(j_verta)
        cells.junctions_vertices[cell2].append(j_vertb)
        cells.junctions_vertices[cell4].append(j_verta)
        cells.junctions_vertices[cell4].append(j_vertb)        

        cells.junctions_edges[cell1].remove(j_edgeab)
        cells.junctions_edges[cell3].remove(j_edgeab)

        cells.junctions_edges[cell2].append(j_edgeab)
        cells.junctions_edges[cell4].append(j_edgeab)

        junctions.adjacent_cells[j_edgeab] = (cell2, cell4)

        edge23 = cells.graph.edge(cell1, cell3)
        cells.graph.remove_edge(edge23)
        ce24 = cells.graph.add_edge(cell2, cell4)

        junctions.graph.remove_edge(j_edgeaf)
        junctions.graph.remove_edge(j_edgebd)

        junctions.graph.add_edge(j_verta, j_vertd)
        junctions.graph.add_edge(j_vertb, j_vertf)
        
    #def division(self, cell):

        
        
def triangle_geometry(rtz0, rtz1, rtz2):
    c_code = """
    double r0 = rtz0[0];
    double t0 = rtz0[1];
    double z0 = rtz0[2];

    double r1 = rtz1[0];
    double t1 = rtz1[1];
    double z1 = rtz1[2];

    double r2 = rtz2[0];
    double t2 = rtz2[1];
    double z2 = rtz2[2];

    double d01 = sqrt(r0*r0 + r1*r1 - 2*r0*r1*cos(t1-t0) + (z1-z0)*(z1-z0));
    double d02 = sqrt(r0*r0 + r2*r2 - 2*r0*r2*cos(t2-t0) + (z2-z0)*(z2-z0));
    double d12 = sqrt(r1*r1 + r2*r2 - 2*r1*r2*cos(t2-t1) + (z2-z1)*(z2-z1));
    double p = d01 + d02 + d12;
    double area012 = sqrt(p * (p-d01) * (p-d02) * (p-d12));

    py::tuple results(4);
    results[0] = d01;
    results[1] = d02;
    results[2] = d12;
    results[3] = area012;
    return_val = results;
    
    """
    return weave.inline(c_code,
                        arg_names=['rtz0', 'rtz1', 'rtz2'],
                        headers=['<math.h>'])

