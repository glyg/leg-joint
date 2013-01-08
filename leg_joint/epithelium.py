#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import graph_tool.all as gt
import numpy as np
from scipy import weave


from .objects import  AbstractRTZGraph, Cells, ApicalJunctions
from .xml_handler import ParamTree
from .dynamics import Dynamics
from .optimizers import Optimizers

import filters

EpitheliumFilters = filters.EpitheliumFilters

CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(CURRENT_DIR)
PARAMFILE = os.path.join(ROOT_DIR, 'default', 'params.xml')

# See [the tau manifesto](http://tauday.com/tau-manifesto)
tau = 2. * np.pi


class Epithelium(EpitheliumFilters,
                 AbstractRTZGraph,
                 Dynamics,
                 Optimizers):
    """
    The ::class::`Epithelium` is the container for all the simulation.
    
    """
    def __init__(self, graphXMLfile=None, paramtree=None, paramfile=PARAMFILE):
        """
        
        """
        # Parametrisation
        if paramtree == None:
            self.paramtree = ParamTree(paramfile)
        else:
            self.paramtree = paramtree
        self.params = self.paramtree.absolute_dic

        # Graph instanciation
        if graphXMLfile is None:
            self.graph = gt.Graph(directed=True)
            self.new = True
        else:
            self.graph = gt.load_graph(graphXMLfile)
            self.new = False
            self.xmlfname = graphXMLfile

        self.__verbose__ = False
        EpitheliumFilters.__init__(self)
        # All the geometrical properties are packed here
        AbstractRTZGraph.__init__(self)
        # Cells and Junctions subgraphs initialisation
        print 'Initial cells'
        self.cells = Cells(self)
        print 'Initial junctions'
        self.junctions = ApicalJunctions(self)
        if self.new:
            self.is_alive.a = 1
            # Remove cell to cell edges after graph construction
            efilt = self.is_ctoj_edge.copy()
            efilt.a += self.is_junction_edge.a
            self.graph.set_edge_filter(efilt)
            self.graph.purge_edges()
            self.set_vertex_state()
            self.set_edge_state()

        self.diamonds = self.graph.new_edge_property('object')
        self.cells.update_junctions() #Also registers the triangles
        self.junctions.update_adjacent() #Also registers the diamonds
        # Dynamical components
        Dynamics.__init__(self)    
        Optimizers.__init__(self)
        if self.__verbose__: print 'isotropic relaxation'
        self.isotropic_relax()
        if self.__verbose__: print 'Periodic boundary'
        self.periodic_boundary_condition()
        if self.__verbose__: print 'Update apical'
        self.update_geometry()
        if self.__verbose__: print 'Update gradient'
        self.update_gradient()
        
    def __str__(self):
        num_cells = self.is_cell_vert.a.sum()
        num_edges = self.is_junction_edge.a.sum()
        str1 = ['Epithelium with %i cells and %i junction edges' % (num_cells,
                                                                    num_edges)]
        str1.append('Vertex Properties:\n'
                    '==================')
        for key in sorted(self.graph.vertex_properties.keys()):
            str1.append('    * %s' % key)
        str1.append('Edge Properties:\n'
                    '================')
        for key in sorted(self.graph.edge_properties.keys()):
            str1.append('    * %s' % key)
        return '\n'.join(str1)

    @property
    def num_sides(self):
        return self.graph.degree_property_map('out')
        
    def update_apical_geom(self):
        print 'Deprecated, use `update_geometry` instead'
        return self.update_geometry()
    
    def update_geometry(self):
        # Edges
        if self.__verbose__:
            print ('Geometry update on %i edges'
                   % self.graph.num_edges())
        for cell in self.cells:
            self.set_cell_pos(cell)
        self.update_deltas()
        self.update_edge_lengths()
        # Cells
        if self.__verbose__: print ('Cells geometry update on %i vertices'
                                    % self.graph.num_vertices())
        for j_edge in self.junctions:
            self.diamonds[j_edge].update_geometry()

        for cell in self.cells:
            self._one_cell_geom(cell)
        rho_lumen = self.params['rho_lumen']
        self.cells.vols.fa = self.cells.areas.fa * (self.rhos.fa
                                                    - rho_lumen)
    def _one_cell_geom(self, cell):
        """
        """
        j_edges = self.cells.junctions[cell]
        if len(j_edges) < 3:
            if not self.is_alive[cell]:
                return
            if self.__verbose__:
                print('''Two edges ain't enough to compute
                      area for cell %s''' % cell)
            self.cells.areas[cell] = self.cells.prefered_area[cell]
            self.cells.vols[cell] = self.cells.prefered_vol[cell]
            self.cells.perimeters[cell] = 0.
            self.is_alive[cell] = 0
        else:
            self.cells.areas[cell] = 0.
            self.cells.perimeters[cell] = 0.
            for j_edge in j_edges:
                tr = self.diamonds[j_edge].triangles[cell]
                self.cells.areas[cell] += tr.area * tr.sign
                self.cells.perimeters[cell] += tr.length
            
        
    def set_new_pos(self, new_rtz_pos):
        self.set_junction_pos(new_rtz_pos)
        
        
    @filters.active
    def set_junction_pos(self, new_rtz_pos):
        new_rtz_pos = new_rtz_pos.flatten()
        assert len(new_rtz_pos) / 3 == self.graph.num_vertices()
        self.rhos.fa = new_rtz_pos[::3]
        thetas = new_rtz_pos[1::3] / self.mean_rho
        thetas = thetas % tau
        self.thetas.fa = thetas
        self.zeds.fa = new_rtz_pos[2::3]
        self.sigmas.fa = self.thetas.fa * self.rhos.fa
    
    def set_cell_pos(self, cell):
        ##  Update cell position 
        j_rtz = np.array([[self.rhos[jv], self.thetas[jv], self.zeds[jv]]
                          for jv in cell.out_neighbours()])
        ### set z and rho
        self.rhos[cell] = j_rtz[:,0].mean()
        self.zeds[cell] = j_rtz[:,2].mean()
        ### set periodic theta
        raw_dtheta = j_rtz[:,1] - self.thetas[cell]
        pbc_theta = j_rtz[:,1]
        pbc_theta[raw_dtheta <= -tau/2] += tau
        pbc_theta[raw_dtheta > tau/2] -= tau
        self.thetas[cell] = pbc_theta.mean()
        self.sigmas[cell] = self.thetas[cell] * self.rhos[cell]
        
    def reset_topology(self):
        self.cells.update_junctions()
        self.junctions.update_adjacent()
        self.update_geometry()
        self.update_gradient()

    def check_phase_space(self, gamma, lbda):
        # See the energies.pynb notebook for the derivation of this:
        mu = 6 * np.sqrt(2. / (3 * np.sqrt(3)))
        if (gamma < - lbda / (2 * mu)):
            report= ("Contractility is too low,"
                     "Soft network not supported")
            return False, report
        if 2 * gamma * mu**2 > 4:
            lambda_max = 0.
        else:
            lambda_max = ((4 - 2 * gamma * mu**2) / 3.)**(3./2.) / mu
        if lbda > lambda_max:
            report = ("Invalid value for the line tension: "
                      "it should be lower than %.2f "
                      "for a contractility of %.2f "
                    % (lambda_max, gamma))
            return False, report
        return True, 'ok!'

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
            if self.__verbose__:
                print ('''Warning: previous %s to %s 
                       edge is re-created.'''
                       % (str(j_verta), str(j_vertb)))
            self.graph.remove_edge(j_edgeab)
        j_edgeab = self.graph.add_edge(j_verta, j_vertb)
        self.is_junction_edge[j_edgeab] = 1
        self.is_new_edge[j_edgeab] = 1
        self.is_ctoj_edge[j_edgeab] = 0
        line_tension0 = self.params['line_tension']
        self.junctions.line_tensions[j_edgeab] = line_tension0
        self.junctions.adjacent_cells[j_edgeab] = cell0, cell1
        if self.cells.junctions[cell0] is None:
            self.cells.junctions[cell0] = [j_edgeab,]
        else:
            self.cells.junctions[cell0].append(j_edgeab)

        if self.cells.junctions[cell1] is None:
            self.cells.junctions[cell1] = [j_edgeab,]
        else:
            self.cells.junctions[cell1].append(j_edgeab)

        ctoj_0a = self.graph.edge(cell0, j_verta)
        if ctoj_0a is not None:
            # print ("Warning: previous cell %s "
            #        "to junction vertex %s edge is re-created."
            #        ) % (str(cell0), str(j_verta))
            self.graph.remove_edge(ctoj_0a)
        ctoj_0a = self.graph.add_edge(cell0, j_verta)
        self.is_junction_edge[ctoj_0a] = 0
        self.is_ctoj_edge[ctoj_0a] = 1
        self.is_new_edge[ctoj_0a] = 1
        
        ctoj_0b = self.graph.edge(cell0, j_vertb)
        if ctoj_0b is not None:
            if self.__verbose__:
                print ('''
                       Warning: previous cell %s 
                       to junction vertex %s edge is re-created.
                       '''
                       % (str(cell0), str(j_vertb)))
            self.graph.remove_edge(ctoj_0b)
        ctoj_0b = self.graph.add_edge(cell0, j_vertb)
        self.is_junction_edge[ctoj_0b] = 0
        self.is_ctoj_edge[ctoj_0b] = 1
        self.is_new_edge[ctoj_0b] = 1
        self.set_cell_pos(cell0)

        if cell1 is not None:
            ctoj_1a = self.graph.edge(cell1, j_verta)
            if ctoj_1a is not None:
                if self.__verbose__ :
                    print('''
                          Warning: previous cell %s 
                          to junction vertex %s edge is re-created.
                          '''
                          % (str(cell1), str(j_verta)))
                self.graph.remove_edge(ctoj_1a)
            ctoj_1a = self.graph.add_edge(cell1, j_verta)
            self.is_junction_edge[ctoj_1a] = 0
            self.is_ctoj_edge[ctoj_1a] = 1
            self.is_new_edge[ctoj_1a] = 1

            ctoj_1b = self.graph.edge(cell1, j_vertb)
            if ctoj_1b is not None:
                if self.__verbose__ :
                    print('''
                          Warning: previous cell %s 
                          to junction vertex %s edge is re-created.
                          '''
                          % (str(cell1), str(j_vertb)))
                self.graph.remove_edge(ctoj_1b)
            ctoj_1b = self.graph.add_edge(cell1, j_vertb)
            self.is_junction_edge[ctoj_1b] = 0
            self.is_ctoj_edge[ctoj_1b] = 1
            self.is_new_edge[ctoj_1b] = 1
            self.set_cell_pos(cell1)
        return j_verta, j_vertb, cell0, cell1

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
            j_edgeab = self.graph.edge(j_vertb, j_verta)
        if j_edgeab is None:
            print "Warning: junction from %s to %s doesn't exist" % (
                str(j_edgeab.source()), str(j_edgeab.target()))
            return
        self.cells.junctions[cell0].remove(j_edgeab)
        self.cells.junctions[cell1].remove(j_edgeab)
        self.junctions.adjacent_cells[j_edgeab] = []
        self.graph.remove_edge(j_edgeab)
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

    @filters.j_edges_in
    def merge_j_verts(self, jv0, jv1):
        vertex_trash = []
        edge_trash = []
        je = self.any_edge(jv0, jv1)
        if je is None:
            raise ValueError('Can only merge connected edges')
            
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

