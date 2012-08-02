#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy import optimize
import graph_tool.all as gt
from xml_handler import ParamTree
from scipy import weave

from objects import  AbstractRTZGraph, Cells, AppicalJunctions

CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(CURRENT_DIR)
PARAMFILE = os.path.join(ROOT_DIR, 'default', 'params.xml')

# See (the tau manifesto)[http://tauday.com/tau-manifesto]
tau = 2. * np.pi


class Epithelium(AbstractRTZGraph):
    """
    """
    def __init__(self, paramtree=None, paramfile=PARAMFILE):
        """
        """
        # Parametrisation
        if paramtree == None:
            self.paramtree = ParamTree(paramfile)
        else:
            self.paramtree = paramtree
        self.params = self.paramtree.absolute_dic
        # Graph instanciation
        self.graph = gt.Graph(directed=True)
        # All the geometrical properties are packed here
        AbstractRTZGraph.__init__(self)
        # Cells and Junctions subgraphs initialisation
        self.cells = Cells(self)
        self.update_deltas()
        self.update_edge_lengths()
        self.junctions = AppicalJunctions(self)

        self.update_apical_geom()
        # self.relax_rhos()
        # self.update_energy()


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
    def energy_grad(self):
        return self.graph.vertex_properties["energy_grad"]
    @property
    def is_local_vert(self):
        return self.graph.vertex_properties["is_local_vert"]
    @property
    def is_local_cell(self):
        return self.graph.vertex_properties["is_local_cell"]
    @property
    def is_local_j_vert(self):
        return self.graph.vertex_properties["is_local_j_vert"]
    @property
    def is_local_j_edge(self):
        return self.graph.edge_properties["is_local_j_edge"]
    @property
    def is_local_edge(self):
        return self.graph.edge_properties["is_local_edge"]
    @property
    def is_local_both(self):
        return self.graph.vertex_properties["is_local_both"]

    def find_energy_min(self, tol=1e-8):
        vfilt = self.is_local_vert.copy()
        vfilt.a *= (1 - self.is_cell_vert.a)
        self.graph.set_vertex_filter(vfilt)
        pos0 = np.array([(s, z) for s, z in zip(self.sigmas.fa,
                                                self.zeds.fa)])
        max_sigma_d = np.abs(self.dsigmas.fa).mean() * 2.
        max_zed_d = max_sigma_d
        bounds = np.zeros((2 * pos0.shape[0],),
                          dtype=[('min', np.float32),
                                 ('max', np.float32)])
        s_bounds = np.vstack((self.sigmas.fa - max_sigma_d,
                             self.sigmas.fa + max_sigma_d)).T
        z_bounds = np.vstack((self.zeds.fa - max_zed_d,
                             self.zeds.fa + max_zed_d)).T
        print z_bounds.shape
        for n in range(pos0.shape[0]):
            bounds[2 * n] = (s_bounds[n, 0], s_bounds[n, 1])
            bounds[2 * n + 1] = (z_bounds[n, 0], z_bounds[n, 1])
        # print "bounds :"+str(bounds)
        self.graph.set_vertex_filter(None)


        # output = optimize.fmin(self.opt_energy,
        #                        pos0.flatten(),
        #                        ftol=tol, xtol=0.01,
        #                        args=(self.is_local_vert,),
        #                        callback=self.opt_callback)

        # try:
        #     output = optimize.fmin_ncg(self.opt_energy,
        #                                pos0.flatten(),
        #                                fprime=self.opt_gradient,
        #                                avextol=tol,
        #                                retall=True,
        #                                args=(self.is_local_vert,),
        #                                callback=self.opt_callback)
        # except:
        #     self.graph.set_vertex_filter(vfilt)            
        #     self.set_new_pos(pos0, vfilt)
        #     self.graph.set_vertex_filter(None)            
        #     print "unable to find local min"
        #     output = 0

        # output = optimize.fmin_bfgs(self.opt_energy,
        #                             pos0.flatten(),
        #                             fprime=self.opt_gradient,
        #                             args=(self.is_local_vert,),
        #                             gtol=tol,
        #                             norm=np.inf,
        #                             retall=1,
        #                             callback=self.opt_callback)

        output = optimize.fmin_l_bfgs_b(self.opt_energy,
                                        pos0.flatten(),
                                        fprime=None, #self.opt_gradient,
                                        args=(self.is_local_vert,),
                                        approx_grad=1,
                                        bounds=bounds.flatten(),
                                        m=10,
                                        factr=1e12,
                                        pgtol=tol,
                                        epsilon=1e-08,
                                        iprint=1,
                                        maxfun=150,
                                        disp=None)
        return pos0, output

    def opt_energy(self, sz_pos, vfilt):
        # Position setting
        # print 'opt_energy : '+str(sz_pos[1::2])
        self.set_new_pos(sz_pos, vfilt)
        self.update_apical_geom(vfilt)
        energy = self.calc_energy(vfilt)
        return energy

    def opt_gradient(self, sz_pos, vfilt):
        # # position setting
        self.set_new_pos(sz_pos, vfilt)
        self.update_apical_geom(vfilt)
        self.update_gradient(vfilt)
        gradient = self.calc_gradient(vfilt)
        return gradient

    def opt_callback(self, sz_pos):
        self.update_apical_geom(self.is_local_vert)
        self.update_gradient(self.is_local_vert)

    def calc_energy(self, vfilt=None, efilt=None):
        # Cell vertices (filtered by the vertex filter `vfilt`)
        if vfilt is None:
            vfilt = self.is_cell_vert
        else:
            vfilt = vfilt.copy()
            vfilt.a *= self.is_cell_vert.a
        self.graph.set_vertex_filter(self.is_cell_vert)
        elastic_term = 0.5 * self.cells.elasticities.fa * (
            self.cells.areas.fa - self.cells.prefered_area.fa)**2
        contractile_term = 0.5 * self.cells.contractilities.fa * \
                           self.cells.perimeters.fa**2
        self.graph.set_vertex_filter(None)
        # Appical junction related terms (filtered over by `efilt`)
        if efilt is None:
            efilt = self.is_junction_edge.copy()
        else:
            efilt = efilt.copy()
            efilt.a *= self.is_junction_edge.a
        self.graph.set_edge_filter(self.is_junction_edge)
        tension_term = self.junctions.line_tensions.fa * \
                       self.junctions.edge_lengths.fa
        self.graph.set_edge_filter(None)
        return elastic_term.sum() + contractile_term.sum() + tension_term.sum()

    def calc_gradient(self, vfilt=None):
        if vfilt == None:
            self.graph.set_vertex_filter(self.is_cell_vert, inverted=True)
        else:
            vfilt = vfilt.copy()
            vfilt.a *= (1 - self.is_cell_vert.a)
            self.graph.set_vertex_filter(vfilt)
        gradient = np.zeros(self.graph.num_vertices() * 2)
        gradient[::2] = - self.grad_sigma.fa
        gradient[1::2] = - self.grad_zed.fa
        # print 'calc_grad : '+str(self.zeds.fa)
        self.graph.set_vertex_filter(None)
        return gradient

    
    def update_apical_geom(self, vfilt=None, efilt=None):
        """
        The area is approximated as the sum
        of the areas of the triangles
        formed by the cell position and each junction
        """
        self.update_deltas(efilt)
        self.update_edge_lengths(efilt)
        self.update_cell_positions(vfilt)
        # cell to junctions edges only
        if efilt == None:
            efilt = self.is_ctoj_edge.copy()
        else:
            efilt = efilt.copy()
            efilt.a *= self.is_ctoj_edge.a
        self.graph.set_edge_filter(efilt)
        self.graph.set_vertex_filter(vfilt)
        for cell in self.cells:
            if cell.out_degree() < 3:
                #print 'Lonely cell %s' % cell
                continue
            area = 0.
            perimeter = 0.
            j_edges = self.cell_junctions(cell)
            if len(j_edges) < 3:
                self.cells.areas[cell] = self.cells.prefered_area[cell]
                self.cells.perimeters[cell] = 0.
                #print "Less than 3 edges found for cell %s" % str(cell)
                continue
            for j_edge in j_edges:
                if j_edge is None:
                    continue
                perimeter += self.edge_lengths[j_edge]
                ctoj0 = self.graph.edge(cell, j_edge.source())
                ctoj1 = self.graph.edge(cell, j_edge.target())
                area += np.abs(self.dsigmas[ctoj0] * self.dzeds[ctoj1]
                               - self.dsigmas[ctoj1] * self.dzeds[ctoj0])/2.
            self.cells.areas[cell] = area
            self.cells.perimeters[cell] = perimeter
        self.graph.set_vertex_filter(None)
        self.graph.set_edge_filter(None)
        
    def update_cell_positions(self, vfilt=None):
        if vfilt == None:
            self.graph.set_vertex_filter(None)
        else:
            vfilt = vfilt.copy()
            vfilt.a += (1 - self.is_cell_vert.a)
            self.graph.set_vertex_filter(vfilt)
        efilt = self.is_ctoj_edge.copy()
        #efilt.a += (1 - self.at_boundary.a)
        self.graph.set_edge_filter(efilt)
        for cell in self.cells:
            j_sigma = 0.
            j_zed = 0.
            if cell.out_degree() < 4:
                continue
            for je in cell.out_edges():
                sigma = self.sigmas[je.target()]
                if not self.at_boundary[je]:
                    j_sigma += sigma
                elif sigma > self.sigmas[cell]:
                    j_sigma += sigma - tau * self.rhos[cell]
                elif sigma < self.sigmas[cell]:
                    j_sigma += sigma + tau * self.rhos[cell]
                j_zed += self.zeds[je.target()]
            self.sigmas[cell] = j_sigma / cell.out_degree()
            self.zeds[cell] = j_zed / cell.out_degree()
        self.graph.set_vertex_filter(None)
        self.graph.set_edge_filter(None)

    def update_gradient(self, vfilt=None, efilt=None, inverted=False):
        # Cell vertices
        if vfilt == None:
            vfilt = self.is_cell_vert.copy()
        else:
            vfilt = vfilt.copy()
            vfilt.a *= self.is_cell_vert.a
        self.graph.set_vertex_filter(vfilt)
        elastic_term =  self.cells.elasticities.fa \
                       * (self.cells.areas.fa - self.cells.prefered_area.fa )
        contractile_term =  self.cells.contractilities.fa \
                           * self.cells.perimeters.fa
        cell_grad = elastic_term + contractile_term
        self.graph.vertex_properties["energy_grad"].fa = cell_grad
        self.graph.set_vertex_filter(None)
        # Junction edges
        tension = self.junctions.line_tensions
        self.graph.set_edge_filter(efilt)
        self.grad_sigma.fa = 0.
        self.grad_zed.fa = 0.

        for edge in self.junctions:
            j_src, j_trgt = edge.source(), edge.target()
            for cell in self.adjacent_cells(edge):
                perp_sigma, perp_zed =  self.outward_uvect(cell, edge)
                self.grad_sigma[j_src] += self.energy_grad[cell] * perp_sigma
                self.grad_zed[j_src] += self.energy_grad[cell] * perp_zed
                self.grad_sigma[j_trgt] += self.energy_grad[cell] * perp_sigma
                self.grad_zed[j_trgt] += self.energy_grad[cell] * perp_zed
            u_sg = self.u_dsigmas[edge]
            u_zg = self.u_dzeds[edge]
            self.grad_sigma[j_src] += tension[edge] * u_sg
            self.grad_zed[j_src] += tension[edge] * u_zg
            self.grad_sigma[j_trgt] += - tension[edge] * u_sg
            self.grad_zed[j_trgt] += - tension[edge] * u_zg

    def outward_uvect(self, cell, j_edge):
        """
        Returns the (sigma, zed) coordinates of the unitary vector
        perpendicular to the junction edge `j_edge` and the cell `cell`
        """
        if not cell in self.adjacent_cells(j_edge):
            raise AttributeError("Cell not adjacent to junction")
        perp_zed = self.u_dsigmas[j_edge]
        perp_sigma = - self.u_dzeds[j_edge]
        ctoj1 = self.graph.edge(cell, j_edge.source())
        ctoj2 = self.graph.edge(cell, j_edge.target())
        median_dzed = (self.dzeds[ctoj1] + self.dzeds[ctoj2])
        median_dsigma = (self.dsigmas[ctoj1] + self.dsigmas[ctoj2])
        if perp_sigma * median_dsigma < 0:
            perp_sigma *= -1
        if perp_zed * median_dzed < 0:
            perp_zed *= -1
        return perp_sigma, perp_zed

    def set_local_mask(self, cell):
        self.graph.set_edge_filter(None)
        self.graph.set_vertex_filter(None)
        cell = self.graph.vertex(cell)
        self.is_local_vert[cell] = 1
        for neighbour in cell.all_neighbours():
            self.is_local_vert[neighbour] = 1
            for edge in neighbour.all_edges():
                self.is_local_edge[edge] = 1
        self.is_local_cell.a = self.is_local_vert.a \
                               * self.is_cell_vert.a \
                               + (1 - self.is_cell_vert.a)

    def remove_local_mask(self, cell):
        self.graph.set_edge_filter(None)
        self.graph.set_vertex_filter(None)
        self.is_local_vert[cell] = 0
        for neighbour in cell.all_neighbours():
            self.is_local_vert[neighbour] = 0
            for edge in neighbour.all_edges():
                self.is_local_edge[edge] = 0
        self.is_local_cell.a = self.is_local_vert.a \
                               * self.is_cell_vert.a\
                               + (1 - self.is_cell_vert.a)
    
    def type1_transition(self, elements):
        """
        Type one transition (see the definition in
        Farhadifar et al. Curr Biol. 2007 Dec 18;17(24):2095-104.
        Suppplementary figure S1)
        
        In ASCII art (letters represent junctions and number represent cells):

        e 2 d                  
         \ /         e  d        e  2  d  
          b           \/          \   /
        1 | 3  ---->  ab  ----> 1  a-b  3  
          a           /\          /   \    
         / \         f  c        f  4  c 
        f 4 c                     

        Paramters
        =========
        elements: graph edge or vertex:
            Can be either:
            * two cell vertices (1 and 3),
            * two junction vertices (a and b)
            * or a single edge (between a and b)
        """
        # Cells
        if len(elements) == 2 and self.is_cell_vert[elements[0]]:
            cell1 = elements[0]
            cell3 = elements[1]
            j_edges1 =  self.cell_junctions(cell1)
            j_edges3 =  self.cell_junctions(cell3)
            try:
                j_edgeab = [je for je in j_edges1 if je in j_edges3][0]
            except IndexError:
                print ("No valid junction found "
                       "beetween cells %s and %s " % (cell1, cell3))
                return
            j_verta = j_edgeab.source()
            j_vertb = j_edgeab.target()
        # Junction vertices
        elif len(elements) == 2 and not self.is_cell_vert[elements[0]]:
            j_verta, j_vertb = elements
            j_edgeab = self.graph.edge(elements)
            if j_edgeab is None:
                print "Invalid junction %s" % str(j_edgeab)
                return
            try:
                cell1, cell3 = self.adjacent_cells(j_edgeab)
            except ValueError:
                print ("No adgacent cells found"
                       "for junction %s" % str(j_edgeab))
                return
        # Junction edges
        elif self.is_junction_edge(elements):
            j_edgeab = elements
            j_verta, j_vertb = j_edgeab.source(), j_edgeab.target()
            try:
                cell1, cell3 = self.adjacent_cells(j_edgeab)
            except ValueError:
                print ("No adgacent cells found"
                       "for junction %s" % str(j_edgeab))
                return
        else:
            raise ValueError("Invalid argument %s" % str(elements))
        try:
            vecinos_a = [jv for jv in self.ordered_neighbours(j_verta)
                         if not self.is_cell_vert[jv]]
            vecinos_b = [jv for jv in self.ordered_neighbours(j_vertb)
                         if not self.is_cell_vert[jv]]
            j_vertf, j_vertc = [jv for jv in vecinos_a if jv != j_vertb]
            j_verte, j_vertd = [jv for jv in vecinos_b if jv != j_verta]
        except ValueError:
            print "Valid only for 3-way junctions"
            return
        j_edgeac = self.graph.edge(j_verta, j_vertc)
        if j_edgeac is None:
            j_edgeac = self.graph.edge(j_vertc, j_verta)
        j_edgebe = self.graph.edge(j_vertb, j_verte)
        if j_edgebe is None:
            j_edgebe = self.graph.edge(j_verte, j_vertb)
        if j_edgebe is None or j_edgeac is None:
            print "Invalid geometry"
            return
        if not cell1 in self.adjacent_cells(j_edgeac):
            j_vertc, j_vertf = j_vertf, j_vertc
            j_edgeac = self.graph.edge(j_verta, j_vertc)
            if j_edgeac is None:
                j_edgeac = self.graph.edge(j_vertc, j_verta)
        if not cell3 in self.adjacent_cells(j_edgebe):
            j_verte, j_vertd = j_vertd, j_verte
            j_edgebe = self.graph.edge(j_vertb, j_verte)
            if j_edgebe is None:
                j_edgebe = self.graph.edge(j_verte, j_vertb)
        cell2 = self.adjacent_cells(j_edgebe)[1]
        print "adjacent cells edge be : %s, %s" % (
            str(self.adjacent_cells(j_edgebe)[0]),
            str(self.adjacent_cells(j_edgebe)[1]))
        if cell2 == cell3:
            cell2 = self.adjacent_cells(j_edgebe)[0]
        cell4 = self.adjacent_cells(j_edgeac)[1]
        if cell4 == cell1:
            cell4 = self.adjacent_cells(j_edgeac)[0]

        modified_cells = [cell1, cell2, cell3, cell4]
        modified_jverts = [j_verta, j_vertb, j_vertc,
                           j_vertd, j_verte, j_vertf]
        for cell, i in zip(modified_cells, [1, 2, 3, 4]):
            print 'cell %i: %s' %(i, str(cell))
        for jv, i in zip(modified_jverts, 'abcdef'):
            print 'junction vertice %s: %s' %(i, str(jv))

        self.remove_junction(j_verta, j_vertb, cell1, cell3)
        self.remove_junction(j_verta, j_vertc, cell1, cell4)
        self.remove_junction(j_vertb, j_verte, cell2, cell3)
        self.add_junction(j_verta, j_vertb, cell2, cell4)
        self.add_junction(j_verta, j_verte, cell2, cell3)
        self.add_junction(j_vertb, j_vertc, cell1, cell4)

        sigma_a = self.sigmas[j_verta]
        sigma_b = self.sigmas[j_vertb]
        zed_a = self.zeds[j_verta]
        zed_b = self.zeds[j_vertb]
        
        center_sigma = (sigma_a + sigma_b)/2.
        center_zed = (zed_a + zed_b)/2.

        delta_s = np.abs(sigma_b - sigma_a)
        delta_z = np.abs(zed_b - zed_a)
        if self.sigmas[cell1] < self.sigmas[cell3]:
            self.sigmas[j_verta] = center_sigma + delta_z/2.
            self.sigmas[j_vertb] = center_sigma - delta_z/2.
        else:
            self.sigmas[j_vertb] = center_sigma + delta_z/2.
            self.sigmas[j_verta] = center_sigma - delta_z/2.
        if self.zeds[cell1] < self.zeds[cell3]:
            self.zeds[j_verta] = center_zed + delta_s/2.
            self.zeds[j_vertb] = center_zed - delta_s/2.
        else:
            self.zeds[j_vertb] = center_zed + delta_s/2.
            self.zeds[j_verta] = center_zed - delta_s/2.

        self.set_local_mask(cell1)
        self.set_local_mask(cell3)
        
        self.update_apical_geom(vfilt=self.is_local_cell,
                                efilt=None)
        return modified_cells, modified_jverts

    def cell_division(self, mother_cell,
                      phi_division=np.pi/7,
                      verbose=False):
        tau = 2 * np.pi
        daughter_cell = self.graph.add_vertex()
        self.rhos[daughter_cell] = self.rhos[mother_cell]
        self.zeds[daughter_cell] = self.zeds[mother_cell]
        self.sigmas[daughter_cell] = self.sigmas[mother_cell]
        print "Cell %s is born" % str(daughter_cell)
        self.is_cell_vert[daughter_cell] = 1
        if self.is_local_vert[mother_cell]:
            self.set_local_mask(daughter_cell)
        sigmas_rel = self.sigmas.fa  - self.sigmas[mother_cell]
        zeds_rel = self.zeds.fa  - self.zeds[mother_cell]
        # arctan is defined between -pi and pi
        phi = np.arctan2(sigmas_rel, zeds_rel) + tau/2
        phi_rel = (phi - phi_division ) % tau
        junction_trash = []
        new_junctions = []
        new_jvs = []
        for j_edge in self.cell_junctions(mother_cell):
            if j_edge is None:
                continue
            j_src = j_edge.source()
            j_trgt = j_edge.target() 
            # 
            phi_trgt, phi_src = phi_rel[j_trgt], phi_rel[j_src]
            dphi = (phi_trgt - phi_src) % tau
            if dphi < tau/2:
                phi_trgt, phi_src = phi_src, phi_trgt
                j_src, j_trgt = j_trgt, j_src
                if verbose: print 'swap:!'
            # print "Between %s and %s" %(j_src, j_trgt)
            # print "@ "+str((phi_trgt, phi_src))
            if phi_src > tau/2 and phi_trgt > tau/2:
                # print 'both higher than pi'
                continue
            elif phi_src <= tau/2 and phi_trgt <= tau/2:
                cell0, cell1 = self.adjacent_cells(j_edge)
                junction_trash.append((j_src, j_trgt, cell0, cell1))
                adj_cell = cell1 if cell0 == mother_cell else cell0
                new_junctions.append((j_src, j_trgt, adj_cell, daughter_cell))
                # print 'both lower than pi'
            elif phi_src > tau/2 and phi_trgt <= tau/2:
                cell0, cell1 = self.adjacent_cells(j_edge)
                adj_cell = cell1 if cell0 == mother_cell else cell0
                new_jv = self.graph.add_vertex()
                new_jvs.append(new_jv)
                self.is_cell_vert[new_jv] = 0
                # New junction in the midle of the edge
                self.sigmas[new_jv] = (self.sigmas[j_src] +
                                        self.sigmas[j_trgt]) / 2.
                self.zeds[new_jv] = (self.zeds[j_src] +
                                      self.zeds[j_trgt]) / 2.
                self.rhos[new_jv] = (self.rhos[j_src] +
                                      self.rhos[j_trgt]) / 2.
                junction_trash.append((j_src, j_trgt, cell0, cell1))
                new_junctions.append((new_jv, j_src,
                                      adj_cell, mother_cell))
                new_junctions.append((new_jv, j_trgt,
                                      adj_cell, daughter_cell))
                # print "Source to mother"
                # print "Added vertex %s " % str(new_jv)
            elif phi_src <= tau/2 and phi_trgt > tau/2:
                cell0, cell1 = self.adjacent_cells(j_edge)
                adj_cell = cell1 if cell0 == mother_cell else cell0
                new_jv = self.graph.add_vertex()
                new_jvs.append(new_jv)
                self.is_cell_vert[new_jv] = 0
                # New junction in the midle of the edge
                self.sigmas[new_jv] = (self.sigmas[j_src] +
                                        self.sigmas[j_trgt]) / 2.
                self.zeds[new_jv] = (self.zeds[j_src] +
                                      self.zeds[j_trgt]) / 2.
                self.rhos[new_jv] = (self.rhos[j_src] +
                                      self.rhos[j_trgt]) / 2.
                junction_trash.append((j_src, j_trgt, cell0, cell1))
                new_junctions.append((new_jv, j_src,
                                      adj_cell, daughter_cell))
                new_junctions.append((new_jv, j_trgt,
                                      adj_cell, mother_cell))
                # print "Source to daughter"
                # print "Added vertex %s " % str(new_jv)

        for (j_src, j_trgt, cell0, cell1) in junction_trash:
            self.remove_junction(j_src, j_trgt, cell0, cell1)
        for (j_src, j_trgt, cell0, cell1) in new_junctions:
            j = self.add_junction(j_src, j_trgt, cell0, cell1)
        # Cytokinesis
        j = self.add_junction(new_jvs[0], new_jvs[1],
                              mother_cell, daughter_cell)
        self.set_local_mask(daughter_cell)
        # Updates
        vfilt = self.is_local_vert.copy()
        vfilt.a = self.is_local_vert.a * self.is_cell_vert.a \
                  + 1 - self.is_cell_vert.a
        self.update_apical_geom(vfilt=vfilt)
        #self.update_energy(vfilt=self.is_local_vert)
        return j

        
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
            # print ("Warning: previous %s to %s "
            #        "edge is re-created." % (str(j_verta),
            #                                 str(j_vertb)))
            self.graph.remove_edge(j_edgeab)
        j_edgeab = self.graph.add_edge(j_verta, j_vertb)
        self.is_junction_edge[j_edgeab] = 1
        self.is_ctoj_edge[j_edgeab] = 0
        line_tension0 = self.params['line_tension']
        self.junctions.line_tensions[j_edgeab] = line_tension0

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
            # print ("Warning: previous cell %s "
            #        "to junction vertex %s edge is re-created."
            #        ) % (str(cell0), str(j_vertb))
            self.graph.remove_edge(ctoj_0b)
        ctoj_0b = self.graph.add_edge(cell0, j_vertb)
        self.is_junction_edge[ctoj_0b] = 0
        self.is_ctoj_edge[ctoj_0b] = 1
        self.is_new_edge[ctoj_0b] = 1


        if cell1 is not None:
            ce01 = self.graph.edge(cell0, cell1)
            if ce01 is not None:
                # print ("Warning: previous cell %s "
                #        "to cell %s edge is re-created."
                #        ) % (str(cell0), str(cell1))
                self.graph.remove_edge(ce01)
            ce01 = self.graph.add_edge(cell0, cell1)
            self.is_junction_edge[ce01] = 0
            self.is_ctoj_edge[ce01] = 0
            self.is_new_edge[ce01] = 1

            ctoj_1a = self.graph.edge(cell1, j_verta)
            if ctoj_1a is not None:
                # print ("Warning: previous cell %s "
                #        "to junction vertex %s edge is re-created."
                #        ) % (str(cell1), str(j_verta))
                self.graph.remove_edge(ctoj_1a)
            ctoj_1a = self.graph.add_edge(cell1, j_verta)
            self.is_junction_edge[ctoj_1a] = 0
            self.is_ctoj_edge[ctoj_1a] = 1
            self.is_new_edge[ctoj_1a] = 1

            ctoj_1b = self.graph.edge(cell1, j_vertb)
            if ctoj_1b is not None:
                # print ("Warning: previous cell %s "
                #        "to junction vertex %s edge is re-created."
                #        ) % (str(cell1), str(j_vertb))
                self.graph.remove_edge(ctoj_1b)
            ctoj_1b = self.graph.add_edge(cell1, j_vertb)
            self.is_junction_edge[ctoj_1b] = 0
            self.is_ctoj_edge[ctoj_1b] = 1
            self.is_new_edge[ctoj_1b] = 1

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
            ce01 = self.graph.edge(cell0, cell1)
            if ce01 is not None:
                self.graph.remove_edge(ce01)
            else:
                ce10 = self.graph.edge(cell1, cell0)
                self.graph.remove_edge(ce10)

    ## Junction edges related  functions 
    def adjacent_cells(self, j_edge, efilt=None):
        if efilt == None:
            efilt = self.is_ctoj_edge
        self.graph.set_edge_filter(efilt)
        jv0 = j_edge.source()
        jv1 = j_edge.target()
        cells_a = [cell for cell in jv0.in_neighbours()]
        cells_b = [cell for cell in jv1.in_neighbours()]
        common_cells = [cell for cell in cells_a if cell in cells_b]
        self.graph.set_edge_filter(None)
        return common_cells

    ## Cell vertices topological functions
    def cell_junctions(self, cell):
        self.graph.set_edge_filter(self.is_ctoj_edge)
        self.graph.set_vertex_filter(None)
        j_edges = []
        for jv0 in cell.out_neighbours():
            for jv1 in cell.out_neighbours():
                e = self.any_edge(jv0, jv1)
                if e is None or e in j_edges: continue
                if self.is_junction_edge[e]: j_edges.append(e)
        self.graph.set_edge_filter(None)
        return j_edges
                
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

