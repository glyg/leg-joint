#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import glob
#import filters
from graph_representation import epithelium_draw
from optimizers import find_energy_min

from datetime import datetime

CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(CURRENT_DIR)
GRAPH_SAVE_DIR = os.path.join(ROOT_DIR, 'saved_graphs')

def snapshot(func, *args, **kwargs):
    def new_func(eptm, *args, **kwargs):
        out = func(eptm, *args, **kwargs)
        now = datetime.now()
        xml_save = os.path.join(GRAPH_SAVE_DIR,
                                'xml', 'tmp',
                                'eptm_%s.xml' % now.isoformat())
        eptm.graph.save(xml_save)
        # outfname2d = os.path.join(GRAPH_SAVE_DIR,
        #                           'png', 'tmp',
        #                           'eptm2d_%s.png' % now.isoformat())
        # outfname3d = os.path.join(GRAPH_SAVE_DIR,
        #                           'png', 'tmp',
        #                           'eptm3d_%s.png' % now.isoformat())
        # epithelium_draw(eptm, output2d=outfname2d, output3d=outfname3d)
        return out
    return new_func


def apoptosis(eptm, a_cell, idx=0,
              vol_reduction=0.01,
              contractility=1.,
              radial_tension=0.):
    
    eptm.set_local_mask(None)
    eptm.set_local_mask(a_cell)
    
    eptm.cells.prefered_vol[a_cell] *= vol_reduction
    eptm.cells.contractilities[a_cell] *= contractility
    mean_lt = np.array([eptm.junctions.line_tensions[je]
                        for je in eptm.cells.junctions[a_cell]]).mean()
    for jv in a_cell.out_neighbours():
        eptm.junctions.radial_tensions[jv] = radial_tension * mean_lt
        for v in jv.all_neighbours():
            if v == a_cell: 
                continue
            if not eptm.is_cell_vert[v]:
                continue
            eptm.set_local_mask(v)
    find_energy_min(eptm, tol=1e-4)
    tag = 'vr%.2f_ctr%.2f_rt%.2f' % (vol_reduction, contractility,
                                     radial_tension)

    png_dir = os.path.join(GRAPH_SAVE_DIR, 'png', 'apopto_'+tag)
    try:
        os.mkdir(png_dir)
    except OSError:
        pass

    xml_dir = os.path.join(GRAPH_SAVE_DIR, 'xml', 'apopto_'+tag)
    try:
        os.mkdir(xml_dir)
    except OSError:
        pass
    out2d = os.path.join(png_dir, 'apopto_2d_%04i.png' % idx)
    out3d = os.path.join(png_dir, 'apopto_3d_%04i.png' % idx)
    out_xml = os.path.join(xml_dir, 'apopto_%04i.xml' % idx)
    eptm.graph.save(out_xml)
    epithelium_draw(eptm, output2d=out2d, output3d=out3d, d_theta=np.pi/2)

@snapshot
def type1_transition(eptm, elements, verbose=False):
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
     / \         c  f        c  4  f 
    c 4 f                     

    Paramters
    =========
    elements: graph edge or vertex:
        Can be either:
        * two cell vertices (1 and 3),
        * two junction vertices (a and b)
        * or a single edge (between a and b)
    """
    #### Parsing arguments
    # Cells
    if len(elements) == 2 and eptm.is_cell_vert[elements[0]]:
        cell1 = elements[0]
        cell3 = elements[1]
        j_edges1 =  eptm.cells.junctions[cell1]
        j_edges3 =  eptm.cells.junctions[cell3]
        try:
            j_edgeab = [je for je in j_edges1 if je in j_edges3][0]
        except IndexError:
            print ("No valid junction found "
                   "beetween cells %s and %s " % (cell1, cell3))
            return
        j_verta = j_edgeab.source()
        j_vertb = j_edgeab.target()
    # Junction vertices
    elif len(elements) == 2 and not eptm.is_cell_vert[elements[0]]:
        j_verta, j_vertb = elements
        j_edgeab = eptm.graph.edge(elements)
        if j_edgeab is None:
            print "Invalid junction %s" % str(j_edgeab)
            return
        try:
            cell1, cell3 = eptm.junctions.adjacent_cells[j_edgeab]
        except ValueError:
            print ("No adgacent cells found"
                   "for junction %s" % str(j_edgeab))
            return
    # Junction edges
    elif eptm.is_junction_edge(elements):
        j_edgeab = elements
        j_verta, j_vertb = j_edgeab.source(), j_edgeab.target()
        try:
            cell1, cell3 = eptm.junctions.adjacent_cells[j_edgeab]
        except ValueError:
            print ("No adgacent cells found"
                   "for junction %s" % str(j_edgeab))
            return
    else:
        raise ValueError("Invalid argument %s" % str(elements))
    #### Done parsing arguments

    try:
        vecinos_a = [jv for jv in eptm.ordered_neighbours(j_verta)
                     if not eptm.is_cell_vert[jv]]
        vecinos_b = [jv for jv in eptm.ordered_neighbours(j_vertb)
                     if not eptm.is_cell_vert[jv]]
        j_vertf, j_vertc = [jv for jv in vecinos_a if jv != j_vertb]
        j_verte, j_vertd = [jv for jv in vecinos_b if jv != j_verta]
    except ValueError:
        print('''
              Valid only for 3-way junctions
              For type 1 transition between cells
              %s and %s
              ''' %(str(cell1), str(cell3)))
        eptm.set_local_mask(cell1)
        eptm.set_local_mask(cell3)
        return None

    j_edgeac = eptm.any_edge(j_verta, j_vertc)
    j_edgebe = eptm.any_edge(j_vertb, j_verte)
    if None in (j_edgebe, j_edgeac):
        raise ValueError("Invalid geometry")


    # #### come back to the documented name convention
    # if eptm.zeds[cell1] < eptm.zeds[cell3]:
    #     #switch 1 and 3
    #     cell3, cell1 = cell1, cell3
    
        
    if not cell1 in eptm.junctions.adjacent_cells[j_edgeac]:
        #Switch f and c
        j_vertc, j_vertf = j_vertf, j_vertc 
        j_edgeac = eptm.any_edge(j_verta, j_vertc)
    if not cell3 in eptm.junctions.adjacent_cells[j_edgebe]:
        #Switch d and e
        j_verte, j_vertd = j_vertd, j_verte 
        j_edgebe = eptm.any_edge(j_vertb, j_verte)
    cell2 = eptm.junctions.adjacent_cells[j_edgebe][1]
    if verbose : print(''' adjacent cells edge be : %s, %s
                       ''' % (
                           str(eptm.junctions.adjacent_cells[j_edgebe][0]),
                           str(eptm.junctions.adjacent_cells[j_edgebe][1])))
    if cell2 == cell3:
        cell2 = eptm.junctions.adjacent_cells[j_edgebe][0]
    cell4 = eptm.junctions.adjacent_cells[j_edgeac][1]
    if cell4 == cell1:
        cell4 = eptm.junctions.adjacent_cells[j_edgeac][0]

    modified_cells = [cell1, cell2, cell3, cell4]
    modified_jverts = [j_verta, j_vertb, j_vertc,
                       j_vertd, j_verte, j_vertf]
    if eptm.__verbose__ : 
        for cell, i in zip(modified_cells, [1, 2, 3, 4]):
            print 'cell %i: %s' %(i, str(cell))
        for jv, i in zip(modified_jverts, 'abcdef'):
            print 'junction vertice %s: %s' %(i, str(jv))

    eptm.remove_junction(j_verta, j_vertb, cell1, cell3)
    eptm.remove_junction(j_verta, j_vertc, cell1, cell4)
    eptm.remove_junction(j_vertb, j_verte, cell2, cell3)
    eptm.add_junction(j_verta, j_vertb, cell2, cell4)
    eptm.add_junction(j_verta, j_verte, cell2, cell3)
    eptm.add_junction(j_vertb, j_vertc, cell1, cell4)

    eptm.update_rhotheta()
    eptm.update_dsigmas()
    sigma_a = eptm.sigmas[j_verta]
    sigma_b = eptm.sigmas[j_vertb]
    theta_a = eptm.thetas[j_verta]
    theta_b = eptm.thetas[j_vertb]

    zed_a = eptm.zeds[j_verta]
    zed_b = eptm.zeds[j_vertb]
    rho_a = eptm.rhos[j_verta]
    rho_b = eptm.rhos[j_vertb]
    period = rho_a * 2 * np.pi
    delta_t = theta_b - theta_a
    if delta_t >  np.pi:
        delta_t -= 2 * np.pi
        sigma_a += 2 * np.pi * rho_a
        print 'flip'
        flip = -1
    elif delta_t < -np.pi:
        delta_t += 2 * np.pi
        sigma_b += 2 * np.pi * rho_a
        print 'flip'
        flip = -1
    else:
        flip = 1

    center_sigma = (sigma_a + sigma_b) / 2
    delta_s = delta_t * rho_a
    center_zed = (zed_a + zed_b)/2.
        
    delta_z = np.abs(zed_b - zed_a)
    if eptm.sigmas[cell1] < eptm.sigmas[cell3]:
        eptm.sigmas[j_verta] = center_sigma + flip * delta_z/2.
        eptm.sigmas[j_vertb] = center_sigma - flip * delta_z/2.
    else:
        eptm.sigmas[j_vertb] = center_sigma + flip * delta_z/2.
        eptm.sigmas[j_verta] = center_sigma - flip * delta_z/2.
    if eptm.zeds[cell1] < eptm.zeds[cell3]:
        eptm.zeds[j_verta] = center_zed + delta_s/2.
        eptm.zeds[j_vertb] = center_zed - delta_s/2.
    else:
        eptm.zeds[j_vertb] = center_zed + delta_s/2.
        eptm.zeds[j_verta] = center_zed - delta_s/2.
        
    eptm.set_local_mask(cell1)
    eptm.set_local_mask(cell3)
    eptm.update_xy()
    eptm.reset_topology()
    return modified_cells, modified_jverts


@snapshot
def cell_division(eptm, mother_cell,
                  phi_division=None,
                  verbose=False):
    tau = 2 * np.pi
    if phi_division is None:
        phi_division = np.random.random() * tau
    eptm.update_rhotheta()
    eptm.update_dsigmas()
    a0 = eptm.params['prefered_area']
    h = eptm.params['prefered_height']
    v0 = a0 * (eptm.rhos[mother_cell] - eptm.params['rho_lumen'])
    eptm.cells.prefered_vol[mother_cell] = v0    
    daughter_cell = eptm.new_vertex(mother_cell)
    eptm.is_cell_vert[daughter_cell] = 1
    eptm.cells.ages[mother_cell] += 1
    eptm.cells.ages[daughter_cell] = 0
    eptm.cells.junctions[daughter_cell] = []
    
    print "Cell %s is born" % str(daughter_cell)
    
    junction_trash = []
    new_junctions = []
    new_jvs = []
    for j_edge in eptm.cells.junctions[mother_cell]:
        if j_edge is None:
            continue
        j_src, j_trgt = j_edge
    
        sigma_src = eptm.dsigmas[eptm.graph.edge(mother_cell, j_src)]
        zed_src = eptm.dzeds[eptm.graph.edge(mother_cell, j_src)]
        sigma_trgt = eptm.dsigmas[eptm.graph.edge(mother_cell, j_trgt)]
        zed_trgt = eptm.dzeds[eptm.graph.edge(mother_cell, j_trgt)]
        
        phi_trgt = np.arctan2(sigma_trgt, zed_trgt) + tau/2
        phi_trgt = (phi_trgt - phi_division) % tau
        phi_src = np.arctan2(sigma_src, zed_src) + tau/2
        phi_src = (phi_src - phi_division) % tau
        
        ## edge is on the mother side
        if phi_src > tau/2 and phi_trgt > tau/2:
            continue
        
        ## edge is on the daughter side
        elif phi_src <= tau/2 and phi_trgt <= tau/2:

            cell0, cell1 = eptm.junctions.adjacent_cells[j_edge]
            junction_trash.append((j_src, j_trgt, cell0, cell1))

            adj_cell = cell1 if cell0 == mother_cell else cell0
            new_junctions.append((j_src, j_trgt, adj_cell, daughter_cell))
        
        ## edge is cut by the division    
        elif ((phi_src > tau/2 and phi_trgt <= tau/2)
              or ( phi_src <= tau/2 and phi_trgt > tau/2 )) :
            
            cell0, cell1 = eptm.junctions.adjacent_cells[j_edge]
            adj_cell = cell1 if cell0 == mother_cell else cell0
            new_jv = eptm.new_vertex(j_src)
            new_jvs.append(new_jv)
            sigma_n = - ((sigma_src * zed_trgt
                          - zed_src * sigma_trgt)
                         / (zed_src - zed_trgt
                            + (sigma_trgt - sigma_src
                           )/ np.tan(phi_division)))
            zed_n = sigma_n / np.tan(phi_division)
            ## The midle of the segment is closer to the final optimum
            # sigma_n = (sigma_src + sigma_trgt) / 2.
            # zed_n = (zed_src + zed_trgt) / 2.
            
            eptm.rhos[new_jv] = (eptm.rhos[j_src] +
                                 eptm.rhos[j_trgt]) / 2.
            eptm.zeds[new_jv] = zed_n + eptm.zeds[mother_cell]
            eptm.sigmas[new_jv] = sigma_n + eptm.sigmas[mother_cell]

            # Periodic Boundary conditions
            if eptm.sigmas[new_jv] >= tau * eptm.rhos[new_jv]:
                eptm.sigmas[new_jv] -= tau * eptm.rhos[new_jv]
            elif eptm.sigmas[new_jv] < 0:
                eptm.sigmas[new_jv] += tau * eptm.rhos[new_jv]
            eptm.thetas[new_jv] = eptm.sigmas[new_jv] / eptm.rhos[new_jv]
            junction_trash.append((j_src, j_trgt, cell0, cell1))
            if phi_src <= tau/2:
                new_junctions.append((new_jv, j_trgt,
                                      adj_cell, mother_cell))
                new_junctions.append((new_jv, j_src,
                                      adj_cell, daughter_cell))
            else:
                new_junctions.append((new_jv, j_src,
                                      adj_cell, mother_cell))
                new_junctions.append((new_jv, j_trgt,
                                      adj_cell, daughter_cell))
    if not len(new_jvs) == 2:
        print('Problem in the division of cell %s'
              % str(mother_cell))
        eptm.is_alive[daughter_cell] = 0
        return
    for (j_src, j_trgt, cell0, cell1) in junction_trash:
        eptm.remove_junction(j_src, j_trgt, cell0, cell1)
    for (j_src, j_trgt, cell0, cell1) in new_junctions:
        j = eptm.add_junction(j_src, j_trgt, cell0, cell1)
    # Cytokinesis
    septum = eptm.add_junction(new_jvs[0], new_jvs[1],
                               mother_cell, daughter_cell)
    eptm.set_local_mask(daughter_cell)
    eptm.update_xy()
    
    # eptm.graph.set_vertex_filter(eptm.is_local_vert)
    # eptm.graph.set_edge_filter(eptm.is_local_edge)
    eptm.reset_topology()
    # eptm.graph.set_vertex_filter(None)
    # eptm.graph.set_edge_filter(None)

    if eptm.__verbose__: print 'Division completed'
    return septum

@snapshot
def type3_transition(eptm, cell, reduce_edgenum=True, verbose=False):
    """
    
    That's when a three faced cell disappears.
    """
    eptm.graph.set_vertex_filter(None)
    eptm.graph.set_edge_filter(None)
    j_edges = eptm.cells.junctions[cell]
    edge_lengths = np.array([eptm.edge_lengths[edge]
                             for edge in j_edges])
    if len(j_edges) != 3 and reduce_edgenum:
        if verbose:
            print ('''%i edges left''' % len(j_edges))
        je = j_edges[edge_lengths.argmin()]
        cell1, cell3 = eptm.junctions.adjacent_cells[je]
        modified = type1_transition(eptm, (cell1, cell3),
                                    verbose=verbose)
        return 
    old_jvs = [old_jv for old_jv in cell.out_neighbours()]
    new_jv = eptm.new_vertex(old_jvs[0])
    if verbose: print ('Cell %s removed, edge %s created'
                       % (cell, new_jv))
    edge_trash = []
    cell_neighbs = []

    for old_jv in old_jvs:
        for edge in old_jv.out_edges():
            if edge in j_edges: continue
            new_edge = eptm.new_j_edge(new_jv, edge.target())
            if verbose: print 'new_j_edge %s ' %edge 
        for edge in old_jv.in_edges():
            if eptm.is_ctoj_edge[edge]:
                cell0 = edge.source()
                eptm.new_ctoj_edge(cell0, new_jv)
                if verbose: print 'new_ctoj_edge %s ' %edge
                if cell0 not in cell_neighbs:
                    cell_neighbs.append(cell0)
            elif edge in j_edges: continue
            else:
                if verbose: print 'new_j_edge %s ' %edge 
                eptm.new_j_edge(edge.source(), new_jv)
        edge_trash.extend(old_jv.all_edges())
        eptm.is_alive[old_jv] = 0
    
    for edge in edge_trash:
        try:
            eptm.graph.remove_edge(edge)
        except ValueError:
            print 'invalid edge'
            continue
    eptm.set_local_mask(None)
    for n_cell in cell_neighbs:
        eptm.set_local_mask(n_cell)
    eptm.is_alive[cell] = 0
    eptm.reset_topology()
    return new_jv

def remove_cell(eptm, cell):

    if isinstance(cell, int):
        cell = eptm.graph.vertex(cell)
    
    eptm.set_local_mask(None)
    eptm.graph.set_vertex_filter(None)
    eptm.graph.set_edge_filter(None)
    vertex_trash = []
    new_ctojs = []
    new_jes =[]
    ctojs = [ctoj for ctoj in cell.out_edges()]
    cell_jes = eptm.cells.junctions[cell]
    jvs = [jv for jv in cell.out_neighbours()]
    edge_trash = ctojs
    edge_trash.extend(cell_jes)
    new_jv = eptm.new_vertex(jvs[0])
    print 'new vertex %s' % str(new_jv)
    eptm.is_local_vert[jv] = 1
    eptm.ixs[new_jv] = eptm.ixs[cell]
    eptm.wys[new_jv] = eptm.wys[cell]
    eptm.zeds[new_jv] = eptm.zeds[cell]
    adjacent_cells = []
    for jv in jvs:
        vertex_trash.append(jv)
        for edge in jv.all_edges():
            if edge in edge_trash:
                continue
            edge_trash.append(edge)
            if eptm.is_ctoj_edge[edge]:
                adj_cell = edge.source()
                adjacent_cells.append(adj_cell)
                new_ctojs.append((adj_cell, new_jv))
            elif eptm.is_junction_edge[edge]:
                jv0, jv1 = edge
                opposite = jv0 if jv1 == jv else jv1
                new_jes.append((opposite, new_jv))
        
    for neighb_cell, jv in new_ctojs:
        ctoj = eptm.new_edge(neighb_cell, jv, ctojs[0])
    for jv0, jv1 in new_jes:
        je = eptm.new_edge(jv0, jv1, cell_jes[0])
        
    eptm.is_alive[cell] = 0
    eptm.is_cell_vert[cell] = 0
    for v in vertex_trash:
        eptm.is_alive[v] = 0
        eptm.is_cell_vert[v] = 0
    for e in edge_trash:
        try:
            eptm.graph.remove_edge(e)
        except ValueError:
            print('edge already destroyed')
        
    eptm.reset_topology()
    eptm.update_geometry()
    
def resolve_small_edges(eptm, threshold=5e-2, vfilt=None, efilt=None):
    # Collapse 3 sided cells
    if vfilt == None:
        vfilt_3sides = eptm.is_cell_vert.copy()
    else:
        vfilt_3sides = vfilt.copy()
    eptm.graph.set_edge_filter(eptm.is_ctoj_edge,
                               inverted=True)
    degree_pm =  eptm.graph.degree_property_map('out').a
    vfilt_3sides.a *= [degree_pm == 3][0] * eptm.is_alive.a
    eptm.graph.set_vertex_filter(vfilt_3sides)
    cells = [cell for cell in eptm.graph.vertices()]
    if eptm.__verbose__:
        print(''' There are %i three sided cells
              ''' % len(cells))

    eptm.graph.set_vertex_filter(None)
    eptm.graph.set_edge_filter(None)
    new_jvs = [eptm.type3_transition(cell, threshold)
               for cell in cells]
    eptm.reset_topology()
    # Type 1 transitions
    if efilt == None:
        efilt_je = eptm.is_junction_edge.copy()
    else:
        efilt_je = efilt.copy()
        efilt_je.a *= eptm.is_junction_edge.a
    efilt_je.a *= [eptm.edge_lengths.a < threshold][0]
    eptm.graph.set_edge_filter(efilt_je)
    visited_cells = []
    if eptm.__verbose__:
        print('%i  type 1 transitions are going to take place'
              % efilt_je.a.sum())
    short_edges = [e for e in eptm.graph.edges()]
    eptm.graph.set_edge_filter(None)
    eptm.graph.set_edge_filter(None)
    for edge in short_edges:
        eptm.set_local_mask(None)
        cells = eptm.junctions.adjacent_cells[edge]
        if len(cells) != 2:
            continue
        else:
            cell0, cell1 = cells
        if cell0 in visited_cells or cell1 in visited_cells:
            continue
        visited_cells.extend([cell0, cell1])
        if (len(eptm.cells.junctions[cell0]) < 4) or (
            len(eptm.cells.junctions[cell1]) < 4):
            continue
        print 'Type 1 transition'
        energy0 = eptm.calc_energy()
        backup_graph = eptm.graph.copy()
        modified = type1_transition(eptm, (cell0, cell1))
        if modified is not None:
            pos0, pos1 = eptm.find_energy_min()
            energy1 = eptm.calc_energy()
            if energy0 < energy1:
                print 'Undo transition!'
                eptm.graph = backup_graph
        eptm.graph.set_edge_filter(None)
