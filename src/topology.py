#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import filters



def type1_transition(eptm, elements):
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
    if len(elements) == 2 and eptm.is_cell_vert[elements[0]]:
        cell1 = elements[0]
        cell3 = elements[1]
        j_edges1 =  eptm.cell_junctions(cell1)
        j_edges3 =  eptm.cell_junctions(cell3)
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
            cell1, cell3 = eptm.adjacent_cells(j_edgeab)
        except ValueError:
            print ("No adgacent cells found"
                   "for junction %s" % str(j_edgeab))
            return
    # Junction edges
    elif eptm.is_junction_edge(elements):
        j_edgeab = elements
        j_verta, j_vertb = j_edgeab.source(), j_edgeab.target()
        try:
            cell1, cell3 = eptm.adjacent_cells(j_edgeab)
        except ValueError:
            print ("No adgacent cells found"
                   "for junction %s" % str(j_edgeab))
            return
    else:
        raise ValueError("Invalid argument %s" % str(elements))
    try:
        vecinos_a = [jv for jv in eptm.ordered_neighbours(j_verta)
                     if not eptm.is_cell_vert[jv]]
        vecinos_b = [jv for jv in eptm.ordered_neighbours(j_vertb)
                     if not eptm.is_cell_vert[jv]]
        j_vertf, j_vertc = [jv for jv in vecinos_a if jv != j_vertb]
        j_verte, j_vertd = [jv for jv in vecinos_b if jv != j_verta]
    except ValueError:
        print "Valid only for 3-way junctions"
        return
    j_edgeac = eptm.any_edge(j_verta, j_vertc)
    j_edgebe = eptm.any_edge(j_vertb, j_verte)
    if j_edgebe is None or j_edgeac is None:
        print "Invalid geometry"
        return
    if not cell1 in eptm.adjacent_cells(j_edgeac):
        #Switch f and c
        j_vertc, j_vertf = j_vertf, j_vertc 
        j_edgeac = eptm.any_edge(j_verta, j_vertc)
    if not cell3 in eptm.adjacent_cells(j_edgebe):
        #Switch d and e
        j_verte, j_vertd = j_vertd, j_verte 
        j_edgebe = eptm.any_edge(j_vertb, j_verte)
    cell2 = eptm.adjacent_cells(j_edgebe)[1]
    print "adjacent cells edge be : %s, %s" % (
        str(eptm.adjacent_cells(j_edgebe)[0]),
        str(eptm.adjacent_cells(j_edgebe)[1]))
    if cell2 == cell3:
        cell2 = eptm.adjacent_cells(j_edgebe)[0]
    cell4 = eptm.adjacent_cells(j_edgeac)[1]
    if cell4 == cell1:
        cell4 = eptm.adjacent_cells(j_edgeac)[0]

    modified_cells = [cell1, cell2, cell3, cell4]
    modified_jverts = [j_verta, j_vertb, j_vertc,
                       j_vertd, j_verte, j_vertf]
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

    sigma_a = eptm.sigmas[j_verta]
    sigma_b = eptm.sigmas[j_vertb]
    zed_a = eptm.zeds[j_verta]
    zed_b = eptm.zeds[j_vertb]

    center_sigma = (sigma_a + sigma_b)/2.
    center_zed = (zed_a + zed_b)/2.

    delta_s = np.abs(sigma_b - sigma_a)
    delta_z = np.abs(zed_b - zed_a)
    if eptm.sigmas[cell1] < eptm.sigmas[cell3]:
        eptm.sigmas[j_verta] = center_sigma + delta_z/2.
        eptm.sigmas[j_vertb] = center_sigma - delta_z/2.
    else:
        eptm.sigmas[j_vertb] = center_sigma + delta_z/2.
        eptm.sigmas[j_verta] = center_sigma - delta_z/2.
    if eptm.zeds[cell1] < eptm.zeds[cell3]:
        eptm.zeds[j_verta] = center_zed + delta_s/2.
        eptm.zeds[j_vertb] = center_zed - delta_s/2.
    else:
        eptm.zeds[j_vertb] = center_zed + delta_s/2.
        eptm.zeds[j_verta] = center_zed - delta_s/2.

    eptm.set_local_mask(cell1)
    eptm.set_local_mask(cell3)

    eptm.update_apical_geom()
    return modified_cells, modified_jverts

def type3_transition(eptm, cell, threshold=0.5, verbose=True):
    '''
    That's when a three faced cell disappears
    '''
    eptm.graph.set_vertex_filter(None)
    eptm.graph.set_edge_filter(None)
    j_edges = eptm.cell_junctions(cell)
    edge_lengths = np.array([eptm.edge_lengths[edge]
                             for edge in j_edges])

    if len(j_edges) != 3:
        if verbose:
            print ('''%i edges left''' % len(j_edges))

            je = j_edges[edge_lengths.argmin()]
        cell1, cell3 = eptm.adjacent_cells(je)
        modified = type1_transition(eptm, (cell1, cell3))
        return 
    if edge_lengths.min() > threshold: return
    old_jvs = [old_jv for old_jv in cell.out_neighbours()]
    new_jv = eptm.new_vertex(old_jvs[0])
    edge_trash = []
    for old_jv in old_jvs:
        for edge in old_jv.out_edges():
            if edge in j_edges: continue
            new_edge = eptm.graph.add_edge(new_jv, edge.target())
            for prop in eptm.graph.edge_properties.values():
                prop[new_edge] = prop[edge]
        for edge in old_jv.in_edges():
            if edge in j_edges: continue
            eptm.graph.add_edge(edge.source(), new_jv)
            for prop in eptm.graph.edge_properties.values():
                prop[new_edge] = prop[edge]

        edge_trash.extend(old_jv.all_edges())
        eptm.is_alive[old_jv] = 0
    
    for edge in edge_trash:
        try:
            eptm.graph.remove_edge(edge)
        except ValueError:
            continue
    eptm.is_alive[cell] = 0
    return new_jv


def cell_division(eptm, mother_cell,
                  phi_division=None,
                  verbose=False):
    tau = 2 * np.pi
    if phi_division is None:
        phi_division = np.random.random() * tau
    daughter_cell = eptm.new_vertex(mother_cell)
    print "Cell %s is born" % str(daughter_cell)
    eptm.is_cell_vert[daughter_cell] = 1
    junction_trash = []
    new_junctions = []
    new_jvs = []
    for j_edge in eptm.cell_junctions(mother_cell):
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
        
        if phi_src > tau/2 and phi_trgt > tau/2:
            continue
        
        elif phi_src <= tau/2 and phi_trgt <= tau/2:
        
            cell0, cell1 = eptm.adjacent_cells(j_edge)
            junction_trash.append((j_src, j_trgt, cell0, cell1))

            adj_cell = cell1 if cell0 == mother_cell else cell0
            new_junctions.append((j_src, j_trgt, adj_cell, daughter_cell))

        elif ((phi_src > tau/2 and phi_trgt <= tau/2)
              or ( phi_src <= tau/2 and phi_trgt > tau/2 )) :

            cell0, cell1 = eptm.adjacent_cells(j_edge)
            adj_cell = cell1 if cell0 == mother_cell else cell0
            new_jv = eptm.new_vertex(j_src)
            new_jvs.append(new_jv)
            sigma_n = - ((sigma_src * zed_trgt
                          - zed_src * sigma_trgt)
                         / (zed_src - zed_trgt
                            + (sigma_trgt - sigma_src
                           )/ np.tan(phi_division)))
            zed_n = sigma_n / np.tan(phi_division)
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
        print 'problem'
        eptm.is_alive[daughter_cell] = 0
        print new_jvs
        return
    for (j_src, j_trgt, cell0, cell1) in junction_trash:
        eptm.remove_junction(j_src, j_trgt, cell0, cell1)
    for (j_src, j_trgt, cell0, cell1) in new_junctions:
        j = eptm.add_junction(j_src, j_trgt, cell0, cell1)
    # Cytokinesis
    j = eptm.add_junction(new_jvs[0], new_jvs[1],
                          mother_cell, daughter_cell)
    eptm.set_local_mask(daughter_cell)
    # Updates
    eptm.update_apical_geom()
    eptm.update_gradient()
    eptm.set_vertex_state()
    return j
