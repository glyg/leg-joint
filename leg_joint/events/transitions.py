# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

import graph_tool.all as gt


def find_rosettes(eptm):
    eptm.graph.set_vertex_filter(eptm.is_cell_vert,
                                 inverted=True)
    total = eptm.graph.degree_property_map('total')
    eptm.graph.set_vertex_filter(None)
    return gt.find_vertex_range(eptm.graph, total, (4, 20))

def solve_rosette(eptm, central_vert, tension_increase=1.):
    cells = [cell for cell in central_vert.in_neighbours()
             if eptm.is_cell_vert[cell]]
    if len(cells) == 3:
        eptm.log.info('Rosette done')
        return None
    eptm.set_local_mask(None)
    j_edges = [je for je in central_vert.all_edges()
               if eptm.is_junction_edge[je]]
    j_verts = [je.source() if je.source() != central_vert else je.target()
               for je in j_edges]
    #log.info(len(j_verts))
    rel_thetas = np.array([eptm.thetas[jv] - eptm.thetas[central_vert]
                           for jv in j_verts])
    rel_thetas[rel_thetas > np.pi] -= 2 * np.pi
    rel_thetas[rel_thetas < -np.pi] += 2 * np.pi
    upper_jvs = [jv for jv in j_verts
                 if rel_thetas[j_verts.index(jv)] > 0]
    lower_jvs = [jv for jv in j_verts
                 if rel_thetas[j_verts.index(jv)] < 0]
    split_cells = []
    for cell in cells:
        jvs = set(jv for jv in cell.out_neighbours())
        upper = set(upper_jvs)
        lower = set(lower_jvs)
        if len(jvs & upper) and len(jvs & lower):
            split_cells.append(cell)
    if not len(split_cells) == 2:
        raise ValueError()
    new_jv = eptm.new_vertex(central_vert)
    eptm.zeds[new_jv] += 0.1
    eptm.zeds[central_vert] -= 0.1
    to_remove = [(central_vert, jv) +
                  tuple(eptm.junctions.adjacent_cells[
                      eptm.any_edge(central_vert, jv)])
                  for jv in upper_jvs]
    to_add = [(new_jv, jv) +
              tuple(eptm.junctions.adjacent_cells[
                  eptm.any_edge(central_vert, jv)])
              for jv in upper_jvs]
    to_add.append((central_vert, new_jv) + tuple(split_cells))
    for junc in to_remove:
        eptm.remove_junction(*junc)
    eptm.set_local_mask(None)
    for cell in cells:
        eptm.set_local_mask(cell, wider=True)
    eptm.reset_topology(local=True)
    return new_jv

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

    Parameters
    ----------

    eptm:  ::class:`Epithelium` instance

    elements: graph edge or vertex:
        Can be either:

        * two cell vertices (1 and 3),
        * two junction vertices (a and b)
        * or a single edge (between a and b)

    verbose: bool, default `False`
        if True, prints informations

    """
    #### Parsing arguments
    try:
        # Cells
        if len(elements) == 2 and eptm.is_cell_vert[elements[0]]:
            cell1 = elements[0]
            cell3 = elements[1]
            j_edges1 =  eptm.cells.junctions[cell1]
            j_edges3 =  eptm.cells.junctions[cell3]
            try:
                j_edgeab = [je for je in j_edges1 if je in j_edges3][0]
            except IndexError:
                eptm.log.error("No valid junction found "
                               "beetween cells %s and %s " % (cell1, cell3))
                return
            j_verta = j_edgeab.source()
            j_vertb = j_edgeab.target()
        # Junction vertices
        elif len(elements) == 2 and not eptm.is_cell_vert[elements[0]]:
            j_verta, j_vertb = elements
            j_edgeab = eptm.graph.edge(elements)
            if j_edgeab is None:
                eptm.log.error("Invalid junction %s" % str(j_edgeab))
                return
            try:
                cell1, cell3 = eptm.junctions.adjacent_cells[j_edgeab]
            except ValueError:
                eptm.log.error("No adgacent cells found"
                               "for junction %s" % str(j_edgeab))
                return
        else:
            raise ValueError("Invalid argument %s" % str(elements))

    # Junction edges
    except TypeError:
        if eptm.is_junction_edge[elements]:
            j_edgeab = elements
            j_verta, j_vertb = j_edgeab.source(), j_edgeab.target()
            try:
                cell1, cell3 = eptm.junctions.adjacent_cells[j_edgeab]
            except ValueError:
                eptm.log.error("No adgacent cells found"
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
        eptm.log.error('''
                       Valid only for 3-way junctions
                       For type 1 transition between cells
                       %s and %s
                       ''' %(str(cell1), str(cell3)))
        eptm.set_local_mask(cell1)
        eptm.set_local_mask(cell3)
        return None

    j_edgeac = eptm.any_edge(j_verta, j_vertc)
    j_edgebe = eptm.any_edge(j_vertb, j_verte)
    if j_edgebe is None or j_edgeac is None:
        raise ValueError("Invalid geometry")

    if not cell1 in eptm.junctions.adjacent_cells[j_edgeac]:
        #Switch f and c
        j_vertc, j_vertf = j_vertf, j_vertc
        j_edgeac = eptm.any_edge(j_verta, j_vertc)
    if not cell3 in eptm.junctions.adjacent_cells[j_edgebe]:
        #Switch d and e
        j_verte, j_vertd = j_vertd, j_verte
        j_edgebe = eptm.any_edge(j_vertb, j_verte)
    cell2 = eptm.junctions.adjacent_cells[j_edgebe][1]
    eptm.log.debug(''' adjacent cells edge be : %s, %s
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
    for cell, i in zip(modified_cells, [1, 2, 3, 4]):
        eptm.log.debug('cell %i: %s' %(i, str(cell)))
    for jv, i in zip(modified_jverts, 'abcdef'):
        eptm.log.debug('junction vertice %s: %s' %(i, str(jv)))

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
    delta_t = theta_b - theta_a
    if delta_t >  np.pi:
        delta_t -= 2 * np.pi
        sigma_a += 2 * np.pi * rho_a
        flip = -1
    elif delta_t < -np.pi:
        delta_t += 2 * np.pi
        sigma_b += 2 * np.pi * rho_a
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
    eptm.reset_topology(local=True)
    return modified_cells, modified_jverts

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
        eptm.log.debug('''%i edges left''' % len(j_edges))
        je = j_edges[edge_lengths.argmin()]
        cell1, cell3 = eptm.junctions.adjacent_cells[je]
        modified = type1_transition(eptm, (cell1, cell3),
                                    verbose=verbose)
        return
    old_jvs = [old_jv for old_jv in cell.out_neighbours()]
    new_jv = eptm.new_vertex(old_jvs[0])
    eptm.log.debug('Cell %s removed, edge %s created'
                   % (cell, new_jv))
    edge_trash = []
    cell_neighbs = []

    for old_jv in old_jvs:
        for edge in old_jv.out_edges():
            if edge in j_edges: continue
            new_edge = eptm.new_j_edge(new_jv, edge.target())
            eptm.log.debug('new_j_edge %s ' %edge)
        for edge in old_jv.in_edges():
            if eptm.is_ctoj_edge[edge]:
                cell0 = edge.source()
                eptm.new_ctoj_edge(cell0, new_jv)
                eptm.log.debug('new_ctoj_edge %s ' %edge)
                if cell0 not in cell_neighbs:
                    cell_neighbs.append(cell0)
            elif edge in j_edges: continue
            else:
                eptm.log.debug('new_j_edge %s ' %edge)
                eptm.new_j_edge(edge.source(), new_jv)
        edge_trash.extend(old_jv.all_edges())
        eptm.is_alive[old_jv] = 0

    for edge in edge_trash:
        try:
            eptm.graph.remove_edge(edge)
        except ValueError:
            eptm.log.error('invalid edge')
            continue
    eptm.set_local_mask(None)
    for n_cell in cell_neighbs:
        eptm.set_local_mask(n_cell)
    eptm.is_alive[cell] = 0
    eptm.reset_topology()
    return new_jv
