# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np

from ..io import hdf_snapshot


@hdf_snapshot
#@png_snapshot
def cell_division(eptm, mother_cell,
                  phi_division=None,
                  verbose=False):
    '''
    Devides a cell
    '''

    tau = 2 * np.pi
    if phi_division is None:
        phi_division = np.random.random() * tau
    eptm.update_rhotheta()
    eptm.update_dsigmas()
    a0 = eptm.params['prefered_area']
    v0 = a0 * (eptm.rhos[mother_cell] - eptm.rho_lumen)
    eptm.cells.prefered_vol[mother_cell] = v0
    daughter_cell = eptm.new_vertex(mother_cell)
    eptm.is_cell_vert[daughter_cell] = 1
    eptm.cells.ages[mother_cell] += 1
    eptm.cells.ages[daughter_cell] = 0
    eptm.cells.junctions[daughter_cell] = []

    eptm.log.info("Cell {} is born".format(str(daughter_cell)))

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
        eptm.log.error('Problem in the division of cell {}'.format(
            str(mother_cell)))
        eptm.is_alive[daughter_cell] = 0
        return
    for (j_src, j_trgt, cell0, cell1) in junction_trash:
        eptm.remove_junction(j_src, j_trgt, cell0, cell1)
    for (j_src, j_trgt, cell0, cell1) in new_junctions:
        ### Strange behaviour of gt here
        eptm.graph.set_edge_filter(None)
        eptm.graph.set_vertex_filter(None)
        j = eptm.add_junction(j_src, j_trgt, cell0, cell1)
    # Cytokinesis
    septum = eptm.add_junction(new_jvs[0], new_jvs[1],
                               mother_cell, daughter_cell)
    eptm.set_local_mask(mother_cell, wider=True)
    eptm.set_local_mask(daughter_cell, wider=True)
    eptm.update_xy()

    # eptm.graph.set_vertex_filter(eptm.is_local_vert)
    # eptm.graph.set_edge_filter(eptm.is_local_edge)
    eptm.reset_topology()
    # eptm.graph.set_vertex_filter(None)
    # eptm.graph.set_edge_filter(None)

    eptm.log.info('Division completed')
    return septum
