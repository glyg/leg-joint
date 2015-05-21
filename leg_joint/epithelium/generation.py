# -*- coding: utf-8 -*-
'''This module contains the elements necessary to build an epithelium.

`vertex_data`, `cell_data`, `edge_data`, `junction_data` and
`face_data` are dictionnaries of the form:

```python
    {'data_name': (default_value, data_type)}.
```

They are used to instanciate all the necessary property maps over the graph.


`vertex_data` and `edge_data` contain data unspecific to the vertex or
edge type (here cell vertices, junction edges, and triangular
faces). Specific data for cells and junction edges will be instanciated as
PropertyMaps of the corresponding GraphView, except for faces.
Note that graphviews inherits the parent graphs propertymaps
**at instanciation time**.


It is recommanded to add new properties by specifying them here,
although they can also be specified when `Epithelium` is instanciated.
Adding properties dynamically is possible but can lead to inconsitencies
between graphviews and the parent graph's propery maps.

'''

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd
import graph_tool.all as gt


import logging
log = logging.getLogger(__name__)




vertex_data = {
    ## Coordinates
    'x': (0., 'float'),
    'y': (0., 'float'),
    'z': (0., 'float'),
    'rho': (0., 'float'),
    'theta': (0., 'float'),
    'height': (0., 'float'),
    ## Topology
    'is_cell_vert': (0, 'bool'),
    'is_alive': (1, 'bool'),
    'is_active_vert': (1, 'bool')}

### Cell vertices are at the apical surface center of mass
cell_data = {
    ## Cell Geometry
    'perimeter': (0., 'float'),
    'area': (0., 'float'),
    'vol': (0., 'float'),
    ## Dynamical
    'contractility': (0.014, 'float'),
    'vol_elasticity': (0.014, 'float'),
    ## Topology
    'num_sides': (1, 'int')}

edge_data = {
    ## Coordinates
    'dx': (0., 'float'),
    'dy': (0., 'float'),
    'dz': (0., 'float'),
    'edge_length': (0., 'float'),
    ## Gradients
    'gx': (0., 'float'),
    'gy': (0., 'float'),
    'gz': (0., 'float'),
    ## Topology
    'is_junction_edge': (0, 'bool')}

junction_data = {
    ## Dynamical parameters
    'line_tension': (0., 'float'),
    'radial_tension': (0., 'float')}


face_data = {
    ## Normal Coordinates
    'ux': (0., 'float'),
    'uy': (0., 'float'),
    'uz': (0., 'float'),
    ## Geometry
    'sub_area': (0., 'float'),
    'ell_ij': (0., 'float')}



def base_grid(n_cells_x, n_cells_y,
              delta_x=1, delta_y=1):
    """
    Creates a 2D hexagonal grid with the following geometry:

               j       j       j       j
           c       c       c       c
               j       j       j       j
           j       j       j       j
               c       c       c       c
           j       j       j       j
               j       j       j

    Parameters
    ----------

    n_cells_x : int
        Number of cells along the first dimention
        (i.e the number of cell vertices on an axis
         parallel to the first dimention)

    n_cells_y : int
        Number of cells along the second dimention
        (i.e the number of cell vertices on an axis
         parallel to the 2nd dimention)

    delta_x : float, optional
        Spacing between the verticies along first axis,
        default 1
    delta_y : float, optional
        Spacing between the verticies along 2nd axis
        default 1

    Note
    ----

    the total number of vertices N_v is given by  (N_v = n_cells_x * n_cells_y * 3)
    The number of _cell_ verices is n_cells_x * n_cells_y.

    Returns
    -------

    pos: np.ndarray
        Two dimentional array with shape `(N_v, 2)` with the vertex positions
    is_cell_vert: np.ndarray
        Array with shape `(N_v,)` with the vertex labels
        (1 if it is a cell vertex, 0 if it is not)

    """

    n_x = n_cells_x
    n_y = n_cells_y * 3

    xy_grid = np.mgrid[:n_x, :n_y]

    xs = xy_grid[0].astype('float')
    ys = xy_grid[1].astype('float')

    ys[::2, ...] += 0.5
    is_cell_vert = np.zeros_like(xy_grid[0]).astype(np.bool)
    is_cell_vert[::2, ::3] = 1
    is_cell_vert[1::2, 2::3] = 1
    is_cell_vert = is_cell_vert.flatten()

    pos = np.zeros((n_x * n_y, 2))
    pos[:, 0] = xs.flatten() * delta_x
    pos[:, 1] = ys.flatten() * delta_y

    return pos, is_cell_vert


def cylindrical(n_cells_length,
                n_cells_circum,
                l_0, h_0):

    ### Compute the cylinder radius from the number of vertices
    rho_c = n_cells_circum * 3 * l_0 / (2 * np.pi)
    ### Compute the lumen radius from the prefered height
    rho_lumen = rho_c - h_0

    delta_theta = 2 * np.pi / (n_cells_circum * 3)
    delta_z = delta_theta * rho_c * np.sqrt(3)/2

    ztheta_pos, is_cell_vert = base_grid(n_cells_length,
                                         n_cells_circum,
                                         delta_z, delta_theta)
    ztheta_pos[:, 0] -= ztheta_pos[:, 0].max()/2

    ### Pass to cartesian
    xyz_pos = np.zeros((is_cell_vert.size, 3))
    xyz_pos[:, 0] = rho_c * np.cos(ztheta_pos[:, 1])
    xyz_pos[:, 1] = rho_c * np.sin(ztheta_pos[:, 1])
    xyz_pos[:, 2] = ztheta_pos[:, 0]

    graph, pos_vp = gt.geometric_graph(xyz_pos,
                                       delta_z*1.5)

    graph.set_directed(True)
    graph.set_fast_edge_removal(True)

    for prop in vertex_data:
        default, dtype = vertex_data[prop]
        graph.vp[prop] = graph.new_vertex_property(dtype)
        graph.vp[prop].a = default

    ### set vertex type mask
    graph.vp['is_cell_vert'].a = is_cell_vert
    ### Set coordinates
    for i, coord in enumerate('xyz'):
        graph.vp[coord].a = xyz_pos[:, i]

    graph.vp['theta'].a = ztheta_pos[:, 1]

    for prop in edge_data:
        default, dtype = edge_data[prop]
        graph.ep[prop] = graph.new_edge_property(dtype)
        graph.ep[prop].a = default


    ### Set edge type mask
    graph.ep['is_junction_edge'].a = 1
    reorient_edges(graph,
                   graph.vp['is_cell_vert'],
                   graph.ep['is_junction_edge'])
    return graph


def reorient_edges(graph, is_cell_vert, is_junction_edge):
    '''Reorient the graph such that cell to junction edges
    are always from cell to junction. Modifies `graph` and
    `is_junction_edge` inplace

    '''
    to_flip = []
    for edge in graph.edges():
        srce, trgt = edge
        if is_cell_vert[srce] and not is_cell_vert[trgt]:
            is_junction_edge[edge] = 0
        elif not is_cell_vert[srce] and is_cell_vert[trgt]:
            ### Flip the edge
            to_flip.append((srce, trgt))
        elif is_cell_vert[srce] and is_cell_vert[trgt]:
            raise ValueError(
                'Invalid cell to cell edge {}'.format(edge))
        else:
            is_junction_edge[edge] = 1
    log.info('filpped {} edges'.format(len(to_flip)))
    ### Change edges outside of the loop, else bad things
    ### occur - e.g. vicious segfaults
    for (srce, trgt) in to_flip:
        new = graph.add_edge(trgt, srce)
        is_junction_edge[new] = 0
        graph.remove_edge(graph.edge(srce, trgt))
