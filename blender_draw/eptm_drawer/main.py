import sys
sys.path.append('/usr/lib/python3/dist-packages/')

import os
import random

import bpy
import bmesh
import numpy as np


from eptm_drawer import utils
from eptm_drawer import objects
from eptm_drawer import graph_io


def main():

    source_inplace = os.path.join(os.path.dirname(bpy.data.filepath),
                                  "epithelium.xml")
    fname = os.path.join(source_inplace)

    scene = utils.clear_scene()

    #nucleus = objects.set_nucleus()

    graph = graph_io.import_graph(fname)
    is_cell = graph.vertex_properties['is_cell_vert']
    is_alive = graph.vertex_properties['is_alive']
    is_junction = graph.edge_properties['is_junction_edge']
    ixs = graph.vertex_properties['ixs']
    wys = graph.vertex_properties['wys']
    zeds = graph.vertex_properties['zeds']

    cell_color = '#2887c8'
    edge_color = '#3dc828'
    
    for cell in graph.vertices():
        if not is_cell[cell]: continue
        if not is_alive[cell]: continue
        name = 'cell'+str(cell)
        c_x, c_y, c_z =  ixs[cell], wys[cell], zeds[cell]
        j_xx = np.array([ixs[je] for je in cell.out_neighbours()])
        j_yy = np.array([wys[je] for je in cell.out_neighbours()])
        j_zz = np.array([zeds[je] for je in cell.out_neighbours()])
        rel_xx = j_xx - c_x
        rel_yy = j_yy - c_y
        rel_zz = j_zz - c_z
        rel_ss = np.arctan2(rel_yy, rel_xx) * np.hypot(rel_yy, rel_xx)
        phis = np.arctan2(rel_zz, rel_ss)
        indices = np.argsort(phis)
        rel_xx = rel_xx[indices]
        rel_yy = rel_yy[indices]
        rel_zz = rel_zz[indices]
        cell_obj = objects.set_cell(name, c_x, c_y, c_z,
                                    rel_xx, rel_yy, rel_zz, cell_color)
    for je in graph.edges():
        if not is_junction[je]: continue
        v0, v1 = je
        name = 'je'+str(v0)+'to'+str(v1)
        je_obj = objects.set_junction(name,
                                      (ixs[v0], ixs[v1]),
                                      (wys[v0], wys[v1]),
                                      (zeds[v0], zeds[v1]), edge_color)

    scene.frame_current = 0

    # Smooth objects view
    utils.select_all()
    bpy.ops.object.shade_smooth()

    # Add a camera
    bpy.ops.object.camera_add(view_align=True, enter_editmode=False)
    camera = bpy.context.object
    camera.name = "camera"

    scene.objects.active = None
