import sys
# sys.path.append('/home/guillaume/anaconda/envs/python3/lib/python3.4/site-packages/')



sys.path.append('/home/guillaume/Python/hdfgraph')
import os
import random

import bpy
import bmesh
import numpy as np


from . import utils
from . import graph_io
from . import objects

def out_neighbours(cell_idx, ctojs_df, jvs_df):
    '''returns the chunk of the vertices dataframe containing the data
    for the cell's neighbours
    '''
    neighb_idx = ctojs_df.xs(cell_idx, level='source').index.set_names(['vertex_index', 'stamp'])
    return jvs_df.loc[neighb_idx]



def main():

    source_inplace = os.path.join(os.path.dirname(bpy.data.filepath),
                                  "epithelium.h5")
    fname = os.path.join(source_inplace)

    scene = utils.clear_scene()


    vertices_df, edges_df = graph_io.import_graph(fname)
    cells_df = vertices_df[vertices_df['is_cell_vert'] == 1].swaplevel(0, 1).sortlevel()
    jvs_df = vertices_df[vertices_df['is_cell_vert'] == 0].swaplevel(0, 1).sortlevel()
    jes_df = edges_df[edges_df['is_junction_edge'] == 1].swaplevel(0,1).swaplevel(1,2).sortlevel()
    ctojs_df = edges_df[edges_df['is_junction_edge'] == 0].swaplevel(0,1).swaplevel(1,2).sortlevel()

    # is_cell = graph.vertex_properties['is_cell_vert']
    # is_alive = graph.vertex_properties['is_alive']
    # is_junction = graph.edge_properties['is_junction_edge']
    # ixs = graph.vertex_properties['ixs']
    # wys = graph.vertex_properties['wys']
    # zeds = graph.vertex_properties['zeds']

    cell_color = '#2887c8'
    edge_color = '#3dc828'
    for cell_idx, cell_df in cells_df.groupby(level='vertex_index'):
        init_stamp = cell_df.index.get_level_values('stamp')[0]
        init_cell = cell_df.iloc[0]

        if not init_cell.is_alive:
            continue
        name = 'cell{}'.format(cell_idx)
        neighbs_df = out_neighbours(cell_idx, ctojs_df, jvs_df).xs(init_stamp, level='stamp')
        if neighbs_df.shape[0] == 0:
            continue
        coords = ['ixs', 'wys', 'zeds']
        c_x, c_y, c_z =  init_cell[coords]
        j_xx, j_yy, j_zz = neighbs_df[coords].values.T
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
    # for je in graph.edges():
    #     if not is_junction[je]: continue
    #     v0, v1 = je
    #     name = 'je'+str(v0)+'to'+str(v1)
    #     je_obj = objects.set_junction(name,
    #                                   (ixs[v0], ixs[v1]),
    #                                   (wys[v0], wys[v1]),
    #                                   (zeds[v0], zeds[v1]), edge_color)
    for je_idx, je_df in jes_df.groupby(level=['source', 'target']):
        src_idx, trgt_idx = je_idx
        init_stamp = je_df.index.get_level_values(level='stamp')[0]
        name = 'je{}to{}'.format(src_idx, trgt_idx)
        j_src = jvs_df.xs((src_idx, init_stamp))
        j_trgt = jvs_df.xs((trgt_idx, init_stamp))
        je_obj = objects.set_junction(name,
                                      (j_src.ixs, j_trgt.ixs),
                                      (j_src.wys, j_trgt.wys),
                                      (j_src.zeds, j_trgt.zeds), edge_color)




    scene.frame_current = 0

    # Smooth objects view
    utils.select_all()
    bpy.ops.object.shade_smooth()

    # Add a camera
    bpy.ops.object.camera_add(view_align=True, enter_editmode=False)
    camera = bpy.context.object
    camera.name = "camera"

    scene.objects.active = None
