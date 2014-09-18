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

def out_neighbours(vert_idx, vertices_df, edges_df):
    '''returns the chunk of the vertices dataframe containing the data
    for the vertex's out neighbours
    '''
    try:
        neighb_idx = edges_df.xs(vert_idx, level='source').index.set_names(['vertex_index', 'stamp'])
    except KeyError:
        return None
    return vertices_df.loc[neighb_idx]


def out_edges(vert_idx, vertices_df, edges_df):
    '''returns the chunk of the edges dataframe containing the data
    for the vertex's out edges
    '''
    try:
        out_edges = edges_df.xs(vert_idx, level='source').index.set_names(['vertex_index', 'stamp'])
    except KeyError:
        return None
    return out_edges



def main():

    source_inplace = os.path.join(os.path.dirname(bpy.data.filepath),
                                  "epithelium.h5")
    fname = os.path.join(source_inplace)

    scene = utils.clear_scene()

    cell_color = '#2887c8'
    edge_color = '#3dc828'


    vertices_df, edges_df = graph_io.import_graph(fname)
    vertices_df = vertices_df.swaplevel(0, 1).sortlevel()
    edges_df = edges_df.swaplevel(0,1).swaplevel(1,2).sortlevel()
    cells_df = vertices_df[vertices_df['is_cell_vert'] == 1]
    jvs_df = vertices_df[vertices_df['is_cell_vert'] == 0]
    jes_df = edges_df[edges_df['is_junction_edge'] == 1]
    ctojs_df = edges_df[edges_df['is_junction_edge'] == 0]


    # for cell_idx, cell_df in cells_df.groupby(level='vertex_index'):
    #     init_stamp = cell_df.index.get_level_values('stamp')[0]
    #     init_cell = cell_df.iloc[0]

    #     if not init_cell.is_alive:
    #         continue
    #     name = 'cell{}'.format(cell_idx)
    #     neighbs_df = out_neighbours(cell_idx, ctojs_df, jvs_df).xs(init_stamp, level='stamp')
    #     if neighbs_df.shape[0] == 0:
    #         continue
    #     coords = ['ixs', 'wys', 'zeds']
    #     c_x, c_y, c_z =  init_cell[coords]
    #     j_xx, j_yy, j_zz = neighbs_df[coords].values.T
    #     rel_xx = j_xx - c_x
    #     rel_yy = j_yy - c_y
    #     rel_zz = j_zz - c_z
    #     rel_ss = np.arctan2(rel_yy, rel_xx) * np.hypot(rel_yy, rel_xx)
    #     phis = np.arctan2(rel_zz, rel_ss)
    #     indices = np.argsort(phis)
    #     rel_xx = rel_xx[indices]
    #     rel_yy = rel_yy[indices]
    #     rel_zz = rel_zz[indices]
    #     cell_obj = objects.set_cell(name, c_x, c_y, c_z, init_stamp,
    #                                 rel_xx, rel_yy, rel_zz, cell_color)

    for (jv_idx, stamp), jv_df in jvs_df.iterrows():

        name = 'jv{}'.format(jv_idx)
        objects.set_jv(name, stamp,
                       jv_df.ixs, jv_df.wys, jv_df.zeds)

    for je_idx, je_df in jes_df.iterrows():
        src_idx, trgt_idx, stamp = je_idx
        #stamp = je_df.index.get_level_values(level='stamp')[0]
        name = 'je{}to{}'.format(src_idx, trgt_idx)
        je_obj = objects.set_junction_arm(name, src_idx, trgt_idx, edge_color)
        # break
        # j_src = jvs_df.xs((src_idx, stamp))
        # j_trgt = jvs_df.xs((trgt_idx, stamp))
        # je_obj = objects.set_junction(name, stamp,
        #                               (j_src.ixs, j_trgt.ixs),
        #                               (j_src.wys, j_trgt.wys),
        #                               (j_src.zeds, j_trgt.zeds),
        #                               edge_color)

    scene.frame_current = 0
    bpy.ops.object.posemode_toggle()
    # Smooth objects view
    utils.select_all()
    bpy.ops.object.shade_smooth()

    # Add a camera
    bpy.ops.object.camera_add(view_align=True, enter_editmode=False)
    camera = bpy.context.object
    camera.name = "camera"

    scene.objects.active = None
