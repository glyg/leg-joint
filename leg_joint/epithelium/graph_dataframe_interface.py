# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import logging
log = logging.getLogger(__name__)


def update_dframes(graph, vertex_df, edge_df, vcols=None, ecols=None):
    if vcols is not None:
        vitems = {col: graph.vertex_properties[col] for col in vcols}.items()
    else:
        vitems = graph.vertex_properties.items()
    if ecols is not None:
        eitems = {col: graph.edge_properties[col] for col in vcols}.items()
    else:
        eitems = graph.edge_properties.items()

    for col, prop in vitems:
        try:
            vertex_df[col] = prop.fa
        except KeyError:
            log.info('Property {} not in vertex dataframe'.format(col))
    for col, prop in eitems:
        try:
            edge_df[col] = prop.fa
        except KeyError:
            log.info('Property {} not in edge dataframe'.format(col))



def complete_pmaps(graph, vertex_df, edge_df, vcols=None, ecols=None):
    '''
    Completes the vertex and edge (internalized) property maps
    with the columns of the vertex_df and edge_df DataFrames.
    Warning: this only initializes the properymaps, values are NOT set


    '''
    if vcols is None:
        vdf_cols = set(vertex_df.columns)
    else:
        vdf_cols = vcols
    vp_cols = set(graph.vertex_properties.keys())
    missing_vps = vdf_cols.difference(vp_cols)

    for new in missing_vps:
        try:
            dtype = _guess_dtype(new, vertex_df)
            new_vp = graph.new_vertex_property(dtype)
            graph.vertex_properties[new] = new_vp
        except ValueError:
            log.warning(
                "Data type not supported for column {}, "
                "it won't be passed as a vector property".format(new))


    if ecols is None:
        edf_cols = set(edge_df.columns)
    else:
        edf_cols = ecols
    ep_cols = set(graph.edge_properties.keys())
    missing_eps = edf_cols.difference(ep_cols)

    for new in missing_eps:
        try:
            dtype = _guess_dtype(new, edge_df)
            new_vp = graph.new_edge_property(dtype)
            graph.edge_properties[new] = new_vp

        except ValueError:
            log.warning(
                "Data type not supported for column {}, "
                "it won't be passed as a vector property".format(new))

def update_pmaps(graph, vertex_df, edge_df, vcols=None, ecols=None):

    if vcols is not None:
        vitems = {col: graph.vertex_properties[col] for col in vcols}.items()
    else:
        vitems = graph.vertex_properties.items()
    if ecols is not None:
        eitems = {col: graph.edge_properties[col] for col in vcols}.items()
    else:
        eitems = graph.edge_properties.items()

    for col, prop in vitems:
        try:
            prop.fa = vertex_df[col]
        except KeyError:
            log.debug('Property {} not in vertex dataframe'.format(col))
    for col, prop in eitems:
        try:
            prop.fa = edge_df[col]
        except KeyError:
            log.debug('Property {} not in edge dataframe'.format(col))

def _guess_dtype(col, df):

    if 'float' in df.dtypes[col].name:
        return 'float'
    elif 'int' in df.dtypes[col].name:
        if set(df[col].unique()).issubset({0, 1}) :
            return 'bool'
        else:
            return 'long'
    else:
        raise ValueError(
            'Unsuported data type {}'.format(df.dtypes[col].name))