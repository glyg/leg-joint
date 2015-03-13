# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import hdfgraph


def hdf_snapshot(func, *args, **kwargs):
    '''
    Decorator for :class:`Epithelium` objects that store
    the object's graph in an HDF5 file, by appending the
    :class:`gt.Graph` PropertyMaps values, indexed by the
    :class:`Epithelium` `stamp` attribute.
    '''
    def new_func(self, *args, **kwargs):
        out = func(self, *args, **kwargs)
        store = self.paths['hdf']
        try:
            hdfgraph.graph_to_hdf(self.graph, store,
                                  stamp=self.stamp,
                                  reset=False,
                                  vert_kwargs={'data_columns':['ixs', 'wys', 'zeds', 'thetas']})
        except:
            self.log.error('HDF snapshot failed at stamp %i' % self.stamp)
        return out
    return new_func
