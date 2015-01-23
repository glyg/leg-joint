# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import warnings
import datetime
import json
import logging

log = logging.getLogger(__name__)

import graph_tool.all as gt
import numpy as np
import hdfgraph


from .data import default_params
from .objects import  AbstractRTZGraph, Cells, ApicalJunctions
from .xml_handler import ParamTree
from .dynamics import Dynamics
from .filters import active, j_edges_in, EpitheliumFilters

CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(CURRENT_DIR)
PARAMFILE = default_params()
GRAPH_SAVE_DIR = '.'


# See [the tau manifesto](http://tauday.com/tau-manifesto)
tau = 2. * np.pi


class Epithelium(EpitheliumFilters,
                 AbstractRTZGraph,
                 Dynamics):
    """The :class:`Epithelium` class is the container for all the simulation.
    It inherits attributes form the following classes:

    * :class:`EpitheliumFilters`, providing utilities to create and
        filter the graph edges and vertices.
    * :class:`AbstractRTZGraph` containing the geometrical aspects of
        the simulation in 3D space (e.g. coordinate systems)
    * :class:`Dynamics` containing the dynamical aspects, i.e. the
        functions to compute the energy and the gradients.

    Please refer to those classes documentations for more details.

    The main attribute of the :class:`Epithelium` is a `graph_tool`
    :class:`gt.Graph` instance. The epithelium graph is oriented and
    contains two types of vertices:

     1 the cell centers, with their properties contained
       in a :class:`Cells` instance

     2 the apical vertices

    Each cell is linked to each of its vertices by
    a graph edge, oriented from the cell to the junction vertices.

    The apical junctions are formed by oriented edges from one
    junction vertex to the other, and their properties are contained
    in a :class:`Junctions` instance.

    """

    def __init__(self, graphXMLfile=None, identifier='0',
                 paramtree=None,
                 paramfile=PARAMFILE, copy=True,
                 graph=None, verbose=False,
                 save_dir=GRAPH_SAVE_DIR,
                 **params):
        """
        Parameters
        ----------
        graphXMLfile : file name or file instance, optional.
            It should point to an `xml` (or `xml.gz`) file as output by
            a graph_tool :class:`Graph` `save` attribute from a previous
            simulation. If it is not provided, a nex graph will be created

        paramtree : an instance of the :class:`ParamTree` class, optional
           the corresponding class is defined in
           the `xml_handler` module. If not provided, paramters
           will be read from the `paramfile` argument

        paramfile : an xml file, defaults to `default/params.xml`
           paramfile contains the paramters values for the simulation.

        graph : a graph_tool :class:`Graph` instance

        verbose : bool, optional
           if `True`, the simulation will output -possibly lots of-
           information on the successive operations.
        """
        # Parametrisation
        if paramtree == None:
            self.paramtree = ParamTree(paramfile)
        else:
            self.paramtree = paramtree
        for key, value in params.items():
            try:
                self.paramtree.change_dic(key, value)
            except KeyError:
                pass
        self.params = self.paramtree.absolute_dic
        if copy or graphXMLfile is None:
            self.set_identifier(identifier)
            self.save_dir = os.path.join(save_dir, identifier)
        else:
            xml_path = os.path.abspath(graphXMLfile)
            splitted = xml_path.split(os.path.sep)
            self.identifier = splitted[-3]
            self.save_dir = os.path.sep.join(splitted[:-2])
        self.stamp = 0
        log.info('Instanciating epithelium %s' % identifier)

        self._init_paths(copy)
        self._set_logger()
        self.log = log

        # Graph instanciation
        if graph is None and graphXMLfile is None:
            log.info('Created new graph')
            self.graph = gt.Graph(directed=True)
            self.new = True
            self.generate = True
        elif graphXMLfile is not None :
            self.graph = gt.load_graph(graphXMLfile)
            self.new = False
            self.generate = False
            self.xmlfname = graphXMLfile
        elif graph is not None:
            self.graph = graph
            self.new = False
            self.generate = False

        self.__verbose__ = verbose
        EpitheliumFilters.__init__(self)
        # All the geometrical properties are packed here
        AbstractRTZGraph.__init__(self)
        self.diamonds = self.graph.new_edge_property('object')

        # Cells and Junctions initialisation
        log.info('Initial cells')
        self.cells = Cells(self)
        log.info('Initial junctions')
        self.junctions = ApicalJunctions(self)
        if self.generate:
            self.is_alive.a = 1
            # Remove cell to cell edges after graph construction
            efilt = self.is_ctoj_edge.copy()
            efilt.a += self.is_junction_edge.a
            if self.__verbose__ == True:
                total_edges = self.graph.num_edges()
                good_edges = efilt.a.sum()
                log.info('removing %i cell to cell edges '
                      % (total_edges - good_edges))
            self.graph.set_edge_filter(efilt)
            self.graph.purge_edges()
            self.graph.set_vertex_filter(None)
            self.graph.set_edge_filter(None)

        self.reset_topology(local=False)
        # Dynamical components
        Dynamics.__init__(self)
        if self.new:
            log.info('Isotropic relaxation')
            self.isotropic_relax()
            log.info('Periodic boundary')
            self.periodic_boundary_condition()
        log.info('Update geometry')
        self.update_geometry()

    def __str__(self):

        num_cells = self.is_cell_vert.fa.sum()
        num_edges = self.is_junction_edge.fa.sum()
        str1 = ['<Epithelium with %i cells and %i junction edges'
                ' at %s>'
                % (num_cells, num_edges, hex(id(self)))]
        # str1.append('Vertex Properties:\n'
        #             '==================')
        # for key in sorted(self.graph.vertex_properties.keys()):
        #     str1.append('    * %s' % key)
        # str1.append('Edge Properties:\n'
        #             '================')
        # for key in sorted(self.graph.edge_properties.keys()):
        #     str1.append('    * %s' % key)

        str1.append('Identifier : %s' % self.identifier)
        str1.append('Directory : %s' % os.path.abspath(self.paths['root']))
        return '\n'.join(str1)

    def set_identifier(self, identifier='', reset=True):
        if not hasattr(self, 'identifier'):
            now = datetime.datetime.isoformat(
                datetime.datetime.utcnow())
            time_tag = '_'.join(now.split(':')).split('.')[0]
            self.identifier = '_'.join((identifier, time_tag))

    def _set_logger(self):

        fh = logging.FileHandler(self.paths['log'])
        fh.setLevel(logging.INFO)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s -'
                                      '%(levelname)s - {} :'
                                      ' %(message)s @ stamp {}'.format(self.identifier,
                                                                       self.stamp))
        fh.setFormatter(formatter)
        # add the handlers to the logger
        log.addHandler(fh)


    def dump_json(self, parameters):

        with open(self.paths['json'], 'w+') as json_file:
            json.dump(parameters, json_file, sort_keys=True)
            log.info('Wrote %s' % self.paths['json'])

    def _init_paths(self, copy):
        '''Creates the paths where graphs will be saved.

        The root directory is stored as a string in the
        `self.save_dir` attribute and the created paths are stored in
        the `self.paths` dictionnary.

        `self.paths` contains the following keys:

        * 'png', 'pdf', 'svg' are paths to _directories_ where
           graph representations will be stored.

        * 'xml' is a directory for static views of the epithelium graph,
           stored in graphML

        * 'hdf' is a path to a *file* containing the graph stored in a
        `.h5` file through the `hdfstore` module (see the module doc
        for more details).

        '''
        self.paths = {'root': os.path.abspath(self.save_dir)}
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        for filetype in ['png', 'pdf', 'svg', 'xml']:
            subdir = os.path.join(self.save_dir, filetype)
            if not os.path.isdir(subdir):
                os.mkdir(subdir)
            self.paths[filetype] = os.path.abspath(subdir)

        store = os.path.join(self.save_dir,
                             'eptm_%s.h5' % self.identifier)
        self.paths['hdf'] = os.path.abspath(store)
        self.paths['json'] = os.path.join(self.save_dir,
                                         'params_%s.json' % self.identifier)
        self.paths['log'] = os.path.join(self.save_dir,
                                         '%s.log' % self.identifier)


    def update_geometry(self):
        '''
        Computes cell positions (at the geometrical center
        of the junction vertices), the edge lengths and the
        cell geometry (area and volume)
        '''

        self.update_cells_pos()
        self.update_rhotheta()
        self.update_deltas()
        self.update_edge_lengths()
        # self.update_dsigmas()
        # Edges
        for j_edge in self.junctions:
            self.diamonds[j_edge].update_geometry()
        # cells
        for cell in self.cells:
            self._one_cell_geom(cell)

    def _one_cell_geom(self, cell):
        """
        """
        j_edges = self.cells.junctions[cell]
        if len(j_edges) < 3:
            if not self.is_alive[cell]:
                return
            log.error('''Two edges ain't enough to compute
                      area for cell %s''' % cell)
            self.cells.vols[cell] = self.cells.prefered_vol[cell]
            self.cells.perimeters[cell] = 0.
            self.is_alive[cell] = 0
        else:
            self.cells.areas[cell] = 0.
            self.cells.perimeters[cell] = 0.
            self.cells.vols[cell] = 0.
            for j_edge in j_edges:
                try:
                    tr = self.diamonds[j_edge].triangles[cell]
                    self.cells.areas[cell] += tr.area
                    self.cells.perimeters[cell] += tr.length
                    self.cells.vols[cell] += tr.vol
                except KeyError:
                    pass

    def set_new_pos(self, new_xyz_pos):
        '''
        Modifies the position of the **active** junction vertices

        Parameters:
        ----------
        new_xyz_pos: ndarray with shape (N, 3)
            where N is the number of active vertices, containing the
            new positions in the cartesian coordinate system
        '''
        self._set_junction_pos(new_xyz_pos)

    @active
    def _set_junction_pos(self, new_xyz_pos):
        new_xyz_pos = new_xyz_pos.flatten()

        if not np.all(np.isfinite(new_xyz_pos)):
            log.critical('''Non finite value for vertices {}'''.format(
                ', '.join(str(jv) for jv in self.graph.vertices())))
            return
        self.ixs.fa = new_xyz_pos[::3]
        self.wys.fa = new_xyz_pos[1::3]
        self.zeds.fa = new_xyz_pos[2::3]

    def update_cells_pos(self):
        for cell in self.cells:
            self._set_cell_pos(cell)

    def _set_cell_pos(self, cell):
        j_xyz = np.array([[self.ixs[jv], self.wys[jv], self.zeds[jv]]
                          for jv in cell.out_neighbours()])
        if len(j_xyz) < 3:
            return
        self.ixs[cell], self.wys[cell], self.zeds[cell] = j_xyz.mean(axis=0)

    def reset_topology(self, local=True):
        '''Computes the epithelium topology, by finding *de novo* the
        cell's junction edges and the adjacent cells for each junction
        edge.  If `local` is `True`, performs this operation only over
        local cells and junction edges.
        '''
        if local:
            for cell in self.cells.local_cells():
                self.cells.update_junctions(cell)
            for j_edge in self.junctions.local_junctions():
                self.junctions.update_adjacent(j_edge)
        else:
            for cell in self.cells:
                self.cells.update_junctions(cell)
            for j_edge in self.junctions:
                self.junctions.update_adjacent(j_edge)

    def add_junction(self, j_verta, j_vertb, cell0, cell1):
        '''Adds a junction to the epithelium, creating a junction edge
        between `j_verta` and `j_vertb`, cell to junction edges
        between `cell0` and `j_edgea`, `cell0` and `j_edgeb`, `cell1`
        and `j_edgea` and `cell1` and `j_edgeb`
        '''
        ##### TODO: This block should go in a decorator
        valid = np.array([obj.is_valid() for obj in
                          (cell0, j_verta, j_vertb)])
        if not valid.all():
            raise ValueError("invalid elements in the argument list"
                             "with cell0 => %s"
                             "     vertex a => %s "
                             "     vertex b => %s " % (str(v) for v in valid)
                             )
        ####
        j_edgeab = self.graph.edge(j_verta, j_vertb)
        if j_edgeab is not None:
            if self.__verbose__:
                warnings.warn('''Previous %s to %s
                             edge is re-created.'''
                             % (str(j_verta), str(j_vertb)))
            self.graph.remove_edge(j_edgeab)
        j_edgeab = self.graph.add_edge(j_verta, j_vertb)
        j_edge_old = self.cells.junctions[cell0][0]
        for e_prop in self.graph.edge_properties.values():
            e_prop[j_edgeab] = e_prop[j_edge_old]
        self.junctions.adjacent_cells[j_edgeab] = cell0, cell1

        self.cells.junctions[cell0].append(j_edgeab)

        if self.cells.junctions[cell1] is None:
            self.cells.junctions[cell1] = [j_edgeab,]
        else:
            self.cells.junctions[cell1].append(j_edgeab)

        prev_ctojs = [ctoj for ctoj in cell0.out_edges()]
        ctoj_old = prev_ctojs[0]

        ctoj_0a = self.graph.edge(cell0, j_verta)
        if ctoj_0a is not None:
            self.graph.remove_edge(ctoj_0a)
        ctoj_0a = self.graph.add_edge(cell0, j_verta)

        ctoj_0b = self.graph.edge(cell0, j_vertb)
        if ctoj_0b is not None:
            self.graph.remove_edge(ctoj_0b)
        ctoj_0b = self.graph.add_edge(cell0, j_vertb)

        for e_prop in self.graph.edge_properties.values():
            e_prop[ctoj_0a] = e_prop[ctoj_old]
            e_prop[ctoj_0b] = e_prop[ctoj_old]
        self._set_cell_pos(cell0)

        if cell1 is not None:
            ctoj_1a = self.graph.edge(cell1, j_verta)
            if ctoj_1a is not None:
                self.graph.remove_edge(ctoj_1a)
            ctoj_1a = self.graph.add_edge(cell1, j_verta)

            ctoj_1b = self.graph.edge(cell1, j_vertb)
            if ctoj_1b is not None:
                self.graph.remove_edge(ctoj_1b)
            ctoj_1b = self.graph.add_edge(cell1, j_vertb)
            for e_prop in self.graph.edge_properties.values():
                e_prop[ctoj_1a] = e_prop[ctoj_old]
                e_prop[ctoj_1b] = e_prop[ctoj_old]

            self._set_cell_pos(cell1)
        return j_verta, j_vertb, cell0, cell1

    def remove_junction(self, j_verta, j_vertb, cell0, cell1):
        '''Removes junction between `j_edgea` and `j_edgeb`, and the
        corresponding cell to junction edges for `cell0` and
        `cell1`
        '''

        #This block should go in a decorator
        valid = np.array([element.is_valid() for element in
                          (cell0, j_verta, j_vertb)])
        if not valid.all():
            raise ValueError("invalid elements in the argument list"
                             "with cell0 => %s"
                             "     vertex a => %s "
                             "     vertex b => %s " % (str(v) for v in valid))
        ####
        j_edgeab = self.graph.edge(j_verta, j_vertb)
        if j_edgeab is None:
            j_edgeab = self.graph.edge(j_vertb, j_verta)
        if j_edgeab is None:
            warnings.warn("Junction from %s to %s doesn't exist"
                         % (str(j_edgeab.source()), str(j_edgeab.target())))
            return
        # self.cells.junctions[cell0].remove(j_edgeab)
        # self.cells.junctions[cell1].remove(j_edgeab)
        # self.junctions.adjacent_cells[j_edgeab] = []

        self.graph.remove_edge(j_edgeab)
        self.cells.update_junctions(cell0)
        self.cells.update_junctions(cell1)
        #self.junctions.update_adjacent()
        ctoj_0a = self.graph.edge(cell0, j_verta)
        if ctoj_0a is not None:
            self.graph.remove_edge(ctoj_0a)
        ctoj_0b = self.graph.edge(cell0, j_vertb)
        if ctoj_0b is not None:
            self.graph.remove_edge(ctoj_0b)
        if cell1 is not None:
            ctoj_1a = self.graph.edge(cell1, j_verta)
            if ctoj_1a is not None:
                self.graph.remove_edge(ctoj_1a)
            ctoj_1b = self.graph.edge(cell1, j_vertb)
            if ctoj_1b is not None:
                self.graph.remove_edge(ctoj_1b)

    @j_edges_in
    def merge_j_verts(self, jv0, jv1):
        '''Merge junction vertices `jv0` and `jv1`. Raises an error if
        those vertices are not connected
        '''
        vertex_trash = []
        edge_trash = []
        je = self.any_edge(jv0, jv1)
        if je is None:
            raise ValueError('Can only merge connected edges')

        edge_trash.append(je)
        for vert in jv1.out_neighbours():
            old_edge = self.graph.edge(jv1, vert)
            if vert != jv0:
                new_edge = self.graph.add_edge(jv0, vert)
            for prop in self.graph.edge_properties.values():
                prop[new_edge] = prop[old_edge]
            edge_trash.append(old_edge)
        for vert in jv1.in_neighbours():
            old_edge = self.graph.edge(vert, jv1)
            if vert != jv0:
                new_edge = self.graph.add_edge(vert, jv0)
            for prop in self.graph.edge_properties.values():
                prop[new_edge] = prop[old_edge]
            edge_trash.append(old_edge)
        vertex_trash.append(jv1)
        return vertex_trash, edge_trash


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
