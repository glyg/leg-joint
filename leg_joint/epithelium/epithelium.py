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

from ..data import default_params

from .generation import cylindrical

from .objects import  Cells, ApicalJunctions
from ..io.xml_handler import ParamTree

from ..topology import Topology
from ..topology.topology import get_faces
from ..geometry import Triangles
from ..dynamics import Dynamics
from ..topology.filters import active, j_edges_in
from .graph_dataframe_interface import complete_pmaps, update_pmaps, update_dframes


CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(CURRENT_DIR)
PARAMFILE = default_params()

import tempfile
GRAPH_SAVE_DIR = tempfile.gettempdir()

# See [the tau manifesto](http://tauday.com/tau-manifesto)
tau = 2. * np.pi


class Epithelium(Topology,
                 Triangles,
                 Dynamics):
    """The :class:`Epithelium` class is the container for all the simulation.
    It inherits attributes form the following classes:

    * :class:`EpitheliumFilters`, providing utilities to create and
        filter the graph edges and vertices.
    * :class:`GraphWrapper` containing the geometrical aspects of
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

    def __init__(self,
                 identifier='0',
                 paramtree=None,
                 paramfile=PARAMFILE, copy=True,
                 graph=None,
                 graphXMLfile=None,
                 hdfstore=None, stamp=-1,
                 verbose=False,
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
        save_dir: str or path
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
        if graph is not None: ### From an existing graph
            self.graph = graph
            self.vertex_df, self.edge_df = hdfgraph.graph_to_dataframes(self.graph)
            self.new = False
            self.generate = False
        elif graphXMLfile is not None: ### From a graphML file
            self.graph = gt.load_graph(graphXMLfile)
            self.vertex_df, self.edge_df = hdfgraph.graph_to_dataframes(self.graph)
            self.new = False
            self.generate = False
            self.xmlfname = graphXMLfile
        elif hdfstore is not None: ### From a h5 file
            self.vertex_df, self.edge_df = hdfgraph.frames_from_hdf(hdfstore, stamp=stamp)
            self.graph = hdfgraph.graph_from_dataframes(self.vertex_df, self.edge_df)
            self.new = False
            self.generate = False
        else: #create
            log.info('Created new graph')
            self.new = True
            self.generate = True
            n_cells_circum = self.params['n_sigmas']
            n_cells_length = self.params['n_zeds']
            l_0 = self.params['lambda_0']
            h_0 = self.params['rho0']
            self.graph, self.vertex_df, self.edge_df = cylindrical(
                n_cells_circum, n_cells_length, l_0, h_0)
        self.rho_lumen = self.params['rho_lumen']

        Topology.__init__(self)
        self._properties_to_attributes()

        # All the geometrical properties are packed here
        _triangles = get_faces(self.graph)
        Triangles.__init__(self, _triangles)

        # Cells and Junctions initialisation
        log.info('Initial cells')
        self.cells = Cells(self)
        log.info('Initial junctions')
        self.junctions = ApicalJunctions(self)

        self.reset_topology(local=False)
        self._dataframes_to_properties()
        self._properties_to_attributes()

        # Dynamical components
        Dynamics.__init__(self)
        if self.new:
            log.info('Isotropic relaxation')
            self.isotropic_relax()
            log.info('Periodic boundary')
            self.periodic_boundary_condition()
        log.info('Update geometry')
        self.update_geometry()

    def _dataframes_to_properties(self):
        complete_pmaps(self.graph, self.vertex_df, self.edge_df)
        update_pmaps(self.graph, self.vertex_df, self.edge_df)

    def _properties_to_attributes(self):
        for name, vp in self.graph.vertex_properties.items():
            setattr(self, name, vp)
        for name, ep in self.graph.edge_properties.items():
            setattr(self, name, ep)

    def _update_dframes(self):
        update_dframes(self.graph, self.vertex_df, self.edge_df)



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
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s -'
            '%(levelname)s - {} :'
            ' %(message)s @ stamp {}'.format(self.identifier,
                                             self.stamp))
        fh.setFormatter(formatter)
        # add the handlers to the logger
        log.addHandler(fh)


    def dump_json(self, parameters):

        with open(self.paths['json'], 'w+') as json_file:
            json.dump(parameters, json_file, sort_keys=True)
            log.info('Wrote {}'.format(self.paths['json']))

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
