# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os

data_path = os.path.dirname(os.path.abspath(__file__))

def default_params():
    '''Returns the path to the default parameter
    file (here: <path/to/leg_joint/data/defaults/params.xml)
    '''
    return os.path.join(data_path, 'defaults', 'params.xml')

def initial_graph_xml():
    '''Returns the path to the epithelium used
    as the initial tissue before cell divisions
    '''
    return os.path.join(data_path, 'graphs', 'initial_graph.xml')


def before_apoptosis_xml():
    '''Returns the path to the epithelium used
    as the initial tissue before apoptosis)
    '''
    return os.path.join(data_path, 'graphs', 'before_apoptosis.xml')

def small_xml():
    '''Returns the path to a small precomputed epithelium
    (mostly for testing)
    '''
    return os.path.join(data_path, 'graphs', 'small.xml')


def get_image(im_name):
    '''Returns the path to the image im_name
    '''
    return os.path.join(data_path, 'imgs', im_name)
