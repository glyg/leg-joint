# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import os
import markdown
import pylab as plt
from .graph_representation import epithelium_draw as draw
from .graph_representation import plot_avg_rho


def get_report(eptm, reset=False, base_path=None):

    if base_path is None:
        base_path = eptm.paths['png']
    report = []
    
    view_2d = os.path.join(base_path, 'tissue_2d.png')
    view_3d = os.path.join(base_path, 'tissue_3d.png')
    if reset:
        draw(eptm, output2d=view_2d, output3d=view_3d)

    profile = os.path.join(base_path, 'radial_profile.png')
    if reset:
        ax = plot_avg_rho(eptm, 20)
        fig = ax.get_figure()
        plt.savefig(profile)
        plt.close(fig)
        
    header = '''##Report for epithelium {}\n\n '''.format(eptm.identifier)
    report.append(header)
    
    fnames = [view_3d, view_2d, profile]
    titles = ['Whole graph 3D representation',
              'Whole graph 2D representation',
              'Radial profile']
    for img_fname, title in zip(fnames, titles):
        report.append(_img_string(img_fname, title))
    return markdown.markdown('\n'.join(report))

def _img_string(img_fname, title, alt_text='', base_url=None):
    if base_url is not None:
        img_fname = ''.join(base_url)
    if len(alt_text) == 0:
        alt_text = title
    string = '''###{1}\n![{2}]({0} "{1}")\n'''.format(img_fname,
                                                      title,
                                                      alt_text)
    return string
