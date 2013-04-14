#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
curdir = os.path.abspath(os.path.curdir)
#print os.path.dirname(curdir)
sys.path.append(os.path.dirname(curdir))
import matplotlib.pyplot as plt
import visvis as vv

import leg_joint as lj



def create_edgeline(eptm):
    for n, je in enumerate(eptm.junctions):
        ps = vv.Pointset(3)
        src = vv.Point(eptm.ixs[je.source()],
                       eptm.wys[je.source()],
                       eptm.zeds[je.source()])
        trgt = vv.Point(eptm.ixs[je.target()],
                        eptm.wys[je.target()],
                        eptm.zeds[je.target()])
        ps.append(src)
        ps.append(trgt)
        if n == 0:
            edgeline = vv.solidLine(ps, radius=0.05, N=4)
            edgeline.faceColor = 'g'
        elif n < 5:

            segment =  vv.solidLine(ps, radius=0.05, N=4)
            edgeline = vv.Mesh(edgeline, vv.solidLine(ps, radius=0.05, N=4))
            edgeline.faceColor = 'g'
            
    return edgeline

#app = vv.use()

    
eptm = lj.Epithelium(graphXMLfile='../saved_graphs/xml/latest.xml',
                     paramfile='../default/params.xml',
                     verbose=False)

edgeline = create_edgeline(eptm)

vv.meshWrite('test.obj', edgeline, 'edgeline')
print 'saved'
