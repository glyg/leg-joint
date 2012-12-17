#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
# We can use this to look directly at the XML file
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import parse, tostring
# The graph
import graph_tool.all as gt
from dynamics import Epithelium

def test1():
    eptm1 = Epithelium(paramfile='default/few_big_cells.xml')
    eptm1.graph.save("saved_graphs/test1.xml")
    print eptm1
    print eptm1.zeds.a[:3]
    return eptm1
    
def test2():

    eptm2 = Epithelium(graphXMLfile="saved_graphs/test1.xml")
    print eptm2
    print eptm2.zeds.a[:3]
    return eptm2


if __name__ == '__main__':
    switch = sys.argv[1]
    if switch == 'all':
        print 'test1:'
        test1()
        print 'test2:'
        test2()
    elif int(switch) == 2:
        print 'test2:'
        test2()
    elif int(switch) == 12:
        print 'test1:'
        test1()
        print 'test2:'
        test2()
        