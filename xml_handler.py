#!/usr/bin/python
# -*- coding: utf-8 -*-

# Handler for the parameters xml files -- to allow phase space exploration

import sys, os
import numpy as np

from xml.etree.ElementTree import Element
from xml.etree.ElementTree import parse, tostring

#Those strings should be respected in the xml file
SPRING_UNIT=u'pN/µm' 
DRAG_UNIT = u'pN.s/µm' 
LENGTH_UNIT = u'µm' 
FREQ_UNIT = u'Hz' 
FORCE_UNIT = u'pN' 
SPEED_UNIT = u'µm/s'


__all__ = ["ParamTree", "indent", "ResultTree"]

def indent(elem, level=0):
    '''utility to have a nice printing of the xml tree
    '''
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

class ParamTree(object):
    """This class defines the container for the simulation parameters.
    It wraps an ElementTree instance whose elements contains the
    name, value (as a string), description and unit of each parameter and a
    dictionnary that is used during the simulation.


    The `value` attribute of the tree is not modified by adimentionalization,
    whereas the value in the dictionnary is changed.
    """

    def __init__(self, filename=None, root=None, adimentionalized=True):

        if filename is not None:
            self.filename = filename
            source = file(filename, "r")
            self.tree = parse(source)
            self.root = self.tree.getroot()
            source.close()
        elif root is not None:
            self.root = root
        else:
            print 'A etree root or a filename should be provided'

        list=[]
        a = self.root.findall("param")
        for i in a:
            n = i.get("name")
            v = i.get("value")
            if '.' in v or 'e' in v:
                v = float(v)
            else:
                v = int(v)
            list.append((n, v))
        self.absolute_dic = dict(list)
        self.relative_dic = dict(list)
        if adimentionalized :
            self.dimentionalize()
            
    def has_unit(self, param, UNIT):

        unit_str = param.find("unit").text
        if unit_str == UNIT:
            return True
        else :
            return False

    def dimentionalize(self):

        '''
        Note that the "value" attribute of the ElementTree instance are
        **not** modified. Also, applying this function on an already
        dimentionalized  dictionnary won"t change any thing
        '''

        prefered_area = self.absolute_dic["prefered_area"]
        elasticity = self.absolute_dic["elasticity"]
        contractility = self.relative_dic["contractility"]
        line_tension = self.relative_dic["line_tension"]

        self.absolute_dic["line_tension"] = line_tension *\
                                            elasticity * prefered_area**(3 / 2.)
        self.absolute_dic["contractility"] = contractility *\
                                             elasticity * prefered_area

    def change_dic(self, key, new_value, write=False,
                   back_up=False, verbose=False):
        '''
        Changes the Element tree and re-creates the associated dictionnary.
        If write is True, re_writes the parameters files
        (older version is backed up if back_up is True)
        
        new_value is absolute - it has units
        '''
        if self.absolute_dic is None:
            self.create_dic()
        a = self.root.findall('param')
        for item in a:
            if item.attrib['name'] == key:
                item.attrib['value'] = str(new_value)
                try:
                    self.absolute_dic[key] = new_value
                except KeyError:
                    print "Couldn't find the parameter %s" %key
                    return 0
                break
        # try:
        #     self.adimentionalize()
        # except KeyError:
        #     print 'Oooups'
        #     raise()

        # FIXME! 
        if write:
            raise NotImplementedError
            #continue
        
            # indent(self.root)
            # xf = open(self.filename, 'w+')
            # if back_up:
            #     bck = self.filename+'.bck'
            #     xfb = open(bck,'w+')
            #     for line in xf:
            #         xfb.write(line)
            #     xfb.close()
            #     xf.seek(0)
            #     print "Backed up old parameter file as %s" %bck
            # else :
            #     print "Warning : %s changed without back up" %self.filename
            
            # xf.write(tostring(self.root))
            # xf.close
            # print "Changed  %s to %03f in file %s" %(key,
            #                                          new_value,
            #                                          self.filename)
        elif verbose:
            print "Warning: parameter %s changed but not written!" %key



        
class ResultTree(ParamTree):    

    def __init__(self, xmlfname = "resuts.xml"):

        xmlfname = os.path.abspath(xmlfname)
        ParamTree.__init__(self, xmlfname, adimentionalized = False)
        datafname = self.root.get("datafile")
        if not os.path.isabs(datafname):
            if '/' not in datafname:
                self.datafname = os.path.join(os.path.dirname(xmlfname), 
                                              datafname)
            else:
                self.datafname = os.path.join(os.path.dirname(xmlfname), 
                                              datafname.split('/')[-1])
        else:
            self.datafname = datafname
        if self.datafname is None or not os.path.isfile(self.datafname):
            raise ValueError, "Corresponding data file not specified"
        if self.datafname.endswith('.npy'):
            self.data = np.load(self.datafname)
        else:
            self.data = np.loadtxt(self.datafname, delimiter=' ',
                                comments = '#')        
    
