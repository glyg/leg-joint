# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# Handler for the parameters xml files -- to allow phase space exploration

from xml.etree.ElementTree import parse
import warnings

#Those strings should be respected in the xml file
SPRING_UNIT='pN/um' 
DRAG_UNIT = 'pN.s/um' 
LENGTH_UNIT = 'um' 
FREQ_UNIT = 'Hz' 
FORCE_UNIT = 'pN' 
SPEED_UNIT = 'um/s'


__all__ = ["ParamTree", "indent"]

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
            with open(filename, "r") as source:
                self.tree = parse(source)
                self.root = self.tree.getroot()
        elif root is not None:
            self.root = root
        else:
            print('A etree root or a filename should be provided')

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
        dimentionalized  dictionnary won"t change anything
        '''
        
        prefered_area = self.absolute_dic["prefered_area"]
        height = self.absolute_dic["prefered_height"]
        prefered_vol = prefered_area * height
        self.absolute_dic['cell_volume'] = prefered_vol
        contractility = self.relative_dic["contractility"]
        line_tension = self.relative_dic["line_tension"]
        radial_tension = self.relative_dic["radial_tension"]
        vol_elasticity = self.relative_dic["vol_elasticity"]
        
        self.absolute_dic["line_tension"] = (
            line_tension * vol_elasticity
            * prefered_area**(3 / 2.) * height**2)
        self.absolute_dic["radial_tension"] = (
            radial_tension * vol_elasticity
            * prefered_area**(3 / 2.) * height**2)
        self.absolute_dic["contractility"]  = (
            contractility * vol_elasticity
            * prefered_area * height**2)

        self.absolute_dic["rho_lumen"] = (self.absolute_dic['rho0']
            - self.absolute_dic["prefered_height"])

        
    def change_dic(self, key, new_value, write=False,
                   back_up=False, verbose=False):
        '''Changes the Element tree and re-creates the associated dictionnary.
        If write is True, re_writes the parameters files
        (older version is backed up if back_up is True)
        
        new_value is dimention less : for the `line_tension`,
        `radial_tension` and `contractility` parameters a dimentionalized 
        value will be computed
        '''
        a = self.root.findall('param')
        for item in a:
            if item.attrib['name'] == key:
                item.attrib['value'] = str(new_value)
                try:
                    self.relative_dic[key] = new_value
                except KeyError:
                    print("Couldn't find the parameter %s" %key)
                    raise
                break

        # FIXME! 
        if write:
            raise NotImplementedError
        elif verbose:
            warnings.warn("Parameter %s changed but not written!" %key)

        self.dimentionalize()
        