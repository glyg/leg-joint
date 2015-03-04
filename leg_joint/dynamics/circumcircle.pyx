# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import numpy as np
cimport numpy as np
cimport cython

# cdef extern from "numpy/arrayobject.h":
#     ctypedef class numpy.ndarray [object PyArrayObject]:
#         cdef char *data
#         cdef int nd
#         cdef Py_intptr_t *dimensions
#         cdef Py_intptr_t *strides
#     cdef void import_array()
#     cdef int  PyArray_ITEMSIZE(np.ndarray)
#     cdef void * PyArray_DATA(np.ndarray)



def c_circumcircle(np.ndarray[dtype=double, ndim=1] sz0,
                   np.ndarray[dtype=double, ndim=1] sz1,
                   np.ndarray[dtype=double, ndim=1] sz2,
                   double cutoff):

    cdef double sigma0 = sz0[0]
    cdef double sigma1 = sz1[0]
    cdef double sigma2 = sz2[0]

    cdef double zed0 = sz0[1]
    cdef double zed1 = sz1[1]
    cdef double zed2 = sz2[1]

    
    cdef double x1 = sigma1 - sigma0
    cdef double y1 = zed1 - zed0
    cdef double x2 = sigma2 - sigma0
    cdef double y2 = zed2 - zed0

    cdef double xc, yc, a1, a2, b1, b2

    if (y1**2 < cutoff**2 and  y2**2 > cutoff**2):
        xc = x1 / 2.
        yc = (x2**2 + y2**2 - x2 * x1) / (2 * y2)
    elif (y2**2 < cutoff**2 and y1**2 > cutoff**2):
        xc = x2 / 2.
        yc = (x1**2 + y1**2 - x1 * x2) / (2 * y1)
    elif (y1**2 + y2**2 < 2 * cutoff**2):
        xc = 1e12
        yc = 1e12
    else:
       a1 = -x1/y1;
       a2 = -x2/y2;
       b1 = (x1*x1 + y1*y1) / (2*y1);
       b2 = (x2*x2 + y2*y2) / (2*y2);
       if ((a2 - a1) * (a2 - a1) < cutoff*cutoff):
           xc = 1e12
           yc = 1e12
       xc = (b1 - b2) / (a2 - a1)
       yc = a1 * xc + b1
    return xc + sigma0, yc + zed0
