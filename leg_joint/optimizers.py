#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from scipy import optimize

import filters

class Optimizers(object):
    def __init__(self):
        pass

    @filters.active
    def precondition(self):
        if self.at_boundary.fa.sum() > 0:
            self.rotate(np.pi)
            self.current_angle = np.pi
            print('rotated')

        self.mean_rho = self.rhos.fa.mean()
            
        pos0 = np.vstack([self.rhos.fa,
                          self.thetas.fa * self.mean_rho,
                          self.zeds.fa]).T.flatten()

        max_rho = 2 * self.rhos.fa.mean()
        min_rho = 0#min(0, self.params['rho_lumen'])
        max_dzed = 2 * self.edge_lengths.fa.mean()
        max_dtheta = np.pi / self.params['n_sigmas']

        if self.__verbose__ : print ('Initial postion  has shape %s'
                              % str(pos0.shape))
        if self.__verbose__ : print ('Max displacement amplitude  : %.3f'
                             % max_dsigma)
        bounds = np.zeros((pos0.shape[0],),
                          dtype=[('min', np.float32),
                                 ('max', np.float32)])
        if self.__verbose__: print ('bounds array has shape: %s'
                             % str(bounds.shape))

        r_bounds = np.array([(min_rho, max_rho),] * self.rhos.fa.size)
        t_bounds = np.vstack((self.thetas.fa - max_dtheta,
                             self.thetas.fa + max_dtheta)).T * self.mean_rho
        z_bounds = np.vstack((self.zeds.fa - max_dzed,
                             self.zeds.fa + max_dzed)).T

        for n in range(pos0.shape[0]/3):
            bounds[3 * n] = (r_bounds[n, 0], r_bounds[n, 1])
            bounds[3 * n + 1] = (t_bounds[n, 0], t_bounds[n, 1])
            bounds[3 * n + 2] = (z_bounds[n, 0], z_bounds[n, 1])
    
        return pos0, bounds

    @filters.local
    def find_energy_min(self, method='fmin_l_bfgs_b',
                        tol=1e-8, approx_grad=0):
        '''
        Performs the energy minimisation
        '''
        pos0, bounds = self.precondition()
        output = 0
        if method == 'fmin_l_bfgs_b':
            # On my box, machine precision is 10^-19, so
            ## I set factr to 1e11 to avoid too long computation
            output = optimize.fmin_l_bfgs_b(self.opt_energy,
                                            pos0.flatten(),
                                            fprime=self.opt_gradient,
                                            approx_grad=approx_grad,
                                            bounds=bounds.flatten(),
                                            factr=1e8,
                                            m=10,
                                            pgtol=tol,
                                            epsilon=1e-8,
                                            iprint=1,
                                            maxfun=150,
                                            disp=None)

        elif method=='fmin':
            output = optimize.fmin(self.opt_energy,
                                   pos0.flatten(),
                                   ftol=tol, xtol=0.01,
                                   callback=self.opt_callback)
        elif method=='fmin_ncg':
            try:
                output = optimize.fmin_ncg(self.opt_energy,
                                                  pos0.flatten(),
                                                  fprime=self.opt_gradient,
                                                  avextol=tol,
                                                  retall=True,
                                                  maxiter=100,
                                                  callback=self.opt_callback)
            except:
                self.set_new_pos(pos0)
                self.graph.set_vertex_filter(None)
                output = 0

        elif method=='fmin_tnc':
            output = optimize.fmin_tnc(self.opt_energy,
                                              pos0.flatten(),
                                              fprime=self.opt_gradient,
                                              pgtol=tol,
                                              bounds=bounds,
                                              maxCGit=0,
                                              disp=5)
        elif method=='fmin_bfgs':
            output = optimize.fmin_bfgs(self.opt_energy,
                                               pos0.flatten(),
                                               fprime=self.opt_gradient,
                                               gtol=tol,
                                               norm=np.inf,
                                               retall=1,
                                               callback=self.opt_callback)
        if not -1e-8 < self.current_angle < 1e-8:
            self.rotate(-self.current_angle)
            self.current_angle = 0.
        return pos0, output

    @filters.local
    def approx_grad(self):
        pos0, bounds = self.precondition()
        grad = optimize.approx_fprime(pos0,
                                      self.opt_energy,
                                      epsilon=1e-9)
        return grad

    @filters.local
    def check_local_grad(self, retall=True):
        pos0, bounds = self.precondition()
        if self.__verbose__: print "Checking gradient"

        chk_out = optimize.check_grad(self.opt_energy,
                                      self.opt_gradient,
                                      pos0.flatten())
        return chk_out

    ### Apical optimization
    def opt_energy(self, rtz_pos):
        """
        After setting the rho, theta, zed position to rtz_pos,
        computes the energy over the graph as filtered
        by vfilt and efilt vertex and edge filters (respectively)
        """
        # Position setting
        self.set_new_pos(rtz_pos)
        self.update_geometry()
        energy = self.calc_energy()
        return energy

    def opt_gradient(self, rtz_pos):
        """
        After setting the rho, theta, zed position to rtz_pos,
        computes the gradient over the filtered graph
        """
        # # position setting
        self.set_new_pos(rtz_pos)
        self.update_geometry()
        self.update_gradient()
        gradient = self.gradient_array()
        return gradient

    def opt_callback(self, rtz_pos):
        """ Call back for the optimization """
        self.periodic_boundary_condition()
        self.update_geometry()
        self.update_gradient()
