#!/usr/bin/env python3
import sys,os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import ode
import tools

mpl,fntsize,labsize = tools.set_mpl(mpl)
##########################################################################

def main():
    p = Parameters()
    fn = Functions(p)

    output = integrator(fn)

    fig = plt.figure(figsize=(12,8))
    plt.plot(output[:,0]*p.ub/p.dc,output[:,1]/output[0,1])
    plt.xlim([0,output[-1,0]*p.ub/p.dc])
    plt.xlabel(r'$u_{b_1}t/d_c$')
    plt.ylabel(r'$p_w/p_{w_0}$')
    plt.savefig('figs/pwpi_step_ub.pdf',bbox_inches='tight')

def integrator(fn):
    t0 = 0.
    tf = 100.*fn.p.dc/fn.p.ub  ## seconds
    num_t_steps = 2**10

    dt = float(tf - t0)/num_t_steps
    y0 = fn.p.y0

    igr = ode(func)
    igr.set_integrator('vode')
    igr.set_f_params(fn)
    igr.set_initial_value(y0, t0)

    output = np.zeros((int(num_t_steps+0.1)+2,len(y0)+1),dtype=np.float64)

    output[0,0] = t0
    output[0,1:] = y0
    i = 1
    preperc = -1
    while igr.successful() and igr.t < tf:
        igr.integrate(igr.t+dt)
        output[i,0] = igr.t
        output[i,1:] = igr.y
        i += 1
        '''
        perc = int(100*float(i)/float(num_t_steps))
        if perc > preperc:
            sys.stdout.write("%d %% completed\r" % (perc))
            sys.stdout.flush()
            preperc = perc
    print('')
        '''
    if i < output.shape[0]:
        output = output[:i,:]

    ### convert units
    #output[:,0] *= 1./86400.        ### time to days

    return output

def func(t,v,fn):
    fn._refresh(v)
    return np.array([fn.pwdot,fn.thetadot,fn.phidot])

class Functions():
    def __init__(self,p):
        '''p is an instance of Parameters class'''
        self.p = p
        self._init_model()
        self.ub = self.p.ub ### fixed velocity

    def _thetadot(self):
        arg = self.theta*self.ub/self.p.dc
        self.thetadot = -arg*np.log(arg)

    def _phidot(self):
        self.phidot = self.pwdot*self.beta + self.phidot_p

    def _phidot_p(self):
        self._thetadot()
        self.phidot_p = -self.p.epsilon_p*self.thetadot/self.theta

    def _pwdot_st(self):
        self.pwdot_st = self.p.ch*(self.p.pwinfty + self.p.pwr - 2.*self.pw)

    def _pwdot_dy(self):
        self._phidot_p()
        self._beta()
        self.pwdot_dy = -self.phidot_p/self.beta

    def _pwdot(self):
        self._pwdot_dy()
        self._pwdot_st()
        self.pwdot = self.pwdot_st + self.pwdot_dy
        self._phidot()

    def _effpress(self):
        self.N = 1. - self.pw

    def _beta(self):
        self._effpress()
        self.beta = self.p.epsilon_e*(1.-self.phi)**2/self.N

    def _refresh(self,v):
        self.pw = v[0]
        self.theta = v[1]
        self.phi = v[2]
        self._pwdot()

    def _init_model(self):
        self.beta = self.p.betapi0

        self.pw0 = self.p.pw0
        self.theta0 = self.p.theta0
        self.phi0 = self.p.phi0

        self.p.y0 = [self.pw0,self.theta0,self.phi0]

class Parameters():
    def __init__(self):
        # self.n = 3.

        self.mun = 0.5
        self.dc = 1.e-1

        self.ub_o_ub0 = 10.
        self.ub0 = 1./(365.*86400.) ### units: m/s
        self.ub = self.ub_o_ub0*self.ub0

        self.epsrat = 50.
        self.epsilon_p = 1.e-2 #1.2e-3
        self.epsilon_e = self.epsrat*self.epsilon_p

        # self.kappah = 1.e-7
        # self.hs = 1.e-1

        self.phi0 = 0.1
        self.pw0 = 0.9  ### pw0/pi

        self.pwinfty = self.pw0 ### pwinfty/pi
        self.pwr = self.pw0     ### pwr/pi

        self.s2y = 365.*86400.
        self.y2s = 1./self.s2y

        self.betapi0 = self.epsilon_e*(1.-self.phi0)**2/(1.-self.pw0)
        # self.ch = 1.e-4 #self.kappah/self.hs**2

        self.theta0 = self.dc/self.ub0
        self.ubc = self.ub0
        self.phic = self.phi0

        ### make sure this is last
        #self.y0 = [self.pw0,self.theta0,self.phi0]

if __name__=='__main__':
    main()
