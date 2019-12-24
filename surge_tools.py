#!/usr/bin/env python3
import sys,os
import numpy as np
from scipy.integrate import solve_ivp
##########################################################################

def surge_mm(a=None,b=None,mun=None,dc=None,ch=None,pw0=None,pwinfty=None,pwr=None,phi0=None,
            epse=None,epsp=None,geoswitch=None,tfyr=100.,stop_mult=10.,pert={'parameter' : 'ub', 'value' : 1.1}):
    p = Parameters()
    p.reconcile_inputs(a,b,mun,dc,ch,pw0,pwinfty,pwr,phi0,epse,epsp,geoswitch)
    p.get_perturbation(pert)
    p.calc_Omega()

    t0 = 0.
    tf = tfyr*86400.*365.
    numsteps = 2**10
    i_numsteps = int(numsteps+0.1)

    dt = (tf - t0)/float(numsteps)
    t_array = np.linspace(t0,tf+dt,i_numsteps)
    t_array[-1] = tf
    t_array_yr = t_array*p.s2y

    p.pwbalance = p.pw0  ### already normalized
    p.alphabalance = p.alpha0
    p.phibalance = p.phi0
    p.mubalance = p.mun

    p.ubnorm_term = (p.alphabalance - p.mubalance*(1.-p.pwbalance))**p.n
    #p.ubref = 2.*p.A*(p.rhoi*p.g)**p.n * p.w**(p.n + 1) / (p.n + 1)
    p.ubbalance_myr = 10.
    p.ubbalance = p.ubbalance_myr/(365.*86400.) #p.ubref * p.ubnorm_term
    p.ubc = p.ubbalance

    p.thetabalance = p.dc / p.ubc
    p.thetac = p.dc / p.ubc

    #alpha = p.alphabalance

    p.ubinit = p.ubpert * p.ubc
    p.thetainit = p.thetapert * p.thetabalance
    p.thickinit = p.h
    p.p_over0 = p.rhoi * p.g * p.thickinit
    p.pwinit = p.pwpert * p.pwbalance * p.p_over0
    p.phiinit = p.phibalance
    p.alphainit = p.alphabalance

    #print(p.ubinit/p.ubbalance)
    if p.pwinit/p.p_over0 > 1.:
        raise ValueError('Pore water pressure exceeds overburden pressure')

    y0 = [p.ubinit,p.thetainit,p.pwinit,p.phiinit,p.thickinit,p.alphainit]

    ## define event to terminate integration if ub > 10.*ub0
    def is_surge_p(t,y,p):
        return stop_mult*p.ubinit - y[0]
    is_surge = lambda t,y: is_surge_p(t,y,p)
    is_surge.terminal = True
    is_surge.direction = -1

    otv = solve_ivp(lambda t,y: surge_solver(t,y,p),[t0,tf],y0,method='Radau',
                         dense_output=True,max_step=dt,events=is_surge)
    return p,otv

def surge_solver(t,y,p):
    eps = p.machine_eps

    ub = np.max([eps, y[0]])
    theta = np.max([eps, y[1]])
    pw_Pa = np.max([eps, y[2]])
    phi = np.max([eps, y[3]])
    h = np.max([eps, y[4]])
    alpha = np.max([eps, y[5]])
    p_over = p.rhoi*p.g*h

    ### nondimensional parameters
    pw = np.min([pw_Pa/p_over, 1.-eps])

    N = 1. - pw
    Lambda = p.epsilon_e * (1. - phi)**2 / p.epsilon_p  ### dimensionless
    beta = p.epsilon_p * Lambda / N

    ### state
    tdotarg = ub * theta / p.dc
    thetadot = -tdotarg * np.log(tdotarg)

    ### water pressure
    pwdot_st = p.ch * (p.pwinfty + p.pwr - 2.*pw)
    pwdot_dy = thetadot * N / (theta * Lambda)
    pwdot = p_over * (pwdot_st + pwdot_dy)

    ### rate and state friction coef
    mu = p.mun + p.a * np.log(ub/p.ubc) + p.b * np.log(theta*p.ubc/p.dc)

    ### porosity
    phidot_e = pwdot * beta / p_over
    phidot_p = -p.epsilon_p * thetadot / theta
    phidot = phidot_e + phidot_p

    ### glacier geometry
    hdot = p.geoswitch * p.zeta * alpha * (p.ubbalance - ub)
    alphadot = hdot * alpha / h

    ### acceleration
    ubdot = p.n * ub * ((alphadot - mu*(pw*hdot/h - pwdot/p_over) -
                    N*p.b*thetadot / theta) / (alpha + (p.a*p.n - mu)*N))

    return [ubdot, thetadot, pwdot, phidot, hdot, alphadot]

class Parameters():
    def __init__(self):
        ### material properties of ice
        self.n = 3.
        self.A = 2.4e-24
        self.rhoi = 900.
        self.g = 9.81

        ### glacier geometry
        self.h = 300.
        self.w = 800.
        self.alpha0 = 5.e-2

        ### rate and state parameters
        self.mun = 0.5
        self.dc = 1.e-1
        self.ab = 0.9
        self.b = 0.015
        self.a = self.b*self.ab

        ### ratio of depth averaged velocity to surface velocity (0.8 <= zeta <= 1)
        self.zeta = 1.

        ### till properties
        self.epsrat = 50.
        self.epsilon_p = 1.e-3
        self.epsilon_e = self.epsrat*self.epsilon_p
        self.ch = 1.e-6 #5.e-7 #self.kappah/self.hs**2

        self.phi0 = 0.1
        self.pw0 = 0.92  ### pw0/pi

        self.pwinfty = self.pw0 ### pwinfty/pi
        self.pwr = self.pw0     ### pwr/pi

        ### init perturbations
        self.ubpert = 1.
        self.thetapert = 1.
        self.pwpert = 1.

        ### special
        self.geoswitch = 1.

        ### misc
        self.y2s = 365.*86400.
        self.s2y = 1./self.y2s
        self.machine_eps = np.finfo(np.float64).eps

    def reconcile_inputs(self,a,b,mun,dc,ch,pw0,pwinfty,pwr,phi0,epse,epsp,geoswitch):
        if a is not None:
            self.a = a
        if b is not None:
            self.b = b
        if mun is not None:
            self.mun = mun
        if dc is not None:
            self.d_c = dc
        if ch is not None:
            self.ch = ch
        if pw0 is not None:
            self.pw0 = pw0
        if pwinfty is not None:
            self.pwinfty = pwinfty
        if pwr is not None:
            self.pwr = pwr
        if phi0 is not None:
            self.phi0 = phi0
        if epse is not None:
            self.epsilon_e = epse
        if epsp is not None:
            self.epsilon_p = epsp
        if geoswitch is not None:
            self.geoswitch = geoswitch

    def get_perturbation(self,pert):
        if pert['parameter'].lower() == 'ub':
            self.ubpert = pert['value']
        elif pert['parameter'].lower() == 'theta':
            self.thetapert = pert['value']
        elif pert['parameter'].lower() == 'pw':
            self.pwpert = pert['value']
        else:
            raise ValueError('unrecognized perturbation parameter %s' % pert['parameter'])

    def calc_Omega(self):
        P = 1.-self.pwinfty
        self.Omega = ((self.a - self.b)*P) / (self.alpha0 - self.mun*P)

if __name__=='__main__':
    main()
