#!/usr/bin/env python3
import sys,os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import ode

from pw_v_time_funs import integrator,func,Functions,Parameters
import tools

mpl,fntsize,labsize = tools.set_mpl(mpl)
##########################################################################

def main():
    p = Parameters()

    fig = plt.figure(figsize=(12,8))
    gs = gridspec.GridSpec(2,1,wspace=0.05,hspace=0.1,height_ratios=[1,1])
    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[1,0])

    diffusion_times_days = [10.,100.,1000.,5000.]
    for th in diffusion_times_days:
        p.ch = 1./(th * 86400.)
        p.epsrat = 50.
        p.epsilon_e = p.epsrat*p.epsilon_p
        Lambda0 = p.epsrat*(1.-p.phi0)**2

        fn = Functions(p)
        output = integrator(fn)

        ax1.plot(output[:,0]*p.ub/(p.dc),output[:,1]/output[0,1],linewidth=2,
                label='$t_h = %d$ days' % th)

    ax1.legend(loc='center right',title='$\epsilon_e/\epsilon_p = %d$ and' % (p.epsrat))
    ax1.set_xlim([0,100]) #output[-1,0]*p.ub/p.dc])
    # ax1.set_xlabel(r'$u_{b_0}t/d_c$',fontsize=labsize)
    ax1.set_xticklabels([])
    ax1.set_ylabel(r'$p_w/p_{w_0}$',fontsize=labsize)


    p.ch = 1./(diffusion_times_days[1] * 86400.)
    eps_ratios = [5.,10.,50.,100.]
    for eps in eps_ratios:
        p.epsrat = eps
        p.epsilon_e = p.epsrat*p.epsilon_p
        Lambda0 = p.epsrat*(1.-p.phi0)**2

        fn = Functions(p)
        output = integrator(fn)

        ax2.plot(output[:,0]*p.ub/(p.dc),output[:,1]/output[0,1],linewidth=2,
                label='$\epsilon_e/\epsilon_p = %d$' % (eps))

    ax2.legend(title='$t_h = %d$ days and' % (1./(86400.*p.ch) + 1.e-6))
    ax2.set_xlim(ax1.get_xlim()) #output[-1,0]*p.ub/p.dc])
    ax2.set_xlabel(r'$u_{b_0}t/d_c$',fontsize=labsize)
    ax2.set_ylabel(r'$p_w/p_{w_0}$',fontsize=labsize)

    plt.savefig('figs/pwpi_step_ub_03.pdf',bbox_inches='tight')

    sys.exit()

    fig = plt.figure(figsize=(12,8))

    for c in [1.1,1.2,1.3,1.4,1.5]:
    #for c in [2.1,2.2,2.3,2.4,2.5]:
    #for c in [3.1,3.2,3.3,3.4,3.5]:
    #for c in [4.1,4.2,4.3,4.4,4.5]:
        p,fn = cases(c)
        output = integrator(fn)
        if int(c) == 1 or int(c) == 2:
            plt.plot(output[:,0]*p.ub/p.dc,output[:,1]/output[0,1],linewidth=2,
                    label='$c_h = %1.0e\ \mathrm{s}^{-1}$' % p.ch)
        elif int(c) == 3:
            plt.plot(output[:,0]*p.ub/p.dc,output[:,1]/output[0,1],linewidth=2,
                    label='$\epsilon_p = %1.1e$' % p.epsilon_p)
        elif int(c) == 4:
            plt.plot(output[:,0]*p.ub/p.dc,output[:,1]/output[0,1],linewidth=2,
                    label='$\epsilon_e/\epsilon_p = %1.1f$' % (p.epsilon_e/p.epsilon_p))

    plt.legend()
    plt.xlim([0,5]) #output[-1,0]*p.ub/p.dc])
    plt.xlabel(r'$u_{b_1}t/d_c$',fontsize=labsize)
    plt.ylabel(r'$p_w/p_{w_0}$',fontsize=labsize)
    plt.savefig('figs/pwpi_step_ub_01.pdf',bbox_inches='tight')

def cases(cnum):
    p = Parameters()
    thetapert = 1.

    if cnum == 0:
        pass
    elif cnum == 1.1:
        p.ch = 1.e-4
    elif cnum == 1.2:
        p.ch = 2.e-4
    elif cnum == 1.3:
        p.ch = 3.e-4
    elif cnum == 1.4:
        p.ch = 6.e-4
    elif cnum == 1.5:
        p.ch = 1.e-3

    elif cnum == 2.1:
        p.ch = 1.e-4
        p.theta0 *= thetapert
    elif cnum == 2.2:
        p.ch = 2.e-4
        p.theta0 *= thetapert
    elif cnum == 2.3:
        p.ch = 3.e-4
        p.theta0 *= thetapert
    elif cnum == 2.4:
        p.ch = 6.e-4
        p.theta0 *= thetapert
    elif cnum == 2.5:
        p.ch = 1.e-3
        p.theta0 *= thetapert

    elif cnum == 3.1:
        p.epsilon_p = 1.e-3
        p.theta0 *= thetapert
    elif cnum == 3.2:
        p.epsilon_p = 1.1e-3
        p.theta0 *= thetapert
    elif cnum == 3.3:
        p.epsilon_p = 1.2e-3
        p.theta0 *= thetapert
    elif cnum == 3.4:
        p.epsilon_p = 1.3e-3
        p.theta0 *= thetapert
    elif cnum == 3.5:
        p.epsilon_p = 1.4e-3
        p.theta0 *= thetapert

    elif cnum == 4.1:
        p.epsilon_e = 4.e-4
        p.theta0 *= thetapert
    elif cnum == 4.2:
        p.epsilon_e = 6.e-4
        p.theta0 *= thetapert
    elif cnum == 4.3:
        p.epsilon_e = 10.e-4
        p.theta0 *= thetapert
    elif cnum == 4.4:
        p.epsilon_e = 12.e-4
        p.theta0 *= thetapert
    elif cnum == 4.5:
        p.epsilon_e = 24.e-4
        p.theta0 *= thetapert

    fn = Functions(p)
    return p,fn

if __name__=='__main__':
    main()
