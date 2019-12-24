#!/usr/bin/env python3
'''
Explore the parameter space for surge model
'''
import sys,os
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import lambertw
from scipy.optimize import minimize,Bounds

import tools
import surge_tools
import concurrent.futures

mpl,fntsize,labsize = tools.set_mpl(mpl)
legend_title = 'applies to all panels'
legend_title_font = 12
##########################################

def main():
    #qdp1()
    #stack_plot()

    explore_b()
    # explore_b_ub_vs_others()
    explore_b_parameterspace_1()

    # b_vs_ch()
    plot_b_vs_ch()
    # b_vs_ch_para_write()

    explore_b_nothin()
    # explore_b_nothin_Lambda()
    #explore_initu_b_nothin()
    #explore_eps_nothin()
    #explore_eps()

    #plot_lamberw()
    #explore_ubss()

#################################################
def explore_ubss():
    fname='figs/ub_v_t_b/ubss_vs_Omega_01.pdf'

    p = surge_tools.Parameters()

    p.a = 0.013
    p.b = 0.05
    p.calc_Omega()

    print(p.Omega)
    fun_ubss = lambda u,n,Omega: np.abs(u**(1./n) + Omega*np.log(u) - 1.)
        #return val

    # ubss0 = 2.
    # bounds = Bounds(lb=ubss0,ub=1.e99)
    # mout = minimize(fun_ubss,[ubss0],args=(p.n,p.Omega),method='L-BFGS-B',bounds=bounds)
    #
    # print(mout)

    # U = np.linspace(0.01,3.,1000)
    # y = U**(1./p.n) + p.Omega*np.log(U) - 1.
    #
    # fig = plt.figure()
    # plt.plot(U,y)
    # plt.show()

#################################################
def fun_b_vs_ch(iter):
    r = iter[0]
    i = iter[1]
    j = iter[2]
    difftime = iter[3]
    b = iter[4]
    a = iter[5]
    pert = iter[6]

    ch = 1./(difftime*86400.)

    p,otv = surge_tools.surge_mm(a=a,b=b,ch=ch,geoswitch=1.,pert=pert)

    ubfinal = otv.y[0][-1]/p.ubbalance
    ubdotfinal = (otv.y[0][-1] - otv.y[0][-2]) / (otv.t[-1] - otv.t[-2])
    ubmax = np.max(otv.y[0])/p.ubbalance

    return [i,j,ubfinal,ubdotfinal,ubmax]

def b_vs_ch_para_write():
    fname='figs/ub_v_t_b/data/ub_vs_ch_phasespace_%s.bin'

    print('unfinished...not working')
    return

    pert = {'parameter' : 'ub', 'value' : 1.1}

    a = 0.013

    rows = 20
    cols = rows
    diffusion_times_days_min = 1.e2
    diffusion_times_days_max = 5.e3
    diffusion_times_days = np.linspace(diffusion_times_days_min,diffusion_times_days_max,rows)

    bvals = np.linspace(0.01,0.05,cols)

    ubfinal = np.empty((len(diffusion_times_days),len(bvals)),dtype=np.float64)
    ubdotfinal = np.empty_like(ubfinal)
    ubmax = np.empty_like(ubfinal)

    iterlist = [None]*(rows*cols)
    r_range = np.arange(rows*cols)
    perc = -1
    for r in r_range:
        nperc = np.int(100.*r / (rows*cols))
        if nperc > perc:
            perc = nperc
            if perc > 90:
                print('%d percent complete' % (perc))
            elif nperc % 10 == 0:
                print('%d percent complete' % (perc))

        i = r // cols
        j = r % cols

        difftime = diffusion_times_days[i]
        b = bvals[j]

        iterlist[r] = [r,i,j,difftime,b,a,pert]

    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        result = executor.map()

def b_vs_ch():
    fname='figs/ub_v_t_b/ub_vs_ch_phasespace_05_%s.pdf'
    dataname = fname.rstrip('.pdf') + '.bin'

    pert = {'parameter' : 'ub', 'value' : 1.1}

    a = 0.013

    diffusion_times_days_min = 1.e2
    diffusion_times_days_max = 5.e3
    diffusion_times_days = np.linspace(diffusion_times_days_min,diffusion_times_days_max,10)

    bvals = np.linspace(0.01,0.05,10)

    ubfinal = np.empty((len(diffusion_times_days),len(bvals)),dtype=np.float64)
    ubdotfinal = np.empty_like(ubfinal)
    ubmax = np.empty_like(ubfinal)

    rows = len(diffusion_times_days)
    cols = len(bvals)
    perc = -1

    iterlist = [None]*(rows*cols)
    for r in range(rows*cols):
        nperc = np.int(100.*r / (rows*cols))
        if nperc > perc:
            perc = nperc
            # if perc > 90:
            #     print('%d percent complete' % (perc))
            # elif nperc % 10 == 0:
            #     print('%d percent complete' % (perc))

        i = r // cols
        j = r % cols

        difftime = diffusion_times_days[i]
        b = bvals[j]

        iterlist[r] = [r,i,j,difftime,b,a,pert]

        ch = 1./(difftime*86400.)

        p,otv = surge_tools.surge_mm(a=a,b=b,ch=ch,geoswitch=1.,pert=pert)

        print(len(otv.y[0]))

        # ubfinal[i,j] = otv.y[0][-1]/p.ubbalance
        # ubdotfinal[i,j] = (otv.y[0][-1] - otv.y[0][-2]) / (otv.t[-1] - otv.t[-2])
        # ubmax[i,j] = np.max(otv.y[0])/p.ubbalance

    # with open(dataname % '%dx%d_%s' % (rows,cols,'ubmax'),'w') as fid:
    #     ubmax.flatten().astype(np.float32).tofile(fid)
    # with open(dataname % '%dx%d_%s' % (rows,cols,'ubfinal'),'w') as fid:
    #     ubfinal.flatten().astype(np.float32).tofile(fid)
    # with open(dataname % '%dx%d_%s' % (rows,cols,'ubdotfinal'),'w') as fid:
    #     ubdotfinal.flatten().astype(np.float32).tofile(fid)

def plot_b_vs_ch():
    # fname='figs/ub_v_t_b/ub_vs_ch_phasespace_07_%s.pdf'
    fname='figs/ub_v_t_b/ub_vs_ch_phasespace_epsrat_50_%s.pdf'
    dataname = fname.rstrip('.pdf') + '.bin'
    rows = 250
    cols = rows
    diffusion_times_days_min = 1.e2
    diffusion_times_days_max = 5.e3
    diffusion_times_days = np.linspace(diffusion_times_days_min,diffusion_times_days_max,rows)
    bvals = np.linspace(0.01,0.05,cols)
    p = surge_tools.Parameters()
    Lambda_hat = p.epsilon_e * (1. - p.phi0)**2 / p.epsilon_p
    p.ubbalance_myr = 10.

    with open(dataname % '%dx%d_%s' % (rows,cols,'ubmax'),'r') as fid:
        ubmax = np.fromfile(fid,dtype=np.float32).reshape(-1,cols)
        if ubmax.shape[0] != rows:
            raise ValueError('rows in ubmax do not match given size...check values')
    with open(dataname % '%dx%d_%s' % (rows,cols,'ubfinal'),'r') as fid:
        ubfinal = np.fromfile(fid,dtype=np.float32).reshape(-1,cols)
        if ubfinal.shape[0] != rows:
            raise ValueError('rows in ubfinal do not match given size...check values')
    with open(dataname % '%dx%d_%s' % (rows,cols,'ubdotfinal'),'r') as fid:
        ubdotfinal = np.fromfile(fid,dtype=np.float32).reshape(-1,cols)
        if ubdotfinal.shape[0] != rows:
            raise ValueError('rows in ubdotfinal do not match given size...check values')

    cmaplevels = np.arange(1.,10.1,0.1)
    fig = plt.figure()
    cmap_special_high = cm.get_cmap('magma',len(cmaplevels))
    cmap_special_low = cm.get_cmap('bone_r',10)
    cmap_special_ndarray = np.vstack((cm.binary(np.linspace(0.1, 0.9, 256./11.)),cm.magma(np.linspace(0.05, 1, 10.*256./11.))))
    cmap_special = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', cmap_special_ndarray)

    # cs = plt.contourf(bvals,diffusion_times_days*p.ubbalance_myr/(p.dc*365.),ubfinal,np.linspace(0,1.,10),cmap=cm.bone_r,extend='max')
    # cs = plt.contourf(bvals,diffusion_times_days*p.ubbalance_myr/(p.dc*365.),ubfinal,cmaplevels,cmap=cm.magma,extend='max')
    # c2 = plt.contour(cs,levels=cs.levels[::1],colors='0.4',linewidths=(0.5,))

    # *p.ubbalance_myr/(Lambda_hat*p.dc*365.)
    cs = plt.contourf(bvals,diffusion_times_days,ubfinal,np.arange(0.,10.1,0.1),cmap=cmap_special,extend='max')
    cbar = plt.colorbar(cs,ticks=[0,1,2,4,6,8,10])

    special_b = [0.022,0.024,0.026,0.028,0.030]
    colours = ['tab:blue','tab:orange','tab:red','tab:green','tab:olive'][::-1]
    for bv,thv,col in zip(special_b,[2600]*len(special_b),colours):
        plt.scatter(bv,thv,c=col,edgecolors='k',linewidths=1,s=100)

    labsize2 = labsize + 5

    cbar.ax.set_ylabel(r'$u_{b_{final}}/\hat{u}_{b_0}$',fontsize=labsize2)
    plt.xlabel(r'$b$',fontsize=labsize2)
    plt.ylabel(r'$t_h$ (days)',fontsize=labsize2)

    plt.savefig(fname % 'final',bbox_inches='tight')

    fig = plt.figure()
    cs = plt.contourf(bvals,diffusion_times_days,ubmax,cmaplevels,cmap=cm.magma,extend='max')
    # c2 = plt.contour(cs,levels=cs.levels[::1],colors='0.4',linewidths=(0.5,))
    cbar = plt.colorbar(cs,ticks=[0,1,2,4,6,8,10])

    cbar.ax.set_ylabel(r'$u_{b_{max}}/\hat{u}_{b_0}$',fontsize=labsize2)
    plt.xlabel(r'$b$',fontsize=labsize2)
    plt.ylabel(r'$t_h$ (days)',fontsize=labsize2)

    plt.savefig(fname % 'max',bbox_inches='tight')
    #plt.show()

#################################################
def explore_eps():
    fname='figs/ub_v_t_b/ub_vs_time_eps_02.pdf'

    pert = {'parameter' : 'ub',
            'value' : 1.1}

    fig = plt.figure(figsize=(12,14))
    gs = gridspec.GridSpec(4,1,wspace=0.05,hspace=0.1,height_ratios=[2,2,1,1])#,width_ratios=[50,1])
    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[1,0])
    ax3 = plt.subplot(gs[2,0])
    ax4 = plt.subplot(gs[3,0])

    for erat,col in zip([10000,1000,50,25,10,1e-2],
                    ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown']):
        eps_p = 1.e-4
        eps_e = erat*eps_p
        p,otv = surge_tools.surge_mm(epse=eps_e,epsp=eps_p,pert=pert)

        ax1.plot(otv.t*p.s2y, otv.y[0]/p.ubbalance,color=col,label='$\epsilon_e/\epsilon_p = %1.2e$' % erat)
        ax2.plot(otv.t*p.s2y, otv.y[1]/p.thetabalance,color=col)
        ax3.plot(otv.t*p.s2y, otv.y[2]/p.p_over0,color=col)
        ax3.plot(otv.t*p.s2y, otv.y[3],'--',color=col)
        ax4.plot(otv.t*p.s2y, otv.y[4]/p.h,color=col)

    ax1.legend()

    ax1.plot([0,100],[1,1],'0.4',linestyle='--')

    ax1.set_ylabel('$u_b/u_b^*$',fontsize=labsize)
    ax2.set_ylabel(r'$\theta u_b^*/d_c$',fontsize=labsize)
    ax3.set_ylabel(r'$p_w/p_{i0}$, $\phi$')#,fontsize=labsize)
    ax4.set_ylabel(r'$h/h_0$, $\alpha/\alpha_0$')#,fontsize=labsize)

    maxtime = 100.
    ax1.set_xlim([0,maxtime])
    ax2.set_xlim([0,maxtime])
    ax3.set_xlim([0,maxtime])
    ax4.set_xlim([0,maxtime])

    ax1.set_ylim([0,10])

    ax1.get_xaxis().set_ticklabels([])
    ax2.get_xaxis().set_ticklabels([])
    ax3.get_xaxis().set_ticklabels([])

    ax4.set_xlabel('Time (years)')

    #if title is not None:
    #    ax1.set_title(title,fontsize=fntsize-5)
    ax1.set_title('No thinning',fontsize=fntsize-5)
    plt.savefig(fname,bbox_inches='tight')

#################################################
def explore_eps_nothin():
    fname='figs/ub_v_t_b/ub_vs_time_eps_nothin_01.pdf'

    pert = {'parameter' : 'ub',
            'value' : 1.1}

    fig = plt.figure(figsize=(12,14))
    gs = gridspec.GridSpec(5,1,wspace=0.05,hspace=0.1,height_ratios=[2,1,1,1,1])#,width_ratios=[50,1])
    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[1,0])
    ax3 = plt.subplot(gs[2,0])
    ax4 = plt.subplot(gs[3,0])
    ax5 = plt.subplot(gs[4,0])

    for erat,col in zip([1.e4,1.e3,1.e2,1.e1,1.e0],
                    ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']):
        print(erat)
        eps_p = 1.e-4
        eps_e = erat*eps_p
        a = 0.013 #a = None #0.9*b
        b = 0.04
        ch = 1./(100.*86400.)
        p,otv = surge_tools.surge_mm(a=a,b=b,ch=ch,epse=eps_e,epsp=eps_p,geoswitch=0.,tfyr=5.,pert=pert)

        ax1.plot(otv.t*p.s2y, otv.y[0]/p.ubbalance,color=col,label='$\epsilon_e/\epsilon_p = %1.2e$' % erat)
        ax2.plot(otv.t*p.s2y, otv.y[1]/p.thetabalance,color=col)
        ax3.plot(otv.t*p.s2y, otv.y[2]/p.p_over0,color=col)
        ax3.plot(otv.t*p.s2y, otv.y[3],'--',color=col)
        #ax4.plot(otv.t*p.s2y, otv.y[4]/p.h,color=col)

    ax1.legend()

    ax1.plot([0,100],[1,1],'0.4',linestyle='--')

    ax1.set_ylabel('$u_b/\hat{u}_b$',fontsize=labsize)
    ax2.set_ylabel(r'$\theta \hat{u}_b/d_c$',fontsize=labsize)
    ax3.set_ylabel(r'$p_w/p_{i}$')#,fontsize=labsize)
    #ax4.set_ylabel(r'$h/h_0$, $\alpha/\alpha_0$')#,fontsize=labsize)

    maxtime = 1.
    ax1.set_xlim([0,maxtime])
    ax2.set_xlim([0,maxtime])
    ax3.set_xlim([0,maxtime])
    #ax4.set_xlim([0,maxtime])

    #ax1.set_ylim([0,10])
    #ax2.set_ylim([0,1])
    ax3.set_ylim([0.915,0.921])

    ax1.get_xaxis().set_ticklabels([])
    ax2.get_xaxis().set_ticklabels([])
    #ax3.get_xaxis().set_ticklabels([])

    ax3.set_xlabel('Time (years)')

    #if title is not None:
    #    ax1.set_title(title,fontsize=fntsize-5)
    #ax1.set_title('No thinning',fontsize=fntsize-5)
    plt.savefig(fname,bbox_inches='tight')

#################################################
def explore_b_nothin(geoswitch=0.,fname='figs/ub_v_t_b/ub_vs_time_b_nothin_03.pdf',
    pert = {'parameter' : 'ub','value' : 1.1},diffusion_times_days=[10,100],tfyr=10.):
    s2yr = 1./(86400.*365.)

    if len(diffusion_times_days) != 2:
        raise ValueError('diffusion_times_days must be list of length 2')

    fig = plt.figure(figsize=(12,14))
    gs = gridspec.GridSpec(6,1,wspace=0.05,hspace=0.1,height_ratios=[2,1,1,1,1,1])#,width_ratios=[50,1])
    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[1,0])
    ax3 = plt.subplot(gs[2,0])
    ax4 = plt.subplot(gs[3,0])
    ax5 = plt.subplot(gs[4,0])
    ax6 = plt.subplot(gs[5,0])

    #ax1.plot([0,100],[1,1],'k--',linewidth=1.5)

    #for b,col in zip([0.14,0.13,0.12,0.10,0.08,0.0],
    ax1.plot([0,10000],[1.1,1.1],'tab:gray',linewidth=1)
    ax4.plot([0,10000],[100,100],'tab:gray',linewidth=1)

    yoffset = 100.
    for th,col in zip(diffusion_times_days,['tab:red','tab:blue']):
        ax4.plot([-100,-1],[yoffset,yoffset],color=col,linewidth=2.5,label='$t_h = %d$ days' % th)
    leg = ax4.legend(loc='lower right',title='all panels')
    leg.set_title(legend_title + '\n' + 'and line types',prop={'size':legend_title_font})

    yoffset = 0.

    bvals = [0.05,0.03,0.01,0.0]
    for singlecol in ['tab:gray','tab:blue']:
        lt1 = True
        for b,col,palp,lw in zip(bvals,[singlecol]*len(bvals),np.linspace(0.3,1,len(bvals))[::-1],np.linspace(2,3,len(bvals))[::-1]):
                    #['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:gray']):
            a = 0.013
            ch = 1./(diffusion_times_days[1]*86400.)
            #epsrat = 40./(1.-0.1)**2
            p,otv = surge_tools.surge_mm(a=a,b=b,ch=ch,geoswitch=geoswitch,tfyr=tfyr,pert=pert)
            if p.a > p.b:
                if lt1:
                    linestyle = '--'
                    lt1 = False
                else:
                    linestyle = '--'
            else:
                linestyle = '-'

            print(p.b, p.a - p.b)
            #ax3.plot([s2yr/p.ch,s2yr/p.ch],[0,1],'b:',linewidth=1.5)
            ub = otv.y[0]
            theta = otv.y[1]
            mu = p.mun + p.a *np.log(ub/p.ubc) + p.b * np.log(theta/p.thetac)

            p_over = p.rhoi*p.g*otv.y[4]
            pw_Pa = otv.y[2]
            Neff = p_over - pw_Pa
            taub = Neff*mu

            taub_hat = p.mun*(p.p_over0 - p.pwinit)

            Lambda_hat = p.epsilon_e * (1. - p.phiinit)**2 / p.epsilon_p
            if p.b > np.finfo(np.float32).eps:
                lterm = p.mun/(p.b*Lambda_hat)
                ublabel = '$b = %1.3f$, $a-b = %1.3f$' % (p.b,p.a - p.b)
            else:
                ublabel = '$b = %1.3f$, $a-b = %1.3f$' % (p.b,p.a - p.b)

            nrmtime = 1./(Lambda_hat*p.thetac) # p.ch
            indu = np.argmin(np.abs(otv.t*nrmtime - 100.))
            ubnrm = ub[indu]/p.ubbalance
            muss = p.mun + (p.a-p.b)*np.log(ubnrm)
            ubss_test = ubnrm**(1./p.n) + p.Omega*np.log(ubnrm)
            theta_test = otv.y[1]
            print(ubss_test,p.Omega,ubnrm,muss-mu[indu],theta_test[indu]-theta_test[indu-1],'%s\n' % col)

            ax1.plot(otv.t*nrmtime, otv.y[0]/p.ubbalance + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw,label=ublabel)
            ax2.plot(otv.t*nrmtime, otv.y[1]/p.thetabalance + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
            ax3.plot(otv.t*nrmtime, 100.*otv.y[3]/p.phiinit + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
            ax4.plot(otv.t*nrmtime, 100.*(otv.y[2]/p.pwinit) + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
            #ax3.plot(otv.t*nrmtime, otv.y[3],'--',color=col)
            ax5.plot(otv.t*nrmtime, mu/p.mun + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
            ax6.plot(otv.t*nrmtime, taub/taub_hat + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)


            #ax4.plot([0.,1000.],[muss/p.mun,muss/p.mun],color=col,linestyle=':')

            #if yoffset < 1.e-6 and p.a > p.b:
            #    print(otv.y[2][-1]/p.pwinit-1.)

        if singlecol == 'tab:gray':
            leg = ax1.legend(loc='center right')
            leg.set_title(legend_title + ' and line colors',prop={'size':legend_title_font})
        yoffset = 0.

    singlecol = 'tab:red'
    lt1 = True
    for b,col,palp,lw in zip(bvals,[singlecol]*len(bvals),np.linspace(0.3,1,len(bvals))[::-1],np.linspace(2,3,len(bvals))[::-1]):
        a = 0.013 #0.9*b
        ch = 1./(diffusion_times_days[0]*86400.)
        #epsrat = 40./(1.-0.1)**2
        p,otv = surge_tools.surge_mm(a=a,b=b,ch=ch,geoswitch=0.,tfyr=tfyr,pert=pert)
        if p.a > p.b:
            if lt1:
                linestyle = '--'
                lt1 = False
            else:
                linestyle = '--'
        else:
            linestyle = '-'

        print(p.b, p.a - p.b, p.ubc*86400.*365.)
        #ax3.plot([s2yr/p.ch,s2yr/p.ch],[0,1],'b:',linewidth=1.5)

        ub = otv.y[0]
        theta = otv.y[1]
        mu = p.mun + p.a *np.log(ub/p.ubc) + p.b * np.log(theta/p.thetac)

        p_over = p.rhoi*p.g*otv.y[4]
        pw_Pa = otv.y[2]
        Neff = p_over - pw_Pa
        taub = Neff*mu

        taub_hat = p.mun*(p.p_over0 - p.pwinit)

        Lambda_hat = p.epsrat*(1.-p.phi0)**2

        nrmtime = 1./(Lambda_hat*p.thetac) # p.ch
        indu = np.argmin(np.abs(otv.t*nrmtime - 1000.))
        ubnrm = ub[indu]/p.ubbalance
        muss = p.mun + (p.a-p.b)*np.log(ubnrm)
        ubss_test = ubnrm**(1./p.n) + p.Omega*np.log(ubnrm)
        theta_test = otv.y[1]
        print(ubss_test,p.Omega,ubnrm,muss-mu[indu],(theta_test[indu]-theta_test[indu-1])/(otv.t[indu] - otv.t[indu-1]),'%s\n' % col)

        ax1.plot(otv.t*nrmtime, otv.y[0]/p.ubbalance,color=col,alpha=palp,linestyle=linestyle,linewidth=lw,label='$b = %1.2e$' % b)
        ax2.plot(otv.t*nrmtime, otv.y[1]/p.thetabalance,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
        ax3.plot(otv.t*nrmtime, 100.*otv.y[3]/p.phiinit + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
        ax4.plot(otv.t*nrmtime, 100.*(otv.y[2]/p.pwinit),color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
        #ax3.plot(otv.t*nrmtime, otv.y[3],'--',color=col)
        ax5.plot(otv.t*nrmtime, mu/p.mun,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
        ax6.plot(otv.t*nrmtime, taub/taub_hat + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)

    ax1.set_ylabel('$u_b/\hat{u}_{b_0}$',fontsize=labsize)
    ax2.set_ylabel(r'$\theta/ \hat{\theta}_0$',fontsize=labsize)
    ax3.set_ylabel(r'$\phi/\hat{\phi}_0\ (\%)$',fontsize=labsize)
    ax4.set_ylabel(r'$p_w/\hat{p}_{w_0}\ (\%)$',fontsize=labsize) #, $\phi$
    ax5.set_ylabel(r'$\mu/\mu_n$',fontsize=labsize)#,fontsize=labsize)
    ax6.set_ylabel(r'$\tau_t/\hat{\tau}_{t_0}$',fontsize=labsize)

    print('Lambda0 = %f' % Lambda_hat)
    maxtime = 4. #max(diffusion_times_days)*Lambda0 ###np.max(otv.t*p.ch)/3.
    ax1.set_xlim([0,maxtime])
    ax2.set_xlim([0,maxtime])
    ax3.set_xlim([0,maxtime])
    ax4.set_xlim([0,maxtime])
    ax5.set_xlim([0,maxtime])
    ax6.set_xlim([0,maxtime])

    #ax1.set_ylim([1,2])
    #ax2.set_ylim([0.5,1])
    #ax3.set_ylim([-0.002,0.06])
    ##ax4.set_ylim([-0.01,0.04])
    #ax4.set_ylim([1.-0.06,1.01])
    #ax5.set_ylim([0.94,1.01])
    ##ax3.set_ylim([0.919,0.9201])
    ##ax3.set_ylim([100*(0.997-1.),100*(1.00002-1.)])

    ax1.get_xaxis().set_ticklabels([])
    ax2.get_xaxis().set_ticklabels([])
    ax3.get_xaxis().set_ticklabels([])
    ax4.get_xaxis().set_ticklabels([])
    ax5.get_xaxis().set_ticklabels([])

    ax6.set_xlabel(r'$t\hat{u}_{b_0}\epsilon_p/(d_c\epsilon_e)$',fontsize=labsize)

    #if title is not None:
    #ax1.set_title('No thinning',fontsize=fntsize-5)
    plt.savefig(fname,bbox_inches='tight')

#################################################
def explore_b_nothin_Lambda(geoswitch=0.,fname='figs/ub_v_t_b/ub_vs_time_b_nothin_Lambda_01.pdf',
    pert = {'parameter' : 'ub','value' : 1.1},diffusion_times_days=[100,100],tfyr=10.):
    s2yr = 1./(86400.*365.)

    epsratios = [5,50]

    if len(diffusion_times_days) != 2:
        raise ValueError('diffusion_times_days must be list of length 2')

    fig = plt.figure(figsize=(12,14))
    gs = gridspec.GridSpec(6,1,wspace=0.05,hspace=0.1,height_ratios=[2,1,1,1,1,1])#,width_ratios=[50,1])
    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[1,0])
    ax3 = plt.subplot(gs[2,0])
    ax4 = plt.subplot(gs[3,0])
    ax5 = plt.subplot(gs[4,0])
    ax6 = plt.subplot(gs[5,0])

    #ax1.plot([0,100],[1,1],'k--',linewidth=1.5)

    #for b,col in zip([0.14,0.13,0.12,0.10,0.08,0.0],
    ax1.plot([0,1000],[1.1,1.1],'tab:gray',linewidth=1)
    ax4.plot([0,1000],[100,100],'tab:gray',linewidth=1)

    yoffset = 100.
    for th,col in zip(epsratios,['tab:red','tab:blue']):
        Lambda = th*(1.-0.1)**2
        ax4.plot([-100,-1],[yoffset,yoffset],color=col,linewidth=2.5,label='$\epsilon_e/\epsilon_p = %d$' % th)
    leg = ax4.legend(loc='lower right')
    leg.set_title(legend_title + '\n' + 'and line types',prop={'size':legend_title_font})

    yoffset = 0.

    bvals = [0.05,0.03,0.01,0.0]
    for singlecol in ['tab:gray','tab:blue']:
        lt1 = True
        for b,col,palp,lw in zip(bvals,[singlecol]*len(bvals),np.linspace(0.3,1,len(bvals))[::-1],np.linspace(2,3,len(bvals))[::-1]):
                    #['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:gray']):
            a = 0.013
            ch = 1./(diffusion_times_days[1]*86400.)
            epsrat = epsratios[1]
            epsp = 1.e-4
            epse = epsrat*epsp
            p,otv = surge_tools.surge_mm(a=a,b=b,ch=ch,epse=epse,epsp=epsp,geoswitch=geoswitch,tfyr=tfyr,pert=pert)

            if p.a > p.b:
                if lt1:
                    linestyle = '--'
                    lt1 = False
                else:
                    linestyle = '--'
            else:
                linestyle = '-'

            print(p.b, p.a - p.b)
            #ax3.plot([s2yr/p.ch,s2yr/p.ch],[0,1],'b:',linewidth=1.5)
            ub = otv.y[0]
            theta = otv.y[1]
            mu = p.mun + p.a *np.log(ub/p.ubc) + p.b * np.log(theta/p.thetac)

            p_over = p.rhoi*p.g*otv.y[4]
            pw_Pa = otv.y[2]
            Neff = p_over - pw_Pa
            taub = Neff*mu

            taub_hat = p.mun*(p.p_over0 - p.pwinit)

            Lambda_hat = p.epsilon_e * (1. - p.phiinit)**2 / p.epsilon_p
            if p.b > np.finfo(np.float32).eps:
                lterm = p.mun/(p.b*Lambda_hat)
                ublabel = '$b = %1.3f$, $a-b = %1.3f$' % (p.b,p.a - p.b)
            else:
                ublabel = '$b = %1.3f$, $a-b = %1.3f$' % (p.b,p.a - p.b)

            nrmtime = 1./p.thetac # p.ch
            indu = np.argmin(np.abs(otv.t*nrmtime - 100.))
            ubnrm = ub[indu]/p.ubbalance
            muss = p.mun + (p.a-p.b)*np.log(ubnrm)
            ubss_test = ubnrm**(1./p.n) + p.Omega*np.log(ubnrm)
            theta_test = otv.y[1]
            print(ubss_test,p.Omega,ubnrm,muss-mu[indu],theta_test[indu]-theta_test[indu-1],'%s\n' % col)

            ax1.plot(otv.t*nrmtime, otv.y[0]/p.ubbalance + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw,label=ublabel)
            ax2.plot(otv.t*nrmtime, otv.y[1]/p.thetabalance + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
            ax3.plot(otv.t*nrmtime, 100.*otv.y[3]/p.phiinit + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
            ax4.plot(otv.t*nrmtime, 100.*(otv.y[2]/p.pwinit) + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
            #ax3.plot(otv.t*nrmtime, otv.y[3],'--',color=col)
            ax5.plot(otv.t*nrmtime, mu/p.mun + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
            ax6.plot(otv.t*nrmtime, taub/taub_hat + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)


            #ax4.plot([0.,1000.],[muss/p.mun,muss/p.mun],color=col,linestyle=':')

            #if yoffset < 1.e-6 and p.a > p.b:
            #    print(otv.y[2][-1]/p.pwinit-1.)

        if singlecol == 'tab:gray':
            leg = ax1.legend(loc='center right')
            leg.set_title(legend_title + ' and line colors',prop={'size':legend_title_font})
        yoffset = 0.

    singlecol = 'tab:red'
    lt1 = True
    for b,col,palp,lw in zip(bvals,[singlecol]*len(bvals),np.linspace(0.3,1,len(bvals))[::-1],np.linspace(2,3,len(bvals))[::-1]):
        a = 0.013 #0.9*b
        ch = 1./(diffusion_times_days[0]*86400.)
        epsrat = epsratios[0]
        epsp = 1.e-4
        epse = epsrat*epsp
        p,otv = surge_tools.surge_mm(a=a,b=b,ch=ch,epse=epse,epsp=epsp,geoswitch=geoswitch,tfyr=tfyr,pert=pert)

        if p.a > p.b:
            if lt1:
                linestyle = '--'
                lt1 = False
            else:
                linestyle = '--'
        else:
            linestyle = '-'

        print(p.b, p.a - p.b, p.ubc*86400.*365.)
        #ax3.plot([s2yr/p.ch,s2yr/p.ch],[0,1],'b:',linewidth=1.5)

        ub = otv.y[0]
        theta = otv.y[1]
        mu = p.mun + p.a *np.log(ub/p.ubc) + p.b * np.log(theta/p.thetac)

        p_over = p.rhoi*p.g*otv.y[4]
        pw_Pa = otv.y[2]
        Neff = p_over - pw_Pa
        taub = Neff*mu

        taub_hat = p.mun*(p.p_over0 - p.pwinit)

        nrmtime = 1./p.thetac # p.ch
        indu = np.argmin(np.abs(otv.t*nrmtime - 1000.))
        ubnrm = ub[indu]/p.ubbalance
        muss = p.mun + (p.a-p.b)*np.log(ubnrm)
        ubss_test = ubnrm**(1./p.n) + p.Omega*np.log(ubnrm)
        theta_test = otv.y[1]
        print(ubss_test,p.Omega,ubnrm,muss-mu[indu],(theta_test[indu]-theta_test[indu-1])/(otv.t[indu] - otv.t[indu-1]),'%s\n' % col)

        ax1.plot(otv.t*nrmtime, otv.y[0]/p.ubbalance,color=col,alpha=palp,linestyle=linestyle,linewidth=lw,label='$b = %1.2e$' % b)
        ax2.plot(otv.t*nrmtime, otv.y[1]/p.thetabalance,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
        ax3.plot(otv.t*nrmtime, 100.*otv.y[3]/p.phiinit + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
        ax4.plot(otv.t*nrmtime, 100.*(otv.y[2]/p.pwinit),color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
        #ax3.plot(otv.t*nrmtime, otv.y[3],'--',color=col)
        ax5.plot(otv.t*nrmtime, mu/p.mun,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
        ax6.plot(otv.t*nrmtime, taub/taub_hat + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)

    ax1.set_ylabel('$u_b/\hat{u}_{b_0}$',fontsize=labsize)
    ax2.set_ylabel(r'$\theta/ \hat{\theta}_0$',fontsize=labsize)
    ax3.set_ylabel(r'$\phi/\hat{\phi}_0\ (\%)$',fontsize=labsize)
    ax4.set_ylabel(r'$p_w/\hat{p}_{w_0}\ (\%)$',fontsize=labsize) #, $\phi$
    ax5.set_ylabel(r'$\mu/\mu_n$',fontsize=labsize)#,fontsize=labsize)
    ax6.set_ylabel(r'$\tau_t/\hat{\tau}_{t_0}$',fontsize=labsize)

    maxtime = max(diffusion_times_days)*10. #np.max(otv.t*p.ch)/3.
    ax1.set_xlim([0,maxtime])
    ax2.set_xlim([0,maxtime])
    ax3.set_xlim([0,maxtime])
    ax4.set_xlim([0,maxtime])
    ax5.set_xlim([0,maxtime])
    ax6.set_xlim([0,maxtime])

    #ax1.set_ylim([1,2])
    #ax2.set_ylim([0.5,1])
    #ax3.set_ylim([-0.002,0.06])
    ##ax4.set_ylim([-0.01,0.04])
    #ax4.set_ylim([1.-0.06,1.01])
    #ax5.set_ylim([0.94,1.01])
    ##ax3.set_ylim([0.919,0.9201])
    ##ax3.set_ylim([100*(0.997-1.),100*(1.00002-1.)])

    ax1.get_xaxis().set_ticklabels([])
    ax2.get_xaxis().set_ticklabels([])
    ax3.get_xaxis().set_ticklabels([])
    ax4.get_xaxis().set_ticklabels([])
    ax5.get_xaxis().set_ticklabels([])

    ax6.set_xlabel(r'$\hat{u}_{b_0}t/d_c$',fontsize=labsize)

    #if title is not None:
    #ax1.set_title('No thinning',fontsize=fntsize-5)
    plt.savefig(fname,bbox_inches='tight')

#################################################
def explore_initu_b_nothin():
    s2yr = 1./(86400.*365.)
    fname = 'figs/ub_v_t_b/ub_vs_time_uinit_b_nothin_01.pdf'
    fname2 = 'figs/ub_v_t_b/ub_vs_time_uinit_b_nothin_pwpert_01.pdf'

    a = 0.013

    ch = 1./(10.*86400.)
    pw0 = 0.92
    pwpertvals = np.linspace(1.,0.98/pw0,25)
    ubpertvals = np.linspace(1.0,2,25)
    bvals = np.linspace(0.01,0.05,25)
    ubss_arr = np.empty((len(ubpertvals),len(bvals)),dtype=np.float64)
    ubss_pwpert_arr = np.empty((len(pwpertvals),len(bvals)),dtype=np.float64)

    omegadiff = np.empty((len(ubpertvals),len(bvals)),dtype=np.float64)

    for i,ubpert in enumerate(ubpertvals):
        print('row %d of %d' % (i,len(ubpertvals)))
        for j,b in enumerate(bvals):
            pert = {'parameter' : 'ub','value' : ubpert}
            p,otv = surge_tools.surge_mm(a=a,b=b,ch=ch,pw0=pw0,geoswitch=0.,tfyr=5.,pert=pert)

            ubss_arr[i,j] = otv.y[0][-10]/p.ubbalance

    fig = plt.figure()
    cs = plt.contourf(bvals,ubpertvals,ubss_arr,cmap=cm.cividis)
    c2 = plt.contour(cs,levels=cs.levels[::1],colors='0.4',linewidths=(0.5,))
    cbar = plt.colorbar(cs)

    cbar.ax.set_ylabel(r'$\hat{u}_{b_1}/\hat{u}_{b_0}$',fontsize=labsize)
    plt.ylabel(r'$u_{b_{pert}}/\hat{u}_{b_0}$',fontsize=labsize)
    plt.xlabel(r'$b$',fontsize=labsize)

    plt.savefig(fname,bbox_inches='tight')

    # for i,pwpert in enumerate(pwpertvals):
    #     print('row %d of %d' % (i,len(pwpertvals)))
    #     for j,b in enumerate(bvals):
    #         pert = {'parameter' : 'pw','value' : pwpert}
    #         p,otv = surge_tools.surge_mm(a=a,b=b,ch=ch,pw0=pw0,geoswitch=0.,tfyr=5.,pert=pert)
    #
    #         ubss_pwpert_arr[i,j] = otv.y[0][-10]/p.ubbalance
    #     print(p.pwinit/p.p_over0)
    #
    # fig = plt.figure()
    # cs = plt.contourf(bvals,pwpertvals,ubss_pwpert_arr,cmap=cm.cividis)
    # c2 = plt.contour(cs,levels=cs.levels[::1],colors='0.4',linewidths=(0.5,))
    # cbar = plt.colorbar(cs)
    #
    # cbar.ax.set_ylabel(r'$\hat{u}_{b_1}/\hat{u}_{b_0}$',fontsize=labsize)
    # plt.ylabel(r'$p_{w_{pert}}/p_{w_{\infty}}$',fontsize=labsize)
    # plt.xlabel(r'$b$',fontsize=labsize)
    #
    # plt.savefig(fname2,bbox_inches='tight')

#################################################
def explore_b():
    fname='figs/ub_v_t_b/ub_vs_time_b_05c.pdf'
    pert = {'parameter' : 'ub', 'value' : 1.1}
    diffusion_times_days = [1.e2,5.e3]
    tfyr = 50.
    geoswitch = 1.

    #explore_b_nothin(geoswitch=geoswitch,fname=fname,pert=pert,diffusion_times_days=diffusion_times_days,tfyr=tfyr)

    fig = plt.figure(figsize=(12,18))
    gs = gridspec.GridSpec(8,1,wspace=0.05,hspace=0.1,height_ratios=[2,1,1,1,1,1,1,1])#,width_ratios=[50,1])
    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[1,0])
    ax3 = plt.subplot(gs[2,0])
    ax4 = plt.subplot(gs[3,0])
    ax5 = plt.subplot(gs[4,0])
    ax6 = plt.subplot(gs[5,0])
    ax7 = plt.subplot(gs[6,0])
    ax8 = plt.subplot(gs[7,0])

    #ax1.plot([0,100],[1,1],'k--',linewidth=1.5)

    #for b,col in zip([0.14,0.13,0.12,0.10,0.08,0.0],
    ax1.plot([0,10000],[1.1,1.1],'tab:gray',linewidth=1)
    # ax3.plot([0,10000],[0,0],'tab:gray',linewidth=1)

    yoffset = -1.
    for th,col in zip(diffusion_times_days,['tab:red','tab:blue']):
        ax2.plot([0,1],[yoffset,yoffset],color=col,linewidth=2.5,label='$t_h = %d$ days' % th)
    leg = ax2.legend(loc='lower right')
    leg.set_title(legend_title + '\n' + 'and line types',prop={'size':legend_title_font})
    yoffset = 0.

    bvals = [0.05,0.03,0.01,0.0]
    for singlecol in ['tab:gray','tab:blue']:
        lt1 = True
        for b,col,palp,lw in zip(bvals,[singlecol]*len(bvals),np.linspace(0.3,1,len(bvals))[::-1],np.linspace(2,3,len(bvals))[::-1]):
                    #['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:gray']):
            a = 0.013
            ch = 1./(diffusion_times_days[1]*86400.)
            #epsrat = 40./(1.-0.1)**2
            p,otv = surge_tools.surge_mm(a=a,b=b,ch=ch,geoswitch=geoswitch,tfyr=tfyr,pert=pert)
            if p.a > p.b:
                if lt1:
                    linestyle = '--'
                    lt1 = False
                else:
                    linestyle = '--'
            else:
                linestyle = '-'

            print(p.b, p.a - p.b)
            #ax3.plot([s2yr/p.ch,s2yr/p.ch],[0,1],'b:',linewidth=1.5)
            ub = otv.y[0]
            theta = otv.y[1]
            mu = p.mun + p.a *np.log(ub/p.ubc) + p.b * np.log(theta/p.thetac)

            thick = otv.y[4]
            alpha = otv.y[5]

            p_over = p.rhoi*p.g*otv.y[4]
            pw_Pa = otv.y[2]
            Neff = p_over - pw_Pa
            taub = Neff*mu

            N_hat = p.p_over0 - p.pwinit
            taub_hat = p.mun*N_hat

            pwdot = np.empty_like(otv.y[2])
            hdot = np.empty_like(pwdot)
            pwdot[:-1] = (otv.y[2][1:] - otv.y[2][:-1]) / (otv.t[1:] - otv.t[:-1])
            hdot[:-1] = (otv.y[4][1:] - otv.y[4][:-1]) / (otv.t[1:] - otv.t[:-1])
            pwdot[-1] = pwdot[-2]
            hdot[-1] = hdot[-2]

            Lambda_hat = p.epsilon_e * (1. - p.phiinit)**2 / p.epsilon_p
            if p.b > np.finfo(np.float32).eps:
                lterm = p.mun/(p.b*Lambda_hat)
                ublabel = '$b = %1.3f$, $a-b = %1.3f$' % (p.b,p.a - p.b)
            else:
                ublabel = '$b = %1.3f$, $a-b = %1.3f$' % (p.b,p.a - p.b)

            nrmtime = p.ubbalance/(Lambda_hat*p.dc) # p.ch
            indu = np.argmin(np.abs(otv.t*nrmtime - 100.))
            ubnrm = ub[indu]/p.ubbalance
            muss = p.mun + (p.a-p.b)*np.log(ubnrm)
            ubss_test = ubnrm**(1./p.n) + p.Omega*np.log(ubnrm)
            theta_test = otv.y[1]
            print(ubss_test,p.Omega,ubnrm,muss-mu[indu],theta_test[indu]-theta_test[indu-1],'%s\n' % col)

            ax1.plot(otv.t*nrmtime, otv.y[0]/p.ubbalance + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw,label=ublabel)
            ax2.plot(otv.t*nrmtime, otv.y[1]/p.thetabalance + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
            ax3.plot(otv.t*nrmtime, otv.y[3]/p.phiinit + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
            ax4.plot(otv.t*nrmtime, 1.*(otv.y[2]/p.pwinit) + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
            ax5.plot(otv.t*nrmtime, thick/thick[0] + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
            ax6.plot(otv.t*nrmtime, Neff/N_hat + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
            ax7.plot(otv.t*nrmtime, mu/p.mun + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
            ax8.plot(otv.t*nrmtime, taub/taub_hat + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)

            #if yoffset < 1.e-6 and p.a > p.b:
            #    print(otv.y[2][-1]/p.pwinit-1.)

        if singlecol == 'tab:gray':
            leg = ax1.legend(loc='upper right')
            leg.set_title(legend_title + ' and line colors',prop={'size':legend_title_font})
        yoffset = 0.

    singlecol = 'tab:red'
    lt1 = True
    for b,col,palp,lw in zip(bvals,[singlecol]*len(bvals),np.linspace(0.3,1,len(bvals))[::-1],np.linspace(2,3,len(bvals))[::-1]):
        a = 0.013 #0.9*b
        ch = 1./(diffusion_times_days[0]*86400.)
        #epsrat = 40./(1.-0.1)**2
        p,otv = surge_tools.surge_mm(a=a,b=b,ch=ch,geoswitch=geoswitch,tfyr=tfyr,pert=pert)
        if p.a > p.b:
            if lt1:
                linestyle = '--'
                lt1 = False
            else:
                linestyle = '--'
        else:
            linestyle = '-'

        print(p.b, p.a - p.b, p.ubc*86400.*365.)
        #ax3.plot([s2yr/p.ch,s2yr/p.ch],[0,1],'b:',linewidth=1.5)

        ub = otv.y[0]
        theta = otv.y[1]
        mu = p.mun + p.a *np.log(ub/p.ubc) + p.b * np.log(theta/p.thetac)

        thick = otv.y[4]
        alpha = otv.y[5]

        p_over = p.rhoi*p.g*otv.y[4]
        pw_Pa = otv.y[2]
        Neff = p_over - pw_Pa
        taub = Neff*mu

        N_hat = p.p_over0 - p.pwinit
        taub_hat = p.mun*N_hat

        Lambda_hat = p.epsilon_e * (1. - p.phiinit)**2 / p.epsilon_p

        nrmtime = p.ubbalance/(Lambda_hat*p.dc) # p.ch
        indu = np.argmin(np.abs(otv.t*nrmtime - 1000.))
        ubnrm = ub[indu]/p.ubbalance
        muss = p.mun + (p.a-p.b)*np.log(ubnrm)
        ubss_test = ubnrm**(1./p.n) + p.Omega*np.log(ubnrm)
        theta_test = otv.y[1]
        print(ubss_test,p.Omega,ubnrm,muss-mu[indu],(theta_test[indu]-theta_test[indu-1])/(otv.t[indu] - otv.t[indu-1]),'%s\n' % col)

        ax1.plot(otv.t*nrmtime, otv.y[0]/p.ubbalance,color=col,alpha=palp,linestyle=linestyle,linewidth=lw,label='$b = %1.2e$' % b)
        ax2.plot(otv.t*nrmtime, otv.y[1]/p.thetabalance,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
        ax3.plot(otv.t*nrmtime, otv.y[3]/p.phiinit + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
        ax4.plot(otv.t*nrmtime, 1.*(otv.y[2]/p.pwinit),color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
        ax5.plot(otv.t*nrmtime, thick/thick[0],color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
        ax6.plot(otv.t*nrmtime, Neff/N_hat,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
        ax7.plot(otv.t*nrmtime, mu/p.mun,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
        ax8.plot(otv.t*nrmtime, taub/taub_hat,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)

    ax1.plot([0,10000],[1,1],'tab:gray',linewidth=1.5)
    ax2.plot([0,10000],[1,1],'tab:gray',linewidth=1.5)
    ax3.plot([0,10000],[1,1],'tab:gray',linewidth=1.5)
    ax4.plot([0,10000],[1,1],'tab:gray',linewidth=1.5)
    ax5.plot([0,10000],[1,1],'tab:gray',linewidth=1.5)
    ax6.plot([0,10000],[1,1],'tab:gray',linewidth=1.5)
    ax7.plot([0,10000],[1,1],'tab:gray',linewidth=1.5)
    ax8.plot([0,10000],[1,1],'tab:gray',linewidth=1.5)

    ax1.set_ylabel('$u_b/\hat{u}_{b_0}$',fontsize=labsize)
    ax2.set_ylabel(r'$\theta/ \hat{\theta}_0$',fontsize=labsize)
    ax3.set_ylabel(r'$\phi/\hat{\phi}_0$',fontsize=labsize)
    ax4.set_ylabel(r'$p_w/\hat{p}_{w_0}$',fontsize=labsize) #, $\phi$
    # ax3.set_ylabel(r'$\phi/\hat{\phi}_{0}$',fontsize=labsize)
    ax5.set_ylabel(r'$h/\hat{h}_0$',fontsize=labsize)
    ax6.set_ylabel(r'$N/\hat{N}_{0}$',fontsize=labsize)
    ax7.set_ylabel(r'$\mu/\mu_n$',fontsize=labsize)#,fontsize=labsize)
    ax8.set_ylabel(r'$\tau_t/\hat{\tau}_{t_0}$',fontsize=labsize)


    maxtime = 100. #np.max(otv.t*p.ch)/3.
    ax1.set_xlim([0,maxtime])
    ax2.set_xlim([0,maxtime])
    ax3.set_xlim([0,maxtime])
    ax4.set_xlim([0,maxtime])
    ax5.set_xlim([0,maxtime])
    ax6.set_xlim([0,maxtime])
    ax7.set_xlim([0,maxtime])
    ax8.set_xlim([0,maxtime])

    ax1.set_ylim([0,10])
    ax2.set_ylim([0.,1.05])
    ax3.set_ylim([0.8,1.02])
    ax4.set_ylim([0.96,1.002])
    # ax6.set_ylim([0.,0.1])
    ax7.set_ylim([0.8,1.02])


    ax1.get_xaxis().set_ticklabels([])
    ax2.get_xaxis().set_ticklabels([])
    ax3.get_xaxis().set_ticklabels([])
    ax4.get_xaxis().set_ticklabels([])
    ax5.get_xaxis().set_ticklabels([])
    ax6.get_xaxis().set_ticklabels([])
    ax7.get_xaxis().set_ticklabels([])
    ax8.set_xlabel(r'$t\hat{u}_{b_0}\epsilon_p/(d_c\epsilon_e)$',fontsize=labsize)

    #if title is not None:
    #ax1.set_title('No thinning',fontsize=fntsize-5)
    plt.savefig(fname,bbox_inches='tight')

#################################################
def explore_b_parameterspace_1():
    fname='figs/ub_v_t_b/parameterspace/ensemble_airfoil_01.pdf'
    pert = {'parameter' : 'ub', 'value' : 1.1}
    diffusion_times_days = [2600.,2600.]
    tfyr = 100.
    geoswitch = 1.

    #explore_b_nothin(geoswitch=geoswitch,fname=fname,pert=pert,diffusion_times_days=diffusion_times_days,tfyr=tfyr)

    fig = plt.figure(figsize=(12,18))
    gs = gridspec.GridSpec(8,1,wspace=0.05,hspace=0.1,height_ratios=[2,1,1,1,1,1,1,1])#,width_ratios=[50,1])
    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[1,0])
    ax3 = plt.subplot(gs[2,0])
    ax4 = plt.subplot(gs[3,0])
    ax5 = plt.subplot(gs[4,0])
    ax6 = plt.subplot(gs[5,0])
    ax7 = plt.subplot(gs[6,0])
    ax8 = plt.subplot(gs[7,0])

    #ax1.plot([0,100],[1,1],'k--',linewidth=1.5)

    #for b,col in zip([0.14,0.13,0.12,0.10,0.08,0.0],
    # ax1.plot([0,10000],[1.1,1.1],'tab:gray',linewidth=1)
    # ax3.plot([0,10000],[0,0],'tab:gray',linewidth=1)

    # yoffset = -1.
    # for th,col in zip(diffusion_times_days,['tab:red','tab:blue']):
    #     ax2.plot([0,1],[yoffset,yoffset],color=col,linewidth=2.5,label='$t_h = %d$ days' % th)
    # leg = ax2.legend(loc='lower right')
    # leg.set_title(legend_title + '\n' + 'and line types',prop={'size':legend_title_font})
    yoffset = 0.

    bvals = [0.022,0.024,0.026,0.028,0.030]
    colours = ['tab:blue','tab:orange','tab:red','tab:green','tab:olive'][::-1]
    linestyle = '-'
    lw = 2
    palp = 1

    lt1 = True
    for b,col in zip(bvals,colours):
        a = 0.013 #0.9*b
        ch = 1./(diffusion_times_days[0]*86400.)
        # epsp = 1.e-3
        # epse = 50.*epsp ,epsp=epsp,epse=epse
        p,otv = surge_tools.surge_mm(a=a,b=b,ch=ch,geoswitch=geoswitch,tfyr=tfyr,pert=pert)



        ub = otv.y[0]
        theta = otv.y[1]
        mu = p.mun + p.a *np.log(ub/p.ubc) + p.b * np.log(theta/p.thetac)

        thick = otv.y[4]
        alpha = otv.y[5]

        p_over = p.rhoi*p.g*otv.y[4]
        pw_Pa = otv.y[2]
        Neff = p_over - pw_Pa
        taub = Neff*mu

        N_hat = p.p_over0 - p.pwinit
        taub_hat = p.mun*N_hat

        Lambda_hat = p.epsilon_e * (1. - p.phiinit)**2 / p.epsilon_p

        nrmtime = p.ubbalance/(Lambda_hat*p.dc) # p.ch
        indu = np.argmin(np.abs(otv.t*nrmtime - 1000.))
        ubnrm = ub[indu]/p.ubbalance
        muss = p.mun + (p.a-p.b)*np.log(ubnrm)
        ubss_test = ubnrm**(1./p.n) + p.Omega*np.log(ubnrm)
        theta_test = otv.y[1]
        # print(ubss_test,p.Omega,ubnrm,muss-mu[indu],(theta_test[indu]-theta_test[indu-1])/(otv.t[indu] - otv.t[indu-1]),'%s\n' % col)

        ax1.plot(otv.t*nrmtime, otv.y[0]/p.ubbalance,color=col,alpha=palp,linestyle=linestyle,linewidth=lw,label='$b = %1.2e$' % b)
        ax2.plot(otv.t*nrmtime, otv.y[1]/p.thetabalance,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
        ax3.plot(otv.t*nrmtime, otv.y[3]/p.phiinit + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
        ax4.plot(otv.t*nrmtime, 1.*(otv.y[2]/p.pwinit),color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
        ax5.plot(otv.t*nrmtime, thick/thick[0],color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
        ax6.plot(otv.t*nrmtime, Neff/N_hat,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
        ax7.plot(otv.t*nrmtime, mu/p.mun,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
        ax8.plot(otv.t*nrmtime, taub/taub_hat,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)

    leg = ax1.legend(loc='upper center')
    leg.set_title('$t_h = 2600$ days and')

    ax1.plot([0,10000],[1,1],'tab:gray',linewidth=1.5)
    ax2.plot([0,10000],[1,1],'tab:gray',linewidth=1.5)
    ax3.plot([0,10000],[1,1],'tab:gray',linewidth=1.5)
    ax4.plot([0,10000],[1,1],'tab:gray',linewidth=1.5)
    ax5.plot([0,10000],[1,1],'tab:gray',linewidth=1.5)
    ax6.plot([0,10000],[1,1],'tab:gray',linewidth=1.5)
    ax7.plot([0,10000],[1,1],'tab:gray',linewidth=1.5)
    ax8.plot([0,10000],[1,1],'tab:gray',linewidth=1.5)

    ax1.set_ylabel('$u_b/\hat{u}_{b_0}$',fontsize=labsize)
    ax2.set_ylabel(r'$\theta/ \hat{\theta}_0$',fontsize=labsize)
    ax3.set_ylabel(r'$\phi/\hat{\phi}_0$',fontsize=labsize)
    ax4.set_ylabel(r'$p_w/\hat{p}_{w_0}$',fontsize=labsize) #, $\phi$
    # ax3.set_ylabel(r'$\phi/\hat{\phi}_{0}$',fontsize=labsize)
    ax5.set_ylabel(r'$h/\hat{h}_0$',fontsize=labsize)
    ax6.set_ylabel(r'$N/\hat{N}_{0}$',fontsize=labsize)
    ax7.set_ylabel(r'$\mu/\mu_n$',fontsize=labsize)#,fontsize=labsize)
    ax8.set_ylabel(r'$\tau_t/\hat{\tau}_{t_0}$',fontsize=labsize)


    maxtime = 250. #np.max(otv.t*p.ch)/3.
    ax1.set_xlim([0,maxtime])
    ax2.set_xlim([0,maxtime])
    ax3.set_xlim([0,maxtime])
    ax4.set_xlim([0,maxtime])
    ax5.set_xlim([0,maxtime])
    ax6.set_xlim([0,maxtime])
    ax7.set_xlim([0,maxtime])
    ax8.set_xlim([0,maxtime])

    ax1.set_ylim([0,10])
    ax2.set_ylim([0.,6.5])
    # ax3.set_ylim([0.4,1.1])
    # ax4.set_ylim([0.96,1.002])
    # # ax6.set_ylim([0.,0.1])
    # ax7.set_ylim([0.8,1.02])


    ax1.get_xaxis().set_ticklabels([])
    ax2.get_xaxis().set_ticklabels([])
    ax3.get_xaxis().set_ticklabels([])
    ax4.get_xaxis().set_ticklabels([])
    ax5.get_xaxis().set_ticklabels([])
    ax6.get_xaxis().set_ticklabels([])
    ax7.get_xaxis().set_ticklabels([])
    ax8.set_xlabel(r'$t\hat{u}_{b_0}\epsilon_p/(d_c\epsilon_e)$',fontsize=labsize)

    #if title is not None:
    #ax1.set_title('No thinning',fontsize=fntsize-5)
    plt.savefig(fname,bbox_inches='tight')


#################################################
def explore_b_ub_vs_others():
    fname='figs/ub_v_t_b/ub_vs_tb_b_05.pdf'
    pert = {'parameter' : 'ub', 'value' : 1.1}
    diffusion_times_days = [1.e2,5.e3]
    tfyr = 50.
    geoswitch = 1.

    #explore_b_nothin(geoswitch=geoswitch,fname=fname,pert=pert,diffusion_times_days=diffusion_times_days,tfyr=tfyr)

    fig = plt.figure(figsize=(12,12))
    gs = gridspec.GridSpec(3,3,wspace=1,hspace=1)#,height_ratios=[2,1,1,1,1,1,1,1])#,width_ratios=[50,1])
    ax00 = plt.subplot(gs[0,0])
    ax01 = plt.subplot(gs[0,1])
    ax02 = plt.subplot(gs[0,2])
    ax10 = plt.subplot(gs[1,0])
    ax11 = plt.subplot(gs[1,1])
    ax12 = plt.subplot(gs[1,2])
    ax20 = plt.subplot(gs[2,0])
    ax21 = plt.subplot(gs[2,1])
    ax22 = plt.subplot(gs[2,2])

    yoffset = 0.
    # for th,col in zip(diffusion_times_days,['tab:red','tab:blue']):
    #     ax10.plot([0,1],[yoffset,yoffset],color=col,linewidth=2.5,label='$c_h^{-1} = %d$ days' % th)
    # ax10.legend(loc='lower left')

    bvals = [0.05,0.03,0.01,0.0]
    for singlecol in ['tab:gray','tab:blue']:
        lt1 = True
        for b,col,palp,lw in zip(bvals,[singlecol]*len(bvals),np.linspace(0.3,1,len(bvals))[::-1],np.linspace(2,3,len(bvals))[::-1]):
                    #['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:gray']):
            a = 0.013
            ch = 1./(diffusion_times_days[1]*86400.)
            #epsrat = 40./(1.-0.1)**2
            p,otv = surge_tools.surge_mm(a=a,b=b,ch=ch,geoswitch=geoswitch,tfyr=tfyr,pert=pert)
            if p.a > p.b:
                if lt1:
                    linestyle = '--'
                    lt1 = False
                else:
                    linestyle = '--'
            else:
                linestyle = '-'

            print(p.b, p.a - p.b)
            #ax10.plot([s2yr/p.ch,s2yr/p.ch],[0,1],'b:',linewidth=1.5)
            ub = otv.y[0]
            theta = otv.y[1]
            mu = p.mun + p.a *np.log(ub/p.ubc) + p.b * np.log(theta/p.thetac)

            thick = otv.y[4]
            alpha = otv.y[5]

            p_over = p.rhoi*p.g*otv.y[4]
            pw_Pa = otv.y[2]
            Neff = p_over - pw_Pa
            taub = Neff*mu

            N_hat = p.p_over0 - p.pwinit
            taub_hat = p.mun*N_hat

            pwdot = np.empty_like(otv.y[2])
            thetadot = np.empty_like(pwdot)
            pwdot[:-1] = (otv.y[2][1:] - otv.y[2][:-1]) / (otv.t[1:] - otv.t[:-1])
            thetadot[:-1] = (otv.y[1][1:] - otv.y[1][:-1]) / (otv.t[1:] - otv.t[:-1])
            pwdot[-1] = pwdot[-2]
            thetadot[-1] = thetadot[-2]

            beta = p.epsilon_e * (1. - otv.y[3])**2 / Neff
            phidot_e = pwdot * beta
            phidot_p = -p.epsilon_p * thetadot / otv.y[1]
            phidot = phidot_e + phidot_p

            Lambda_hat = p.epsilon_e * (1. - p.phiinit)**2 / p.epsilon_p
            if p.b > np.finfo(np.float32).eps:
                lterm = p.mun/(p.b*Lambda_hat)
                ublabel = '$b = %1.3f$, $a-b = %1.3f$, $\hat{\lambda}_0 = %1.2f$' % (p.b,p.a - p.b,lterm)
            else:
                ublabel = '$b = %1.3f$, $a-b = %1.3f$, $\hat{\lambda}_0 = \infty$' % (p.b,p.a - p.b)

            nrmtime = 1./p.thetac # p.ch
            indu = np.argmin(np.abs(otv.t*nrmtime - 100.))
            ubnrm = ub[indu]/p.ubbalance
            muss = p.mun + (p.a-p.b)*np.log(ubnrm)
            ubss_test = ubnrm**(1./p.n) + p.Omega*np.log(ubnrm)
            theta_test = otv.y[1]
            # print(ubss_test,p.Omega,ubnrm,muss-mu[indu],theta_test[indu]-theta_test[indu-1],'%s\n' % col)

            ax00.plot(taub/taub_hat, otv.y[0]/p.ubbalance + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw,label=ublabel)
            ax01.plot(Neff/N_hat, otv.y[0]/p.ubbalance + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
            ax02.plot(Neff/N_hat, taub/taub_hat + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
            ax10.plot(thick/thick[0], otv.y[0]/p.ubbalance + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
            ax11.plot(otv.y[2]/p.pwinit, otv.y[0]/p.ubbalance + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
            ax12.plot(otv.y[2]/p.pwinit, thick/thick[0] + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
            ax20.plot(Neff/N_hat,phidot + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
            ax21.plot(otv.y[3]/p.phiinit, otv.y[2]/p.pwinit + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
            ax22.plot(phidot_e, phidot_p + yoffset,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)

        # if singlecol == 'tab:gray':
        #     ax1.legend(loc='upper right')
        yoffset = 0.

    singlecol = 'tab:red'
    lt1 = True
    for b,col,palp,lw in zip(bvals,[singlecol]*len(bvals),np.linspace(0.3,1,len(bvals))[::-1],np.linspace(2,3,len(bvals))[::-1]):
        a = 0.013 #0.9*b
        ch = 1./(diffusion_times_days[0]*86400.)
        #epsrat = 40./(1.-0.1)**2
        p,otv = surge_tools.surge_mm(a=a,b=b,ch=ch,geoswitch=geoswitch,tfyr=tfyr,pert=pert)
        if p.a > p.b:
            if lt1:
                linestyle = '--'
                lt1 = False
            else:
                linestyle = '--'
        else:
            linestyle = '-'

        print(p.b, p.a - p.b, p.ubc*86400.*365.)
        #ax10.plot([s2yr/p.ch,s2yr/p.ch],[0,1],'b:',linewidth=1.5)

        ub = otv.y[0]
        theta = otv.y[1]
        mu = p.mun + p.a *np.log(ub/p.ubc) + p.b * np.log(theta/p.thetac)

        thick = otv.y[4]
        alpha = otv.y[5]
        taudrive = p.rhoi*p.g*thick*alpha
        taudrive_hat = p.rhoi*p.g*thick[0]*alpha[0]

        p_over = p.rhoi*p.g*otv.y[4]
        pw_Pa = otv.y[2]
        Neff = p_over - pw_Pa
        taub = Neff*mu

        N_hat = p.p_over0 - p.pwinit
        taub_hat = p.mun*N_hat

        pwdot = np.empty_like(otv.y[2])
        thetadot = np.empty_like(pwdot)
        pwdot[:-1] = (otv.y[2][1:] - otv.y[2][:-1]) / (otv.t[1:] - otv.t[:-1])
        thetadot[:-1] = (otv.y[1][1:] - otv.y[1][:-1]) / (otv.t[1:] - otv.t[:-1])
        pwdot[-1] = pwdot[-2]
        thetadot[-1] = thetadot[-2]

        beta = p.epsilon_e * (1. - otv.y[3])**2 / Neff
        phidot_e = pwdot * beta
        phidot_p = -p.epsilon_p * thetadot / otv.y[1]
        phidot = phidot_e + phidot_p

        nrmtime = 1./p.thetac # p.ch
        indu = np.argmin(np.abs(otv.t*nrmtime - 1000.))
        ubnrm = ub[indu]/p.ubbalance
        muss = p.mun + (p.a-p.b)*np.log(ubnrm)
        ubss_test = ubnrm**(1./p.n) + p.Omega*np.log(ubnrm)
        theta_test = otv.y[1]
        # print(ubss_test,p.Omega,ubnrm,muss-mu[indu],(theta_test[indu]-theta_test[indu-1])/(otv.t[indu] - otv.t[indu-1]),'%s\n' % col)

        ax00.plot(taub/taub_hat, otv.y[0]/p.ubbalance,color=col,alpha=palp,linestyle=linestyle,linewidth=lw,label='$b = %1.2e$' % b)
        ax01.plot(Neff/N_hat, otv.y[0]/p.ubbalance,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
        ax02.plot(taudrive/taudrive_hat, taub/taub_hat,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
        ax10.plot(thick/thick[0], otv.y[0]/p.ubbalance,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
        ax11.plot(otv.y[2]/p.pwinit, otv.y[0]/p.ubbalance,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
        ax12.plot(otv.y[2]/p.pwinit, thick/thick[0],color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
        ax20.plot(Neff/N_hat,phidot,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
        ax21.plot(otv.y[3]/p.phiinit, otv.y[2]/p.pwinit,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)
        ax22.plot(phidot_e,phidot_p,color=col,alpha=palp,linestyle=linestyle,linewidth=lw)

    ax00.set_ylabel(r'$u_b/\hat{u}_{b_0}$',fontsize=labsize)
    ax01.set_ylabel(r'$u_b/\hat{u}_{b_0}$',fontsize=labsize)
    ax02.set_ylabel(r'$\tau_t/\hat{\tau}_{t_0}$',fontsize=labsize)
    ax10.set_ylabel(r'$u_b/\hat{u}_{b_0}$',fontsize=labsize)
    ax11.set_ylabel(r'$u_b/\hat{u}_{b_0}$',fontsize=labsize)
    ax12.set_ylabel(r'$h/\hat{h}_0$',fontsize=labsize)
    ax20.set_ylabel(r'$\dot{\phi}$',fontsize=labsize)
    # ax20.set_ylabel(r'$u_b/\hat{u}_{b_0}$',fontsize=labsize)
    ax21.set_ylabel(r'$p_w/\hat{p}_{w_0}$',fontsize=labsize)
    ax22.set_ylabel(r'$\dot{\phi}_p$',fontsize=labsize)

    ax00.set_xlabel(r'$\tau_t/\hat{\tau}_{t_0}$',fontsize=labsize)
    ax01.set_xlabel(r'$N/\hat{N}_0$',fontsize=labsize)
    ax02.set_xlabel(r'$\tau_d/\hat{\tau}_{d_0}$',fontsize=labsize)
    ax10.set_xlabel(r'$h/\hat{h}_0$',fontsize=labsize)
    ax11.set_xlabel(r'$p_w/\hat{p}_{w_0}$',fontsize=labsize)#,fontsize=labsize)
    ax12.set_xlabel(r'$p_w/\hat{p}_{w_0}$',fontsize=labsize)#,fontsize=labsize)
    # ax20.set_xlabel(r'$\theta/ \hat{\theta}_0$',fontsize=labsize)
    ax20.set_xlabel(r'$N/\hat{N}_0$',fontsize=labsize)
    ax21.set_xlabel(r'$\phi/\hat{\phi}_0$',fontsize=labsize)
    ax22.set_xlabel(r'$\dot{\phi}_e$',fontsize=labsize)

    # ax00.set_xlim([0,maxtime])
    # ax01.set_xlim([0,maxtime])
    # ax10.set_xlim([0,maxtime])
    # ax4.set_xlim([0,maxtime])

    ublim = [0.9,3]
    ax00.set_ylim(ublim)
    ax01.set_ylim(ublim)
    ax10.set_ylim(ublim)
    ax11.set_ylim(ublim)
    # ax20.set_ylim(ublim)
    # ax21.set_ylim(ublim)
    # ax22.set_ylim(ublim)

    plt.savefig(fname,bbox_inches='tight')

#################################################
def stack_plot(fname='figs/ub_v_t/ub_vs_time_stack_01.pdf'):
    pert = {'parameter' : 'ub',
            'value' : 1.1}
    p,otv = surge_tools.surge_mm(pert=pert)
    stack_plot_general(p,otv,fname)

#################################################
def stack_plot_general(p,otv,fname,title=None):
    fig = plt.figure(figsize=(12,14))
    gs = gridspec.GridSpec(4,1,wspace=0.05,hspace=0.1,height_ratios=[2,2,1,1])#,width_ratios=[50,1])
    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[1,0])
    ax3 = plt.subplot(gs[2,0])
    ax4 = plt.subplot(gs[3,0])
    #cax = plt.subplot(gs[:,1])

    ax1.plot(otv.t*p.s2y, otv.y[0]/p.ubbalance,'.',linestyle='-',label='$u_b$')
    ax1.set_ylabel('$u_b/u_b^*$',fontsize=labsize)
    ax1.set_ylim([0,10])

    ax2.plot(otv.t*p.s2y, otv.y[1]/p.thetabalance,label=r'$\theta$')
    ax2.set_ylabel(r'$\theta u_b^*/d_c$',fontsize=labsize)

    ax3.plot(otv.t*p.s2y, otv.y[2]/p.p_over0,label=r'$p_w$')
    ax3.plot(otv.t*p.s2y, otv.y[3],label=r'$\phi$')
    ax3.set_ylabel(r'$p_w/p_{i0}$, $\phi$')#,fontsize=labsize)

    ax4.plot(otv.t*p.s2y, otv.y[4]/p.h,linewidth=4,label=r'$h$')
    ax4.plot(otv.t*p.s2y, otv.y[5]/p.alpha0,label=r'$\alpha$')
    ax4.set_ylabel(r'$h/h_0$, $\alpha/\alpha_0$')#,fontsize=labsize)

    ax1.get_xaxis().set_ticklabels([])
    ax2.get_xaxis().set_ticklabels([])
    ax3.get_xaxis().set_ticklabels([])

    ax4.set_xlabel('Time (years)')

    if title is not None:
        ax1.set_title(title,fontsize=fntsize-5)
    plt.savefig(fname,bbox_inches='tight')

#################################################
def qdp1(fname='figs/ub_v_t/ub_vs_time_00.pdf'):
    pert = {'parameter' : 'ub',
            'value' : 1.1}
    p,otv = surge_tools.surge_mm(pert=pert)

    fig = plt.figure(figsize=(12,8))
    plt.plot(otv.t*p.s2y, otv.y[0]/p.ubc,'.',linestyle='-',label='$u_b$')
    plt.plot(otv.t*p.s2y, otv.y[1]/p.thetabalance,label=r'$\theta$')
    plt.plot(otv.t*p.s2y, otv.y[2]/p.p_over0,label=r'$p_w$')
    plt.plot(otv.t*p.s2y, otv.y[3],label=r'$\phi$')
    plt.plot(otv.t*p.s2y, otv.y[4]/p.h,linewidth=4,label=r'$h$')
    plt.plot(otv.t*p.s2y, otv.y[5]/p.alpha0,label=r'$\alpha$')
    plt.plot(otv.t*p.s2y, 1.+otv.t*0.,'--',color='0.6')
    plt.legend()
    plt.ylim([0,10])
    plt.xlabel('Time (years)')
    plt.savefig(fname,bbox_inches='tight')

#################################################
def plot_lamberw():
    xvals = np.linspace(-10.,10.,100)
    yvals = np.empty_like(xvals)
    for i in range(len(yvals)):
        x = xvals[i]
        yvals[i] = np.real(lambertw(np.exp(x)*x)/x)
    fig = plt.figure()
    plt.plot(xvals,yvals,'k')
    plt.ylabel('$\hat{u}_{b1}/\hat{u}_{b0}$')
    plt.xlabel('$1/\Omega$')
    plt.savefig('figs/lambertw_trial.pdf',bbox_inches='tight')

#################################################
if __name__ == '__main__':
    main()
