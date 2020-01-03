
def set_mpl(mpl):
    fntsize = 15
    labsize = fntsize + 5
    font = {'size'   : fntsize}
    axes = {'linewidth' : 1}
    tick = {'major.pad' : 12,
            'major.size' : 8,
            'minor.size' : 5}
    mathfont = {'fontset' : 'cm'}
    mpl.rc('font', **font)
    mpl.rc('axes', **axes)
    mpl.rc('xtick', **tick)
    mpl.rc('ytick', **tick)
    mpl.rc('mathtext', **mathfont)
    #mpl.rc('text' , **{'usetex':True})
    mpl.rcParams['hatch.linewidth'] = 0.1
    return mpl,fntsize,labsize
