#!/usr/bin/env python3

import sdf
import sys
import os
import numpy as np
import scipy.constants as sc
import configobj

import matplotlib as mpl
import matplotlib.patches as mpat
import matplotlib.figure as mf
import matplotlib.backends.backend_agg as mplbea
import matplotlib.cm as mcm
import matplotlib.colorbar as mplcb
from matplotlib.colors import LogNorm
import matplotlib.ticker as mt

def loadconfig(config_file):
    try:
        config = configobj.ConfigObj(config_file,file_error=True)
    except IOError:
        print("Error: failed to open config file")
        sys.exit(-1)
    except configobj.ConfigObjError as err:
        print("Warning: Encountered parsing errors:")
        for e in err.errors:
            print("\t{}".format(e.msg))
        print("\nAttempting to continue...")
        config = err.config
    return(config)

def parse_input():
    if len(sys.argv) < 2:
        print("Error: Expected path to config file")
        sys.exit(-1)
    elif len(sys.argv) > 2:
        print("Warning: Unexpected extra command line arguments.")
    return(sys.argv[1])


def main():
    config_file = parse_input()
    config = loadconfig(config_file)

    try:
        sdf_data = sdf.read(config['sdf_file'])
    except:
        raise

    dirname = os.path.basename(os.getcwd())

    grids =  {'xgrid':None
                ,'ygrid':None
                ,'xbins':None
                ,'ybins':None}
    for grid in grids:
        try:
            grids[grid] = float(config[grid])
        except:
            pass

    if ((grids['xgrid'] is not None and grids['xbins'] is not None) or
       (grids['ygrid'] is not None and grids['ybins'] is not None)):
        print("Both gridsize and numbins are specified... "
              "gridsize will take precedence")

    if ((grids['xgrid'] is None and grids['xbins'] is None) or
       (grids['ygrid'] is None and grids['ybins'] is None)):
        print("Warning no gridsize or numbins specified, defaulting to"
              "100 bins")
        if grids['xbins'] is None:
            grids['xbins'] = 100
        if grids['ybins'] is None:
            grids['ybins'] = 100
    
    try:
        output_name = config['output_name']
    except:
        output_name = 'out'

    try:
        output_path = config['output_path']
    except:
        output_path = '.'

    ell = {'center':None, 'width':None, 'height':None}
    for arg in ell:
        try:
            ell[arg] = float(config['ell_{}'.format(arg)])
        except:
            pass
       

    cutoff_px = float(config['cutoff_px']) * sc.m_e * sc.c
    sdf_e_px = getattr(sdf_data, 'Particles_Px_electron').data
    sdf_e_x = getattr(sdf_data, 'Grid_Particles_electron').data[0]
    sdf_e_y = getattr(sdf_data, 'Grid_Particles_electron').data[1]
    sdf_e_w = getattr(sdf_data, 'Particles_Weight_electron').data
    sdf_gridx = getattr(sdf_data, 'Grid_Grid').data[0]
    sdf_gridy = getattr(sdf_data, 'Grid_Grid').data[1]
    sdf_dens = getattr(sdf_data, 'Derived_Number_Density_electron').data
    try:
        sdf_gridz = getattr(sdf_data, 'Grid_Grid').data[2]
        sdf_e_z = getattr(sdf_data, 'Grid_Particles_electron').data[2]
        is3d = True
    except:
        is3d = False

    limits = {'xmin':sdf_e_x.min()
             ,'xmax':sdf_e_x.max()
             ,'ymin':sdf_e_y.min()
             ,'ymax':sdf_e_y.max()}
    if is3d:
        limits['zmin'] = sdf_e_z.min()
        limits['zmax'] = sdf_e_z.max()
 
    for limit in limits:
        try:
            limits[limit] = float(config[limit])
        except:
            pass
    
    hist_limits = {}
    for limit in limits:
        try:
            hist_limits[limit] = float(config['hist_'+limit])
        except:
            hist_limits[limit] = limits[limit]
            
    xargmin = np.argmin(np.abs(sdf_gridx - limits['xmin']))
    xargmax = np.argmin(np.abs(sdf_gridx - limits['xmax']))
    yargmin = np.argmin(np.abs(sdf_gridy - limits['ymin']))
    yargmax = np.argmin(np.abs(sdf_gridy - limits['ymax']))
    if is3d:
        zargmin = np.argmin(np.abs(sdf_gridz - limits['zmin']))
        zargmax = np.argmin(np.abs(sdf_gridz - limits['zmax']))

    sdf_gridx = sdf_gridx[xargmin:xargmax]
    sdf_gridy = sdf_gridy[yargmin:yargmax]
    sdf_dens = sdf_dens[xargmin:xargmax,yargmin:yargmax]
    if is3d:
        sdf_gridz = sdf_gridz[zargmin:zargmax]

    ### Electron selection magic happens here

    energy_mask = sdf_e_px > cutoff_px
    position_mask = np.full(sdf_e_px.shape, True, dtype=bool)
    if None not in ell.values():
        if is3d:
            position_mask = ( 
            ((sdf_e_x - ell['center'])**2 / (0.25*ell['width']**2)) +
            (sdf_e_y**2 / (0.25*ell['height']**2)) +
            (sdf_e_z**2 / (0.25*ell['height']**2)) < 1 )
        else:
            position_mask = (
            ((sdf_e_x - ell['center'])**2 / (0.25*ell['width']**2)) +
            (sdf_e_y**2 / (0.25*ell['height']**2)) < 1 )

        bunch_electron_mask = np.where(np.logical_and(energy_mask,
                                                position_mask))

    bunch_electron_x = sdf_e_x[bunch_electron_mask].reshape(-1)
    bunch_electron_y = sdf_e_y[bunch_electron_mask].reshape(-1)
    bunch_electron_w = sdf_e_w[bunch_electron_mask].reshape(-1)
    if is3d:
        bunch_electron_z = sdf_e_z[bunch_electron_mask].reshape(-1)

    if grids['xgrid'] is not None:
        xbins = np.arange(hist_limits['xmin'],hist_limits['xmax']+grids['xgrid'],grids['xgrid'])
    else:
        xbins = np.linspace(hist_limits['xmin'],hist_limits['xmax'],grids['xbins'])

    if grids['ygrid'] is not None:
        if hist_limits['ymin'] * hist_limits['ymax'] < 0:
            ybins = np.sort(np.concatenate((
                np.arange(0,hist_limits['ymin']-grids['ygrid'],-grids['ygrid']),
                np.arange(grids['ymin'],hist_limits['ymax']+grids['ygrid'],grids['ygrid']))))

        else:
            ybins = np.arange(hist_limits['ymin'],hist_limits['ymax']+grids['ygrid'],grids['ygrid'])
    else:
        if hist_limits['ymin'] * hist_limits['ymax'] < 0:
            ybins = np.sort(np.concatenate((
                np.linspace(hist_limits['ymin'],0,grids['ybins']//2),
                np.linspace(0,hist_limits['ymax'],grids['ybins']//2)[1:])))
        else:
            ybins = np.linspace(hist_limits['ymin'],hist_limits['ymax'],grids['ybins'])

    
    limlist = [ limits[l]*1e6 for l in ['xmin','xmax','ymin','ymax'] ]
    hist_limlist = [ hist_limits[l]*1e6 for l in ['xmin','xmax','ymin','ymax'] ]


### Debug Plot ####


    fig = mf.Figure(figsize=(8.3,11.7))
    canvas = mplbea.FigureCanvasAgg(fig)
    ax = fig.add_subplot(221)
    ax.imshow(sdf_dens.T
             ,aspect='auto'
             ,extent=limlist
             ,origin='upper'
             ,norm=LogNorm(1e23,1e26)
             ,cmap=mcm.plasma)

    if None not in ell.values():
        ax.add_patch(mpat.Ellipse((ell['center']*1e6,0)
                                 ,ell['width']*1e6
                                 ,ell['height']*1e6
                                 ,fill=True
                                 ,fc='blue'
                                 ,alpha=0.2))

    ax2 = fig.add_subplot(222)
    counts, xedges, yedges = np.histogram2d(bunch_electron_x
                                     ,bunch_electron_y
                                     ,bins=[xbins,ybins]
                                     ,weights=bunch_electron_w)

    areas = np.outer(np.diff(xedges),np.diff(yedges))
    hist_dens = counts / areas
    print("max(dens): {}".format(sdf_dens.max()))
    print("max(hist): {}".format(hist_dens.max()))

    ax2.imshow(hist_dens.T
             ,aspect='auto'
             ,extent=limlist
             ,origin='upper'
             ,norm=LogNorm(1e23,1e26)
             ,cmap=mcm.plasma)
    
    ax3 = fig.add_subplot(223)
    zcounts = ax3.hist(bunch_electron_x*1e6
           ,bins=xbins*1e6
           ,weights=bunch_electron_w*sc.e
#           ,log=True
           )[0]
    ax3.set_xlim(limits['xmin']*1e6,limits['xmax']*1e6)
#    ax3.set_ylim(1e11,5e12)

    ax3.set_xlabel(r'$z\ \mathrm{(\mu m)}$')
    ax3.set_ylabel(r'$\mathrm{d}Q\ \mathrm{(C m^{-3})}$')
    ax.xaxis.set_major_locator(mt.MaxNLocator(5))

    ax4 = fig.add_subplot(224)
    ycounts = ax4.hist(bunch_electron_y*1e6
                      ,bins=ybins*1e6
                      ,weights=bunch_electron_w*sc.e
                      )[0]

    ax4.set_xlim(limits['ymin']*1e6,limits['ymax']*1e6)
#    ax3.set_ylim(1e11,5e12)

    ax4.set_xlabel(r'$z\ \mathrm{(\mu m)}$')
    ax4.set_ylabel(r'$\mathrm{d}Q\ \mathrm{(C m^{-3})}$')

    xcentres = xedges[:-1] + np.diff(xedges)
    ycentres = yedges[:-1] + np.diff(yedges)
    xm, ym = np.meshgrid(xcentres, ycentres, indexing='ij')

    nz = xcentres[np.where(zcounts > zcounts.max()/np.e)]
    ax3.axvline(nz.min()*1e6,alpha=0.5)
    ax3.axvline(nz.max()*1e6,alpha=0.5)
    bunch_length = nz.max() - nz.min()
    print("Bunch Length: {}".format(bunch_length))

    nz = ycentres[np.where(ycounts > ycounts.max()/2)]
    ax4.axvline(nz.min()*1e6,alpha=0.5)
    ax4.axvline(nz.max()*1e6,alpha=0.5)
    bunch_width = nz.max() - nz.min()
    print("Bunch Width: {}".format(bunch_width))

    x_avg = np.average(bunch_electron_x, weights=bunch_electron_w)
    x_vari = np.average((bunch_electron_x - x_avg)**2, weights=bunch_electron_w)
    x_stdev = np.sqrt(x_vari)
    print("Bunch stdev: {}".format(x_stdev))
   
    unscaled_charge = np.sum(counts*sc.e)
    bunch_charge_rad = np.sum(np.pi*np.abs(ym)*counts*sc.e)
    bunch_charge_dep = np.sum(bunch_width*counts*sc.e)

    print("Total charge(rad): {}".format(bunch_charge_rad))
    print("Total charge(depth): {}".format(bunch_charge_dep))
    print("Unscaled charge: {}".format(unscaled_charge))


    ax3.text(0.10,0.95,r'$Q_w = {:.3}\ \mathrm{{\mu C/m}}$'.format(unscaled_charge*1e6)
           ,transform=ax3.transAxes
           )
    ax3.text(0.10,0.90,r'$Q_w = {:.3}\ \mathrm{{pC}}$'.format(bunch_charge_dep*1e12)
           ,transform=ax3.transAxes
           )
    ax3.text(0.10,0.85,r'$Q_r = {:.3}\ \mathrm{{pC}}$'.format(bunch_charge_rad*1e12)
           ,transform=ax3.transAxes
           )

    ax4.text(0.55,0.90,r'$W_b = {:.3}\ \mathrm{{\mu m}}$'.format(bunch_width*1e6)
           ,transform=ax4.transAxes
           )
    ax4.text(0.55,0.85,r'$L_b = {:.3}\ \mathrm{{\mu m}}$'.format(bunch_length*1e6)
           ,transform=ax4.transAxes
           )
    ax4.text(0.55,0.80,r'$\sigma_x = {:.3}\ \mathrm{{\mu m}}$'.format(x_stdev*1e6)
           ,transform=ax4.transAxes
           )

    fig.savefig('{}/{}_debug_fwhm.png'.format(output_path,output_name))
    

### Publication Plot

    ax1loc =[0.09,0.15,0.37,0.625] 
    ax2loc =[0.59,0.15,0.37,0.625] 
    ax1cbloc =[0.09,0.8,0.37,0.05] 
    ax2cbloc =[0.59,0.8,0.37,0.05] 
    ax1norm=LogNorm(1e23,1e26)
    ax2norm = LogNorm(1,100)

    if is3d:
        plot_dens = sdf_dens[:,:,int(0.5*sdf_dens.shape[3])]
    else:
        plot_dens = sdf_dens

    mpl.rcParams.update({'font.size': 8})
    fig = mf.Figure(figsize=(3.2,2))
    canvas = mplbea.FigureCanvasAgg(fig)
    ax = fig.add_axes(ax1loc)
    ax.imshow(plot_dens.T
             ,aspect='auto'
             ,extent=limlist
             ,origin='upper'
             ,norm=ax1norm
             ,cmap=mcm.plasma)

    ax1cba = fig.add_axes(ax1cbloc)
    ax1cb = mplcb.ColorbarBase(ax1cba
                              ,cmap=mcm.plasma
                              ,norm=ax1norm
                              ,orientation='horizontal')
    ax1cba.tick_params(top=True,labelbottom=False,labeltop=True,pad=-1)
    ax1cb.set_ticks((ax1norm.vmin,ax1norm.vmax))
    ax1cba.xaxis.set_label_position('top')
    ax1cb.set_label(r"$n_e\ \mathrm{m^{-3}}$", labelpad=-4)

    ax.xaxis.set_major_locator(mt.LinearLocator(3))
    ax.yaxis.set_major_locator(mt.LinearLocator(3))

    ax.set_xlabel(r'$z\ \mathrm{(\mu m)}$', labelpad=0)
    ax.set_ylabel(r'$y \mathrm{(\mu m)}$', labelpad=-8)

    ax2 = fig.add_axes(ax2loc)
    counts, xbins, patches = ax2.hist(bunch_electron_x*1e6
                                     ,bins=xbins*1e6
                                     ,weights=bunch_electron_w*sc.e*1e9
                                     ,linewidth=0
                                     )
    for count, patch in zip(counts,patches):
        patch.set_facecolor(mcm.plasma(ax2norm(count))) 
    
    ax2cba = fig.add_axes(ax2cbloc)
    ax2cb = mplcb.ColorbarBase(ax2cba
                              ,cmap=mcm.plasma
                              ,norm=ax2norm
                              ,orientation='horizontal')
    ax2cba.tick_params(top=True,labelbottom=False,labeltop=True,pad=-1)
    ax2cb.set_ticks((ax2norm.vmin,ax2norm.vmax))
    ax2cba.xaxis.set_label_position('top')
    ax2cb.set_label(r"$\mathrm{d}Q/\mathrm{d}z\ \mathrm{nC/m}$", labelpad=-4)

    ax2.set_xlim(*hist_limlist[:2])
    ax2.set_ylim(0,ax2norm.vmax)

    ax2.xaxis.set_major_locator(mt.LinearLocator(3))
    ax2.yaxis.set_major_locator(mt.LinearLocator(3))
    
    ax2.set_xlabel(r'$z\ \mathrm{(\mu m)}$',labelpad=0)
    ax2.set_ylabel(r'$\mathrm{d}Q/\mathrm{d}x\ \mathrm{(nC/m)}$',labelpad=-2)
    ax2.text(0.05,0.85,'$dQ = {:.3}\mathrm{{\mu C/m}}$'.format(unscaled_charge*1e6)
           ,transform=ax2.transAxes
           )

    try:
        fig.text(0.05,0.94,'{}'.format(config['title'])
        ,transform=fig.transFigure, fontsize=10)
    except Exception as err:
        print(err)

    fig.savefig('{}/{}_pub_fwhm.png'.format(output_path,output_name) ,dpi=300)
    
    

if __name__ == "__main__":
    main()
