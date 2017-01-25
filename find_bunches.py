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
import matplotlib.colors as mc
import matplotlib.ticker as mt

import warnings

warnings.simplefilter('error', UserWarning)

class AttributeSearchError(AttributeError):
    
    def __init__(self, message, attr):
        super(AttributeError, self).__init__(message)

        self.attr = attr


def getattr_from_any(handles, attr):
    for handle in handles:
        try:
            data = getattr(handle, attr)
        except:
            continue
        else:
            return(data)

    raise AttributeSearchError("No data {} found".format(attr), attr)


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
    dirname = os.path.basename(os.getcwd())

    config_file = parse_input()
    config = loadconfig(config_file)

    sdf_handles = []
    sdf_files = config['sdf_files']
    if type(sdf_files) is not type([]):
        sdf_files = [sdf_files,]

    for sdf_file in sdf_files:
        try:
            sdf_data = sdf.read(sdf_file)
        except:
            print("Warning: Failed to open '{}'".format(sdf_file))
        else:
            sdf_handles.append(sdf_data)

    if len(sdf_handles) < 1:
        print("Error: Unable to open any sdf files")
        print("Exiting....")
        return(-1)


    grids =  {'xgrid':None
                ,'ygrid':None
                ,'zgrid':None
                ,'xbins':None
                ,'ybins':None
                ,'zbins':None}
    for grid in grids:
        try:
            grids[grid] = float(config[grid])
        except:
            pass

    if ((grids['xgrid'] is not None and grids['xbins'] is not None) or
       (grids['ygrid'] is not None and grids['ybins'] is not None) or
       (grids['zgrid'] is not None and grids['zbins'] is not None)):
        print("Both gridsize and numbins are specified... "
              "gridsize will take precedence")

    if ((grids['xgrid'] is None and grids['xbins'] is None) or
       (grids['ygrid'] is None and grids['ybins'] is None) or
       (grids['zgrid'] is None and grids['zbins'] is None)):
        print("Warning no gridsize or numbins specified, defaulting to "
              "100 bins")
        if grids['xbins'] is None:
            grids['xbins'] = 100
        if grids['ybins'] is None:
            grids['ybins'] = 100
        if grids['zbins'] is None:
            grids['zbins'] = 100
    
    try:
        output_name = config['output_name']
    except:
        output_name = 'out'

    try:
        output_path = config['output_path']
    except:
        output_path = '.'

    cutoff_px = float(config['cutoff_px']) * sc.m_e * sc.c
     
    #load sdf data (needs to be done before we can sanity check input)
    sdf_e_px = getattr_from_any(sdf_handles, 'Particles_Px_electron').data
    sdf_e_x = getattr_from_any(sdf_handles, 'Grid_Particles_electron').data[0]
    sdf_e_y = getattr_from_any(sdf_handles, 'Grid_Particles_electron').data[1]
    sdf_e_w = getattr_from_any(sdf_handles, 'Particles_Weight_electron').data
    sdf_gridx = getattr_from_any(sdf_handles, 'Grid_Grid').data[0]
    sdf_gridy = getattr_from_any(sdf_handles, 'Grid_Grid').data[1]
    sdf_dens = getattr_from_any(sdf_handles, 'Derived_Number_Density_electron').data
    try:
        sdf_gridz = getattr_from_any(sdf_handles, 'Grid_Grid').data[2]
        sdf_e_z = getattr_from_any(sdf_handles, 'Grid_Particles_electron').data[2]
        is3d = True
    except:
        is3d = False

    extents = {'xmin':sdf_e_x.min()
             ,'xmax':sdf_e_x.max()
             ,'ymin':sdf_e_y.min()
             ,'ymax':sdf_e_y.max()}
    if is3d:
        extents['zmin'] = sdf_e_z.min()
        extents['zmax'] = sdf_e_z.max()
 
    #parse in spatial limits and sanity check
    limits = {}
    for extent in extents:
        try:
            limits[extent] = float(config[extent])
        except:
            limits[extent] = extents[extent]
        
    if limits['xmax'] < limits['xmin']:
        limits['xmin'], limits['xmax'] = limits['xmax'], limits['xmin'] 
    if limits['ymax'] < limits['ymin']:
        limits['ymin'], limits['ymax'] = limits['ymax'], limits['ymin'] 
    if limits['zmax'] < limits['zmin']:
        limits['zmin'], limits['zmax'] = limits['zmax'], limits['zmin'] 

    for extent in limits:
        if extent.endswith('max'):
            if limits[extent] > extents[extent]:
                limits[extent] = extents[extent]
                print("Warning {} out of range, ignoring".format(extent))
        else:
            if limits[extent] < extents[extent]:
                limits[extent] = extents[extent]
                print("Warning {} out of range, ignoring".format(extent))

    #parse in ellipse parameters and sanity check
    ellipse_sane = False
    ell = {'centerx':None, 'radx':None, 
           'centery':0.0, 'rady':None}
    if is3d:
        ell['centerz'] = 0.0
        ell['radz'] = None
    for arg in ell:
        try:
            ell[arg] = float(config['ell_{}'.format(arg)])
        except:
            pass
    if None not in ell.values():
        if ((ell['centerx'] + ell['radx'] < limits['xmin']) 
           or (ell['centerx'] - ell['radx'] > limits['xmax'])
           or (ell['centery'] + ell['rady'] < limits['ymin']) 
           or (ell['centery'] - ell['rady'] > limits['ymax'])
           or (ell['centerz'] + ell['radz'] < limits['zmin']) 
           or (ell['centerz'] - ell['radz'] > limits['zmax'])):
            print("Error, ellipse is entirely outside of view window")
            return(-1)
        if ((ell['centerx'] - ell['radx'] < limits['xmin']) 
           or (ell['centerx'] + ell['radx'] > limits['xmax'])
           or (ell['centery'] - ell['rady'] < limits['ymin']) 
           or (ell['centery'] + ell['rady'] > limits['ymax'])
           or (ell['centerz'] - ell['radz'] < limits['zmin']) 
           or (ell['centerz'] + ell['radz'] > limits['zmax'])):
            print("Warning, ellipse is clipped by view window")
        ellipse_sane = True

    #trim grid data to specfied limits
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
        sdf_dens = sdf_dens[:,:,zargmin:zargmax]

    #take on axis density slice for plotting
    if is3d:
        plot_dens_xy = sdf_dens[:,:,int(0.5*sdf_dens.shape[2])]
        plot_dens_xz = sdf_dens[:,int(0.5*sdf_dens.shape[1]),:]
    else:
        plot_dens_xy = sdf_dens


    ### Electron selection magic happens here

    energy_mask = sdf_e_px > cutoff_px
    position_mask = np.full(sdf_e_px.shape, True, dtype=bool)
    if ellipse_sane:
        if is3d:
            position_mask = ( 
            ((sdf_e_x - ell['centerx'])**2 / (ell['radx']**2)) +
            ((sdf_e_y - ell['centery'])**2 / (ell['rady']**2)) +
            ((sdf_e_z - ell['centerz'])**2 / (ell['radz']**2)) < 1 )
        else:
            position_mask = (
            ((sdf_e_x - ell['centerx'])**2 / (ell['radx']**2)) +
            ((sdf_e_y - ell['centery'])**2 / (ell['rady']**2)) < 1 )

    bunch_electron_mask = np.where(np.logical_and(energy_mask,
                                                position_mask))


    bunch_electron_x = sdf_e_x[bunch_electron_mask].reshape(-1)
    bunch_electron_y = sdf_e_y[bunch_electron_mask].reshape(-1)
    bunch_electron_w = sdf_e_w[bunch_electron_mask].reshape(-1)
    if is3d:
        bunch_electron_z = sdf_e_z[bunch_electron_mask].reshape(-1)

    print("Found {} particles meeting criteria".format(len(bunch_electron_w)))


### Histogram Grid creation ###

    if grids['xgrid'] is not None:
        xbins = np.arange(limits['xmin'],limits['xmax']+grids['xgrid'],grids['xgrid'])
    else:
        xbins = np.linspace(limits['xmin'],limits['xmax'],grids['xbins'])

    if grids['ygrid'] is not None:
        if limits['ymin'] * limits['ymax'] < 0:
            ybins = np.sort(np.concatenate((
                np.arange(0,limits['ymin']-grids['ygrid'],-grids['ygrid']),
                np.arange(grids['ymin'],limits['ymax']+grids['ygrid'],grids['ygrid']))))

        else:
            ybins = np.arange(limits['ymin'],limits['ymax']+grids['ygrid'],grids['ygrid'])
    else:
        if limits['ymin'] * limits['ymax'] < 0:
            ybins = np.sort(np.concatenate((
                np.linspace(limits['ymin'],0,grids['ybins']//2),
                np.linspace(0,limits['ymax'],grids['ybins']//2)[1:])))
        else:
            ybins = np.linspace(limits['ymin'],limits['ymax'],grids['ybins'])
    
    if is3d:
        if grids['zgrid'] is not None:
            if limits['zmin'] * limits['zmax'] < 0:
                zbins = np.sort(np.concatenate((
                    np.arange(0,limits['zmin']-grids['zgrid'],-grids['zgrid']),
                    np.arange(grids['zmin'],limits['zmax']+grids['zgrid'],grids['zgrid']))))

            else:
                zbins = np.arange(limits['zmin'],limits['zmax']+grids['zgrid'],grids['zgrid'])
        else:
            if limits['zmin'] * limits['zmax'] < 0:
                zbins = np.sort(np.concatenate((
                    np.linspace(limits['zmin'],0,grids['zbins']//2),
                    np.linspace(0,limits['zmax'],grids['zbins']//2)[1:])))
            else:
                zbins = np.linspace(limits['zmin'],limits['zmax'],grids['ybins'])

    
    
    display_limits = [ limits[l]*1e6 for l in ['xmin','xmax','ymin','ymax'] ]


### Histogram Creation ###

    if is3d:
        pos_data_3d = np.column_stack([bunch_electron_x
                                      ,bunch_electron_y
                                      ,bunch_electron_z])
     
        counts3d, histedges = np.histogramdd(pos_data_3d
                                          ,bins=[xbins,ybins,zbins]
                                          ,weights=bunch_electron_w)

        vols3d = np.einsum('i,j,k',*[np.diff(a) for a in histedges])
        hist_dens3d = counts3d / vols3d

        counts2d_xy = np.sum(counts3d, axis=2)
        counts2d_xz = np.sum(counts3d, axis=1)
        areas2d_xz = np.outer(np.diff(histedges[0]),np.diff(histedges[2]))
        hist_dens2d_xz = counts2d_xz / areas2d_xz
        
        counts1d_z = np.sum(counts3d, axis=(0,1))

        print("max(hist_dens3d): {}".format(hist_dens3d.max()))

    else:
        pos_data_2d = np.column_stack([bunch_electron_x
                                      ,bunch_electron_y])

        counts2d_xy, histedges = np.histogramdd(pos_data_2d
                                             ,bins=[xbins,ybins]
                                             ,weights=bunch_electron_w)

    areas2d_xy = np.outer(np.diff(histedges[0]),np.diff(histedges[1]))
    hist_dens2d_xy = counts2d_xy / areas2d_xy

    counts1d_x = np.sum(counts2d_xy,axis=1)
    counts1d_y = np.sum(counts2d_xy,axis=0)

    print("max(hist_dens2d): {}".format(hist_dens2d_xy.max()))
    print("max(sdf_dens): {}".format(sdf_dens.max()))


### Statistical Calculations

    bin_w_x = np.diff(histedges[0])
    bin_ctr_x = histedges[0][:-1] + bin_w_x 
    bin_w_y = np.diff(histedges[1])
    bin_ctr_y = histedges[1][:-1] + bin_w_y
    if is3d:
        bin_w_z = np.diff(histedges[2])
        bin_ctr_z = histedges[2][:-1] + bin_w_z

    nx_rms = bin_ctr_x[np.where(counts1d_x > counts1d_x.max()/np.e)]
    nx_fwhm = bin_ctr_x[np.where(counts1d_x > counts1d_x.max()/2)]
    nx_fw10m = bin_ctr_x[np.where(counts1d_x > counts1d_x.max()/100)]
    rms_x = nx_rms.max() - nx_rms.min()
    fwhm_x = nx_fwhm.max() - nx_fwhm.min() 
    fw10m_x = nx_fw10m.max() - nx_fw10m.min() 
    x_avg = np.average(bunch_electron_x, weights=bunch_electron_w)
    x_vari = np.average((bunch_electron_x - x_avg)**2, weights=bunch_electron_w)
    x_stdev = np.sqrt(x_vari)
    print("Bunch RMS(x): {}".format(rms_x))
    print("Bunch FWHM(x): {}".format(fwhm_x))
    print("Bunch FW10M(x): {}".format(fw10m_x))
    print("Bunch pos stdev(x): {}".format(x_stdev))
    print()

    ny_rms = bin_ctr_y[np.where(counts1d_y > counts1d_y.max()/np.e)]
    ny_fwhm = bin_ctr_y[np.where(counts1d_y > counts1d_y.max()/2)]
    ny_fw10m = bin_ctr_y[np.where(counts1d_y > counts1d_y.max()/100)]
    rms_y = ny_rms.max() - ny_rms.min()
    fwhm_y = ny_fwhm.max() - ny_fwhm.min() 
    fw10m_y = ny_fw10m.max() - ny_fw10m.min() 
    y_avg = np.average(bunch_electron_y, weights=bunch_electron_w)
    y_vari = np.average((bunch_electron_y - y_avg)**2, weights=bunch_electron_w)
    y_stdev = np.sqrt(y_vari)
    print("Bunch RMS(y): {}".format(rms_y))
    print("Bunch FWHM(y): {}".format(fwhm_y))
    print("Bunch FW10M(y): {}".format(fw10m_y))
    print("Bunch pos stdev(y): {}".format(y_stdev))
    print()

    if is3d:
        nz_rms = bin_ctr_z[np.where(counts1d_z > counts1d_z.max()/np.e)]
        nz_fwhm = bin_ctr_z[np.where(counts1d_z > counts1d_z.max()/2)]
        nz_fw10m = bin_ctr_z[np.where(counts1d_z > counts1d_z.max()/100)]
        rms_z = nz_rms.max() - nz_rms.min()
        fwhm_z = nz_fwhm.max() - nz_fwhm.min() 
        fw10m_z = nz_fw10m.max() - nz_fw10m.min() 
        z_avg = np.average(bunch_electron_z, weights=bunch_electron_w)
        z_vari = np.average((bunch_electron_z - z_avg)**2, weights=bunch_electron_w)
        z_stdev = np.sqrt(z_vari)
        print("Bunch RMS(z): {}".format(rms_z))
        print("Bunch FWHM(z): {}".format(fwhm_z))
        print("Bunch FW10M(z): {}".format(fw10m_z))
        print("Bunch pos stdev(z): {}".format(z_stdev))
        print()

    unscaled_charge = np.sum(counts3d)*sc.e
    if not is3d:
        print("Unscaled charge: {}".format(unscaled_charge))
        bunch_charge_rad = np.sum(np.abs(ym)*counts)*np.pi*sc.e
        bunch_charge_dep = unscaled_charge*bunch_width*sc.e
        print("Total charge(rad): {:03g}pc".format(bunch_charge_rad*1e12))
        print("Total charge(depth): {:03g}pc".format(bunch_charge_dep*1e12))
    else:
        print("Total charge: {:03g}pc".format(unscaled_charge*1e12))


### Debug Plot ####


    fig = mf.Figure(figsize=(12,12))
    canvas = mplbea.FigureCanvasAgg(fig)
    ax1 = fig.add_subplot(331)
    ax1.imshow(plot_dens_xy.T
             ,aspect='auto'
             ,extent=display_limits
             ,origin='upper'
             ,norm=mc.LogNorm(1e23,1e26)
             ,cmap=mcm.plasma)

    if None not in ell.values():
        ax1.add_patch(mpat.Ellipse((ell['centerx']*1e6,ell['centery']*1e6)
                                  ,2*ell['radx']*1e6
                                  ,2*ell['rady']*1e6
                                  ,fill=True
                                  ,fc='blue'
                                  ,alpha=0.2))

    ax2 = fig.add_subplot(334)
    ax2.imshow(plot_dens_xz.T
             ,aspect='auto'
             ,extent=display_limits
             ,origin='upper'
             ,norm=mc.LogNorm(1e23,1e26)
             ,cmap=mcm.plasma)

    if None not in ell.values():
        ax2.add_patch(mpat.Ellipse((ell['centerx']*1e6,ell['centery']*1e6)
                                  ,2*ell['radx']*1e6
                                  ,2*ell['radz']*1e6
                                  ,fill=True
                                  ,fc='blue'
                                  ,alpha=0.2))


    ax3 = fig.add_subplot(332)
    ax3.imshow(hist_dens2d_xy.T
             ,aspect='auto'
             ,extent=display_limits
             ,origin='upper'
             ,norm=mc.LogNorm(np.ma.masked_equal(hist_dens2d_xy,0,copy=False).min()
                          ,np.ma.masked_equal(hist_dens2d_xy,0,copy=False).max())
             ,cmap=mcm.plasma)

    if is3d:
        ax4 = fig.add_subplot(335)
        ax4.imshow(hist_dens2d_xz.T
                 ,aspect='auto'
                 ,extent=display_limits
                 ,origin='upper'
                 ,norm=mc.LogNorm(np.ma.masked_equal(hist_dens2d_xz,0,copy=False).min()
                              ,np.ma.masked_equal(hist_dens2d_xz,0,copy=False).max())
                 ,cmap=mcm.plasma)

    # Charge x-distribution    
    ax5 = fig.add_subplot(337)
    bars = ax5.bar(bin_ctr_x*1e6
                  ,counts1d_x
                  ,width=bin_w_x*1e6
                  ,linewidth=0
                  )
    mask_count = np.ma.masked_equal(counts1d_x, 0, copy=False)
    norm = mc.Normalize(mask_count.min(),mask_count.max())
    for i,patch in enumerate(bars.patches):
        if counts1d_x[i] > 0:
            patch.set_facecolor(mcm.plasma(norm(counts1d_x[i])))

    #Indicate extents of bunch measurements
    ax5.axvline(nx_rms.min()*1e6,alpha=0.5, color='red')
    ax5.axvline(nx_rms.max()*1e6,alpha=0.5, color='red')
    ax5.axvline(nx_fwhm.min()*1e6,alpha=0.5, color='green')
    ax5.axvline(nx_fwhm.max()*1e6,alpha=0.5, color='green')
    ax5.axvline(nx_fw10m.min()*1e6,alpha=0.5, color='blue')
    ax5.axvline(nx_fw10m.max()*1e6,alpha=0.5, color='blue')
    
    #naive attempt to autoscale
    delta = 0.1*(nx_fw10m.max() - nx_fw10m.min())
    ax5.set_xlim((nx_fw10m.min()-delta)*1e6
                ,(nx_fw10m.max()+delta)*1e6)

    ax5.set_xlabel(r'$z\ \mathrm{(\mu m)}$')
    ax5.set_ylabel(r'$\mathrm{d}Q\ \mathrm{(C m^{-3})}$')
    ax5.xaxis.set_major_locator(mt.MaxNLocator(5))


    #Charge y_distribution
    ax6 = fig.add_subplot(333)
    bars = ax6.bar(bin_ctr_y*1e6
                  ,counts1d_y
                  ,width=bin_w_y*1e6
                  ,linewidth=0
                  )
    norm = mc.Normalize(counts1d_y.min(),counts1d_y.max())
    for i,patch in enumerate(bars.patches):
        if counts1d_y[i] > 0:
            patch.set_facecolor(mcm.plasma(norm(counts1d_y[i])))
    
    #Indicate extents of bunch measurements
    ax6.axvline(ny_rms.min()*1e6,alpha=0.5, color='red')
    ax6.axvline(ny_rms.max()*1e6,alpha=0.5, color='red')
    ax6.axvline(ny_fwhm.min()*1e6,alpha=0.5, color='green')
    ax6.axvline(ny_fwhm.max()*1e6,alpha=0.5, color='green')
    ax6.axvline(ny_fw10m.min()*1e6,alpha=0.5, color='blue')
    ax6.axvline(ny_fw10m.max()*1e6,alpha=0.5, color='blue')
    
    #naive attempt to autoscale
    delta = 0.1*(ny_fw10m.max() - ny_fw10m.min())
    ax6.set_xlim((ny_fw10m.min()-delta)*1e6
                ,(ny_fw10m.max()+delta)*1e6)

    ax6.set_xlabel(r'$z\ \mathrm{(\mu m)}$')
    ax6.set_ylabel(r'$\mathrm{d}Q\ \mathrm{(C m^{-3})}$')


    #z-density distribution
    if is3d:
        ax7 = fig.add_subplot(336)
        bars = ax7.bar(bin_ctr_z*1e6
                      ,counts1d_z
                      ,width=bin_w_z*1e6
                      ,linewidth=0
                      )
        norm = mc.Normalize(counts1d_z.min(),counts1d_z.max())
        for i,patch in enumerate(bars.patches):
            if counts1d_z[i] > 0:
                patch.set_facecolor(mcm.plasma(norm(counts1d_z[i])))

        #Indicate extents of bunch measurements
        ax7.axvline(nz_rms.min()*1e6,alpha=0.5, color='red')
        ax7.axvline(nz_rms.max()*1e6,alpha=0.5, color='red')
        ax7.axvline(nz_fwhm.min()*1e6,alpha=0.5, color='green')
        ax7.axvline(nz_fwhm.max()*1e6,alpha=0.5, color='green')
        ax7.axvline(nz_fw10m.min()*1e6,alpha=0.5, color='blue')
        ax7.axvline(nz_fw10m.max()*1e6,alpha=0.5, color='blue')
        
        #naive attempt to autoscale
        delta = 0.1*(nz_fw10m.max() - nz_fw10m.min())
        ax7.set_xlim((nz_fw10m.min()-delta)*1e6
                    ,(nz_fw10m.max()+delta)*1e6)

        ax7.set_xlabel(r'$z\ \mathrm{(\mu m)}$')
        ax7.set_ylabel(r'$\mathrm{d}Q\ \mathrm{(C m^{-3})}$')
    
    #Finally print all the numerical results
    ax8 = fig.add_subplot(338)
    ax8.axis('off')

    # Bunch Charge
    if is3d:
        props_string = r'\noindent$Q = {:.3}\ \mathrm{{pC}}$\\ \\'.format(unscaled_charge*1e23)
    else:   
        props_string = ''.join(
            [r'$dQ = {:.3}\ \mathrm{{\mu C/m}}$\\'.format(unscaled_charge*1e6)
            ,r'$Q_w = {:.3}\ \mathrm{{pC}}$\\'.format(bunch_charge_dep*1e12)
            ,r'$Q_r = {:.3}\ \mathrm{{pC}}$\\ \\'.format(bunch_charge_rad*1e12)])

    props_string += ''.join(
       [r'$w_x(RMS) = {:.3}\ \mathrm{{\mu m}}$\\'.format(rms_x*1e6)
       ,r'$w_x(FWHM) = {:.3}\ \mathrm{{\mu m}}$\\'.format(fwhm_x*1e6)
       ,r'$w_x(FW10M) = {:.3}\ \mathrm{{\mu m}}$\\'.format(fw10m_x*1e6)
       ,r'$\sigma_x = {:.3}\ \mathrm{{\mu m}}$\\ \\'.format(x_stdev*1e6)
       ,r'$w_y(RMS) = {:.3}\ \mathrm{{\mu m}}$\\'.format(rms_y*1e6)
       ,r'$w_y(FWHM) = {:.3}\ \mathrm{{\mu m}}$\\'.format(fwhm_y*1e6)
       ,r'$w_y(FW10M) = {:.3}\ \mathrm{{\mu m}}$\\'.format(fw10m_y*1e6)
       ,r'$\sigma_y = {:.3}\ \mathrm{{\mu m}}$\\ \\'.format(y_stdev*1e6)])

    if is3d:
        props_string += ''.join(
           [r'$w_z(RMS) = {:.3}\ \mathrm{{\mu m}}$\\'.format(rms_z*1e6)
           ,r'$w_z(FWHM) = {:.3}\ \mathrm{{\mu m}}$\\'.format(fwhm_z*1e6)
           ,r'$w_z(FW10M) = {:.3}\ \mathrm{{\mu m}}$\\'.format(fw10m_z*1e6)
           ,r'$\sigma_z = {:.3}\ \mathrm{{\mu m}}$\\'.format(z_stdev*1e6)])


    ax8.text(0,0.95
        ,props_string
        ,transform=ax8.transAxes
        ,verticalalignment='top'
        )

    fig.savefig('{}/{}_debug_fwhm.png'.format(output_path,output_name))
    

### Publication Plot

    ax1loc=[0.09,0.15,0.37,0.625] 
    ax2loc=[0.59,0.15,0.37,0.625] 
    ax1cbloc=[0.09,0.8,0.37,0.05] 
    ax2cbloc=[0.59,0.8,0.37,0.05] 
    ax1norm=mc.LogNorm(1e23,1e26)
    ax2norm=mc.LogNorm(1,100)

    mpl.rcParams.update({'font.size': 8})
    fig = mf.Figure(figsize=(3.2,2))
    canvas = mplbea.FigureCanvasAgg(fig)
    ax = fig.add_axes(ax1loc)
    ax.imshow(plot_dens_xy.T
             ,aspect='auto'
             ,extent=display_limits
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
                                     ,weights=bunch_electron_w*sc.e*1e12
                                     ,linewidth=0)

    for count, patch in zip(counts,patches):
        if count > 1:
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

    #ax2.set_xlim(*display_limits[:2])
    #ax2.set_ylim(0,ax2norm.vmax)

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
