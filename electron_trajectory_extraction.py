#!/usr/bin/env python3

import argparse
import os

import epochpy.particles as ep
import epochpy.tracking as et
import epochpy.bubble as eb
import epochpy.fields as ef
import epochpy.header as eh

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.patches as mplpat
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import numpy.lib.recfunctions as nlr
import scipy.interpolate as si
import re
import code
import pickle

intus = si.InterpolatedUnivariateSpline # because omg that is a long one!
c_0 = 299792458

def __argument_parse():

# Set up parser and global options
    parser = argparse.ArgumentParser(description='Find n fastest particles')
    parser.add_argument('-n',type=int,default=10,dest="part_n",
      metavar="N", help="Find N fastest particles")
    parser.add_argument('--species',type=str,default="electron",
      help="Species name in SDF file")
    parser.add_argument('injection_sdf',type=str,metavar="file.sdf",
      help="SDF snapshot with injected bunches")
    parser.add_argument('--select-uninjected',type=str,metavar="file.sdf",
      dest="uninjected_sdf",
      help="SDF snapshot with uninjected electrons of interest")
    parser.add_argument("--gamma_min",type=float,default=2.0,metavar="gamma",
      help="Minimum electron gamma to consider part of bunch")
    parser.add_argument('--clim',nargs=2,type=float,default=(1e24,1e27),
      help="color scale limits")

    return(parser.parse_args())

def find_n_fastest_ids(sdf_file, n, species):
    """
    Returns the n particles from species with the largest value of Px
    found in sdf_file.
    """
    particle_data = ep.species_from_file(sdf_file,species) 
    particle_data = particle_data[particle_data['Px'].argsort()]
    n_fastest = particle_data[:-n]

    return(n_fastest)

def update_ellipse(ax, ellid,  mem={}):
    centre_x = float(input("ROI x centre (um): "))*1e-6
    centre_y = float(input("ROI y centre (um): "))*1e-6
    semi_x = float(input("ROI x semi-axis (um): "))*1e-6
    semi_y = float(input("ROI y semi-axis (um): "))*1e-6
   
    try:
       memo = mem[ax]
    except KeyError:
        mem[ax] = {}
        memo = mem[ax]

    try:
        memo[ellid].center = (centre_x,centre_y)
        memo[ellid].width = 2*semi_x
        memo[ellid].height = 2*semi_y
    except KeyError:
        ell = mplpat.Ellipse((centre_x,centre_y),2*semi_x,2*semi_y, alpha = 0.5)
        ax.add_patch(ell)
        memo[ellid] = ell

    ax.get_figure().canvas.draw()

    if input("Are we done here? (Y/N) ") == "n":
        centre_x, centre_y,semi_x,semi_y = update_ellipse(ax, ellid)
    return(centre_x,centre_y,semi_x,semi_y)


def build_particle_trajectories_by_id_from_files(sdf_list, particle_ids):
    
    particles_by_file = [ (sdf_file, particles_by_id_from_file(sdf_file,
                                      particle_ids)) \
                         for sdf_file in sdf_list ]
    

def __main():
    args = __argument_parse()

    sdf_list = [f for f in os.listdir(os.getcwd()) if f.endswith('.sdf')]


#### Selection of injected particles ####

    grid = ef.grid_from_file(args.injection_sdf) 
    try:
        dens = ef.field_from_file(args.injection_sdf,"Derived_Number_Density_electron") 
    except:
        bins = ef.field_from_file(args.injection_sdf,"Grid_Grid")
        pos = ef.field_from_file(args.injection_sdf,"Grid_Particles_electron")
        weight = ef.field_from_file(args.injection_sdf,"Particles_Weight_electron")
        area = (grid['x'][1] - grid['x'][0])*(grid['y'][1]-grid['y'][0])
        dens,binsx,binsy = np.histogram2d(pos[0],pos[1],bins=bins,weights=weight)
        dens /= area
        


    fig = plt.figure()
    ax = []
    im = []

    ax.append(fig.add_subplot(1,1,1))
    im.append(ax[0].imshow(dens.T, origin='lower', vmin=min(args.clim), vmax=max(args.clim),
        norm=LogNorm(), cmap=plt.get_cmap('CMRmap'), aspect='auto', 
        extent=[grid['x'].min(), grid['x'].max(), grid['y'].min(), grid['y'].max()]))
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(im[0], cax=cax)
    ax[0].set_title("$time t={0}\mathrm{{s}}$".format(eh.get_time(args.injection_sdf)))
    ax[0].set_xlabel("$x\  \mathrm{m}$")
    ax[0].set_ylabel("$y\  \mathrm{m}$")
    cbar.set_label("$n_e\  \mathrm{m^{-3}}$", rotation=270)
    fig.show()
    
    print("########################################################")
    print("#                                                      #")
    print("#             Injected particle selection...           #")
    print("#                                                      #")
    print("########################################################")

    num_bunches = int(input("How many bunches? "))

    inj_ells = []
    inj_ell_pars = ["centre_x", "centre_y", "semi_x", "semi_y"]
    for i in range(num_bunches):
        inj_elldat = dict(zip(inj_ell_pars,update_ellipse(ax[0], i)))
        inj_ells.append(inj_elldat)

    plt.close(fig)


#### Uninjected particle selection ####

    if args.uninjected_sdf is not None:

        grid = ef.grid_from_file(args.uninjected_sdf) 
        try:
            dens = ef.field_from_file(args.uninjected_sdf,"Derived_Number_Density_electron") 
        except:
            bins = ef.field_from_file(args.uninjected_sdf,"Grid_Grid")
            pos = ef.field_from_file(args.uninjected_sdf,"Grid_Particles_electron")
            weight = ef.field_from_file(args.uninjected_sdf,"Particles_Weight_electron")
            area = (grid['x'][1] - grid['x'][0])*(grid['y'][1]-grid['y'][0])
            dens,binsx,binsy = np.histogram2d(pos[0],pos[1],bins=bins,weights=weight)
            dens /= area
     

        fig = plt.figure()
        ax = []
        im = []

        ax.append(fig.add_subplot(1,1,1))
        im.append(ax[0].imshow(dens.T, origin='lower', vmin=min(args.clim), vmax=max(args.clim),
            norm=LogNorm(), cmap=plt.get_cmap('CMRmap'), aspect='auto', 
            extent=[grid['x'].min(), grid['x'].max(), grid['y'].min(), grid['y'].max()]))
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = plt.colorbar(im[0], cax=cax)
        ax[0].set_title("$time t={0}\mathrm{{s}}$".format(eh.get_time(args.uninjected_sdf)))
        ax[0].set_xlabel("$x\  \mathrm{m}$")
        ax[0].set_ylabel("$y\  \mathrm{m}$")
        cbar.set_label("$n_e\  \mathrm{m^{-3}}$", rotation=270)
        fig.show()
       
        print("########################################################")
        print("#                                                      #")
        print("#            Uninjected particle selection...          #")
        print("#                                                      #")
        print("########################################################")

        target_beta = float(input("Target beta? "))
        target_width = float(input("Target beta (half-)interval? "))

        num_bunches = int(input("How many areas? "))

        uninj_ells = []
        uninj_ell_pars = ["centre_x", "centre_y", "semi_x", "semi_y"]
        for i in range(num_bunches):
            uninj_elldat = dict(zip(uninj_ell_pars,update_ellipse(ax[0], i)))
            uninj_ells.append(uninj_elldat)

        plt.close(fig)


#### Injected particle id selection ###

    particles = ep.species_from_file(args.injection_sdf,'electron')

    # use maximum ellipse bounds to trim particles to a box before we start
    # do this because the numpy routines for this are super fast compared to
    # the comparisons we will do later

    # first find the bounding box edges
    minx = min([x['centre_x'] - x['semi_x'] for x in inj_ells])
    maxx = max([x['centre_x'] + x['semi_x'] for x in inj_ells])
    miny = min([x['centre_y'] - x['semi_y'] for x in inj_ells])
    maxy = max([x['centre_y'] + x['semi_y'] for x in inj_ells])

    #select in x
    preselected_particles = particles[
        np.logical_and(particles['Grid0'] < maxx, particles['Grid0'] > minx)]
    #select in y
    preselected_particles = preselected_particles[
        np.logical_and(preselected_particles['Grid1'] < maxy, 
            preselected_particles['Grid1'] > miny)]
    # N.B don't compound these, we'd lose time doing the y bounds search over 
    # the whole x domain, better to clip x first then search y

    # now we can check inside the ellipses.  There is no speed difference
    # between a naive ellipse-outmost vs particle-outmost loop nesting, but 
    # a particle-outmost loop nesting *does* allow us to skip checks once an
    # initial ellipse match is made...

    print("preselected {} particles".format(len(preselected_particles)))

    inj_ellrange = range(len(inj_ells))
    selected_particles = [ [] for a in inj_ellrange]

    for part in preselected_particles:
        for eln in inj_ellrange:
            inj_ell = inj_ells[eln]
            inj_ellc = (((part['Grid0'] - inj_ell['centre_x'])**2/ inj_ell['semi_x']**2)
                + ((part['Grid1'] - inj_ell['centre_y'])**2 / inj_ell['semi_y']**2))
            if inj_ellc < 1.0:
                selected_particles[eln].append(part)
                break
   
    mom_thresh = np.sqrt(args.gamma_min**2 - 1)*( 9.10938356e-31 * c_0)

    selected_particles = [np.vstack(
        [p for p in plist if p['Px'] > mom_thresh]) 
        for plist in selected_particles]
        

    for pn in range(len(selected_particles)):
        fig = plt.figure()
        ax = []
        im = []

        pplt = selected_particles[pn]
        print("{} candidate particles in bunch {}".format(pplt.shape[0],pn))

        ax.append(fig.add_subplot(1,1,1))
        im.append(ax[0].plot(pplt['Grid0'],pplt['Grid1'],'.'))
        ax[0].set_title("$time t={0}\mathrm{{s}}$".format(eh.get_time(args.injection_sdf)))
        ax[0].set_xlabel("$x\  \mathrm{m}$")
        ax[0].set_ylabel("$y\  \mathrm{m}$")
        plt.savefig("output/bunch{}".format(pn))
        plt.close(fig)


    inj_metadata = { 
        'gamma_min' : args.gamma_min,
        'ellipses' : inj_ells,
        'sdf_snapshot' : args.injection_sdf }
        

    print("Writing PIDS for injected particles to injected_pids.pickle...")
    with open("injected_pids.pickle", 'wb') as pp:
        pickle.dump((inj_metadata,selected_particles),pp)

    final_selection = [z for x in selected_particles for y in x['ID'].tolist()
        for z in y]


#### Uninjected particle id selection ###

    if args.uninjected_sdf is not None:

        uninj_particles = ep.species_from_file(args.uninjected_sdf,'electron')

        # use maximum ellipse bounds to trim particles to a box before we start
        # do this because the numpy routines for this are super fast compared to
        # the comparisons we will do later

        # first find the bounding box edges
        minx = min([x['centre_x'] - x['semi_x'] for x in uninj_ells])
        maxx = max([x['centre_x'] + x['semi_x'] for x in uninj_ells])
        miny = min([x['centre_y'] - x['semi_y'] for x in uninj_ells])
        maxy = max([x['centre_y'] - x['semi_y'] for x in uninj_ells])

        #select in x
        preselected_uninj_particles = uninj_particles[
            np.logical_and(uninj_particles['Grid0'] < maxx, uninj_particles['Grid0'] > minx)]
        #select in y
        preselected_uninj_particles = preselected_uninj_particles[
            np.logical_and(preselected_uninj_particles['Grid1'] < maxy, 
                preselected_uninj_particles['Grid1'] > miny)]
        # N.B don't compound these, we'd lose time doing the y bounds search over 
        # the whole x domain, better to clip x first then search y

        # now we can check inside the ellipses.  There is no speed difference
        # between a naive ellipse-outmost vs particle-outmost loop nesting, but 
        # a particle-outmost loop nesting *does* allow us to skip checks once an
        # initial ellipse match is made...

        print("preselected {} uninj_particles".format(len(preselected_uninj_particles)))

        uninj_ellrange = range(len(uninj_ells))
        selected_uninj_particles = [ [] for a in uninj_ellrange]

        for part in preselected_uninj_particles:
            for eln in uninj_ellrange:
                uninj_ell = uninj_ells[eln]
                uninj_ellc = (((part['Grid0'] - uninj_ell['centre_x'])**2/ uninj_ell['semi_x']**2)
                    + ((part['Grid1'] - uninj_ell['centre_y'])**2 / uninj_ell['semi_y']**2))
                if uninj_ellc < 1.0:
                    selected_uninj_particles[eln].append(part)
                    break
      
        beta_min = target_beta - target_width
        beta_min = beta_min if beta_min >=0 else 0
        beta_max = target_beta + target_width
        beta_max = beta_max if beta_max < 1.0 else 1.0 - 1e-10

        print(beta_min)
        print(beta_max)

        mom_min = 9.10938356e-31 * c_0 * (beta_min/np.sqrt(1-beta_min**2))
        mom_max = 9.10938356e-31 * c_0 * (beta_max/np.sqrt(1-beta_max**2))

        try:
            selected_uninj_particles = [np.vstack(
                [p for p in plist if p['Px'] > mom_min]) 
                for plist in selected_uninj_particles]

            selected_uninj_particles = [np.vstack(
                [p for p in plist if p['Px'] < mom_max]) 
                for plist in selected_uninj_particles]
            
            final_uninj_selection = [z for x in selected_uninj_particles for y in x['ID'].tolist()
                for z in y]

            final_uninj_selection =  np.setdiff1d(final_uninj_selection,final_selection).tolist()
        except:
            code.interact(local=locals())

        uninj_metadata = { 
        'beta_min' : beta_min,
        'beta_max' : beta_max,
        'ellipses' : uninj_ells,
        'sdf_snapshot' : args.uninjected_sdf }

        print("{} candidate uninj_particles".format(len(final_uninj_selection)))


### get injected particle trajectories ###

    part_traj = et.get_trajectories_by_id(sdf_list,'electron',final_selection)
    [ x.sort(order='time') for x in part_traj.values()]
    for part in part_traj:
        part_data = part_traj[part]
        label_append = []
        data_append = []
        if 'Gamma' not in part_data.dtype.names:
            gamma_temp = np.sqrt( 1 + 
                (np.power(part_data['Px'],2) + np.power(part_data['Py'],2))
                / (9.10938356e-31 * c_0)**2 )
            label_append.append('Gamma')
            data_append.append(gamma_temp)
        if 'Vx' in part_data.dtype.names:
            betax_temp = part_data['Vx']/c_0
        else:
            if 'Gamma' in part_data.dtype.names:
                betax_temp = part_data['Px']/(9.10938356e-31 *
                 part_data['Gamma'] * c_0)
            else:
                betax_temp = part_data['Px']/(9.10938356e-31 * gamma_temp *c_0)
        label_append.append("Betax")
        data_append.append(betax_temp)
        part_traj[part] = nlr.append_fields(part_data, label_append,
         data_append)

 

#### Get uninjected particle trajectories ####

    if args.uninjected_sdf is not None:

        uninj_part_traj =  et.get_trajectories_by_id(sdf_list,'electron',final_uninj_selection)
        [ x.sort(order='time') for x in uninj_part_traj.values()]
        for part in uninj_part_traj:
            part_data = uninj_part_traj[part]
            label_append = []
            data_append = []
            if 'Gamma' not in part_data.dtype.names:
                gamma_temp = np.sqrt( 1 + 
                    (np.power(part_data['Px'],2) + np.power(part_data['Py'],2))
                    / (9.10938356e-31 * c_0)**2 )
                label_append.append('Gamma')
                data_append.append(gamma_temp)
            if 'Vx' in part_data.dtype.names:
                betax_temp = part_data['Vx']/c_0
            else:
                if 'Gamma' in part_data.dtype.names:
                    betax_temp = part_data['Px']/(9.10938356e-31 *
                     part_data['Gamma'] * c_0)
                else:
                    betax_temp = part_data['Px']/(9.10938356e-31 * gamma_temp *c_0)
            label_append.append("Betax")
            data_append.append(betax_temp)
            uninj_part_traj[part] = nlr.append_fields(part_data, label_append,
             data_append)

        with open('uninjected.pickle', 'wb') as pp:
            pickle.dump((uninj_metadata,uninj_part_traj), pp, protocol=2)

   
    with open('injected.pickle', 'wb') as bp:
        pickle.dump((inj_metadata,part_traj), bp, protocol=2)

if __name__== "__main__":
    __main()
