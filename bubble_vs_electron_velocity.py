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

def update_ellipse(ax, ellid,  memo={}):
    centre = float(input("ROI x centre (um): "))*1e-6
    semi_x = float(input("ROI x semi-axis (um): "))*1e-6
    semi_y = float(input("ROI y semi-axis (um): "))*1e-6
    
    try:
        memo[ellid].center = (centre,0)
        memo[ellid].width = 2*semi_x
        memo[ellid].height = 2*semi_y
    except KeyError:
        ell = mplpat.Ellipse((centre,0),2*semi_x,2*semi_y, alpha = 0.5)
        ax[0].add_patch(ell)
        memo[ellid] = ell

    ax[0].get_figure().canvas.draw()

    if input("Are we done here? (Y/N) ") == "n":
        centre,semi_x,semi_y = update_ellipse(ax, ellid)
    return(centre,semi_x,semi_y)


def build_particle_trajectories_by_id_from_files(sdf_list, particle_ids):
    
    particles_by_file = [ (sdf_file, particles_by_id_from_file(sdf_file,
                                      particle_ids)) \
                         for sdf_file in sdf_list ]
    

def __main():
    args = __argument_parse()

    sdf_list = [f for f in os.listdir(os.getcwd()) if f.endswith('.sdf')]
    zeropoints = eb.find_zero_points(sdf_list)
    
    xSorted = zeropoints[zeropoints[:,0].argsort()]
    tSorted = zeropoints[zeropoints[:,1].argsort()]
    interpXofT = si.InterpolatedUnivariateSpline(tSorted[:,1], tSorted[:,0], k=3)
    interpTofX = si.InterpolatedUnivariateSpline(xSorted[:,0], xSorted[:,1], k=3)
    interpVofT = interpXofT.derivative()
    def interpBofT(val):
        return interpVofT(val)/c_0
  
    plotT = np.linspace(zeropoints[:,1].min(),zeropoints[:,1].max(),
                        num=1000,endpoint=True)
    plotX = np.linspace(zeropoints[:,0].min(),zeropoints[:,0].max(),
                        num=1000,endpoint=True)


    injection_sdf = '0400.sdf'
    cmin = 1e24
    cmax = 1e27

    grid = ef.grid_from_file(injection_sdf) 
    try:
        dens = ef.field_from_file(injection_sdf,"Derived_Number_Density_electron") 
    except:
        bins = ef.field_from_file(injection_sdf,"Grid_Grid")
        pos = ef.field_from_file(injection_sdf,"Grid_Particles_electron")
        weight = ef.field_from_file(injection_sdf,"Particles_Weight_electron")
        area = (grid['x'][1] - grid['x'][0])*(grid['y'][1]-grid['y'][0])
        dens,binsx,binsy = np.histogram2d(pos[0],pos[1],bins=bins,weights=weight)
        dens /= area
        


    fig = plt.figure()
    ax = []
    im = []

    ax.append(fig.add_subplot(1,1,1))
    im.append(ax[0].imshow(dens.T, origin='lower', vmin=cmin, vmax=cmax,
        norm=LogNorm(), cmap=plt.get_cmap('CMRmap'), aspect='auto', 
        extent=[grid['x'].min(), grid['x'].max(), grid['y'].min(), grid['y'].max()]))
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(im[0], cax=cax)
    ax[0].set_title("$time t={0}\mathrm{{s}}$".format(eh.get_time(injection_sdf)))
    ax[0].set_xlabel("$x\  \mathrm{m}$")
    ax[0].set_ylabel("$y\  \mathrm{m}$")
    cbar.set_label("$n_e\  \mathrm{m^{-3}}$", rotation=270)
    fig.show()

    num_bunches = int(input("How many bunches? "))

    ells = []
    ell_pars = ["centre", "semi_x", "semi_y"]
    for i in range(num_bunches):
        elldat = dict(zip(ell_pars,update_ellipse(ax, i)))
        ells.append(elldat)

    plt.close(fig)

    particles = ep.species_from_file(injection_sdf,'electron')

    # use maximum ellipse bounds to trim particles to a box before we start
    # do this because the numpy routines for this are super fast compared to
    # the comparisons we will do later

    # first find the bounding box edges
    minx = min([x['centre'] - x['semi_x'] for x in ells])
    maxx = max([x['centre'] + x['semi_x'] for x in ells])
    miny = min([-x['semi_y'] for x in ells])
    maxy = max([x['semi_y'] for x in ells])

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

    ellrange = range(len(ells))
    selected_particles = [ [] for a in ellrange]

    for part in preselected_particles:
        for eln in ellrange:
            ell = ells[eln]
            ellc = (((part['Grid0'] - ell['centre'])**2/ ell['semi_x']**2)
                + (part['Grid1']**2 / ell['semi_y']**2))
            if ellc < 1.0:
                selected_particles[eln].append(part)
                break
   
    mom_thresh = 1 * 9.10938356e-31 * c_0 #twice the e rest momentum

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
        ax[0].set_title("$time t={0}\mathrm{{s}}$".format(eh.get_time(injection_sdf)))
        ax[0].set_xlabel("$x\  \mathrm{m}$")
        ax[0].set_ylabel("$y\  \mathrm{m}$")
        plt.savefig("output/bunch{}".format(pn))
        plt.close(fig)

    final_selection = [z for x in selected_particles for y in x['ID'].tolist()
        for z in y]
   
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

   
    with open('bubble.pickle', 'wb') as bp:
        pickle.dump(zeropoints, bp, protocol=2)

    with open('particles.pickle', 'wb') as pp:
        pickle.dump((ells,part_traj), pp, protocol=2)

if __name__== "__main__":
    __main()
