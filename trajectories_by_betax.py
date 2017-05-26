#!/usr/bin/env python3

import argparse
import os

import sdf
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
from functools import reduce

import numpy as np
import numpy.lib.recfunctions as nlr
import scipy.interpolate as si
import scipy.constants as sc
import re
import code
import pickle
import time
import logging

intus = si.InterpolatedUnivariateSpline # because omg that is a long one!

def __argument_parse():

# Set up parser and global options
    parser = argparse.ArgumentParser(description='Find particles faster than'
                                                 'beta_x')
    parser.add_argument('beta_x',type=float,metavar="beta_x",
      help="Minimum beta_x required to be selected")

    return(parser.parse_args())

def pids_faster_than_px(sdf_file, px_thresh, species):
    """
    Return a vector of all PIDs in species faster than Px_thres in sdf_file.
    """
    print("Selecting from {}".format(sdf_file))
    sdf_data = sdf.read(sdf_file)
    pid = getattr(sdf_data, 'Particles_ID_{}'.format(species)).data
    px = getattr(sdf_data, 'Particles_Px_{}'.format(species)).data
    return(pid[np.argwhere(px > px_thresh)].reshape(-1))

def __main():
    logging.basicConfig(filename="trajectories_by_betax.log",level=logging.DEBUG)
    timestamps = { 'start',time.time()}

    args = __argument_parse()

    sdf_list = [f for f in os.listdir(os.getcwd()) if f.endswith('.sdf')]

    if not (args.beta_x < 1 and args.beta_x >= 0):
        print("beta_x must be in the range 0 <= beta_x < 1")
        return(-1)

    mom_thresh = args.beta_x*sc.m_e*sc.c / np.sqrt(1-np.power(args.beta_x,2))

    selected_pid_list = [ pids_faster_than_px(f, mom_thresh, 'electron') 
                          for f in sdf_list]

    timestamps['pid_select'] = time.time()
    logging.debug('PID selection took {}'
        ''.format(timestamps['pid_select']-timestamps['start']))

    selected_pids = reduce(np.union1d, selected_pid_list)
    timestamps['pid_reduce'] = time.time()
    logging.debug('PID reduce took {}'
        ''.format(timestamps['pid_reduce']-timestamps['pid_select']))
            
### get particle trajectories ###

    print("Extracting trajectories")
    part_traj = et.get_trajectories_by_id(sdf_list,'electron',selected_pids)
    timestamps['traj_extract'] = time.time()
    logging.debug('Trajectory extraction took {}'
        ''.format(timestamps['traj_extract'] - timestamps['pid_reduce']))
#    [ x.sort(order='time') for x in part_traj.values()]
    print("calculating gamma,beta")
    for part in part_traj:
        part_data = part_traj[part]
        label_append = []
        data_append = []
        if 'Gamma' not in part_data.dtype.names:
            gamma_temp = np.sqrt( 1 + 
                (np.power(part_data['Px'],2) + np.power(part_data['Py'],2))
                / (sc.m_e * sc.c)**2 )
            label_append.append('Gamma')
            data_append.append(gamma_temp)
        if 'Vx' in part_data.dtype.names:
            betax_temp = part_data['Vx']/c_0
        else:
            if 'Gamma' in part_data.dtype.names:
                betax_temp = part_data['Px']/(9.10938356e-31 *
                 part_data['Gamma'] * sc.c)
            else:
                betax_temp = part_data['Px']/(9.10938356e-31 * gamma_temp *c_0)
        label_append.append("Betax")
        data_append.append(betax_temp)
        part_traj[part] = nlr.append_fields(part_data, label_append,
         data_append)

 
    with open('betax{:.2}.pickle'.format(args.beta_x), 'wb') as bp:
        pickle.dump((args.beta_x,part_traj), bp)

if __name__== "__main__":
    __main()
