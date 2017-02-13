#!/usr/bin/env python3

import os
import sys

import sdf
import numpy as np
import scipy as sp
import scipy.constants as sc
import matplotlib as mpl
import matplotlib.figure as mf
import matplotlib.backends.backend_agg as mplbea
import matplotlib.cm as mcm
import matplotlib.colors as mc

import re
import configobj
import functools

import warnings

warnings.simplefilter('error', UserWarning)

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


def sdf_getdata(sdf_file):
    try:
        sdf_handle = sdf.read(sdf_file)
    except Exception as err:
        print(err)
        return((np.nan,np.nan,np.nan))
    try:
        ey = getattr(sdf_handle, "Electric_Field_Ey").data
        eymax = np.max(ey)
        eyargmax = np.unravel_index(np.argmax(ey), ey.shape)[0]
        x = getattr(sdf_handle, "Grid_Grid").data[0][eyargmax]
    except:
        print(err)
        return((np.nan,np.nan,np.nan))
    try: 
        bmax = np.max(getattr(sdf_handle, "Particles_Px_electron").data /
                      getattr(sdf_handle, "Particles_Gamma_electron").data)
        bmax = bmax / (sc.m_e * sc.c)
    except Exception as err:
        print(err)
        bmax = np.nan

    return((x,eymax,bmax))

def main():
    workdir = os.getcwd()

    config_file = parse_input()
    config = loadconfig(config_file)

    try:
        las_lambda = float(config['laser_lambda'])
        las_omega = 2*np.pi*sc.c/las_lambda
    except:
        las_omega = None
        pass

    try:
        if las_omega is None:
            las_omega = float(config['laser_omega'])
    except:
        print("No laser_omega/laser_lambda specified. Fatal. Quitting...")
        return(-1)
    
    print("{:.3e}".format(las_omega))

    try:
        sdf_regex = re.compile(config['sdf_regex'].strip())
    except re.error as err:
        print('Invalid regex specified, full error was:\n{}'.format(err))
        return(-1)
    except:
        sdf_regex = None
        pass

    if sdf_regex is not None:
        sdf_files = [fn for fn in os.listdir(workdir) if sdf_regex.match(fn)]
    else:
        sdf_files = [fn for fn in os.listdir(workdir) if fn.endswith('.sdf')]

    print("Acting on {} sdf files".format(len(sdf_files)))

    results = np.asarray(list(map(sdf_getdata, sdf_files)))

    results[:,1] = results[:,1] * sc.e/(sc.m_e * sc.c * las_omega)

    fig = mf.Figure(figsize=(4,3))
    canvas = mplbea.FigureCanvasAgg(fig)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(results[:,0], results[:,1])
    ax2.plot(results[:,0], results[:,2],color='green')
    ax2.set_ylim(0.8,1.0)

    print("beta_max: {}".format(np.max(results[:,2])))

    fig.savefig('a0.png')
   
    

if __name__ == "__main__":
    main()
