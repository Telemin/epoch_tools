#!/usr/bin/env python3

import os
import sys

import sdf
import numpy as np
np.seterr(all='warn')

import scipy as sp
import scipy.constants as sc
import scipy.interpolate as si
import scipy.optimize as so
import matplotlib as mpl
import matplotlib.figure as mf
import matplotlib.backends.backend_agg as mplbea
import matplotlib.cm as mcm
import matplotlib.colors as mc

import multiprocessing as mp

from pyutils.funchelpers import TextProgress

import re
import configobj
import functools

import warnings

warnings.simplefilter('error', UserWarning)

def parse_input():
    if len(sys.argv) < 2:
        print("Running single threaded")
        return(1)
    try:
        nthreads = int(sys.argv[1])
    except:
        print("Usage: {} [nthreads]".format(sys.argv[0]))
        sys.exit(-1)

    return(nthreads)


def get_lambda(x, efield):
    fey = np.average(np.abs(np.fft.rfft(efield,axis=0)),axis=-1)
    with np.errstate(divide='ignore'):
        lx = 1/np.fft.rfftfreq(x.size, np.average(np.diff(x)))
        fey = fey[~np.isinf(lx)]
        lx = lx[~np.isinf(lx)]

    maxarg = np.argmax(fey)
    loarg = maxarg - 15 if maxarg - 15 >= 0 else 0;
    hiarg = maxarg + 15 if maxarg + 15 < fey.size else -1;

    fey = fey[loarg:hiarg]
    lx = lx[loarg:hiarg]

    ifey = si.interp1d(lx,-fey, kind='cubic', bounds_error=False, fill_value=0)

    lpk = so.minimize(ifey,lx[np.argmax(fey)]).x

    return(lpk)    


def sdf_getdata(sdf_file):
    try:
        sdf_handle = sdf.read(sdf_file)
    except Exception as err:
        print("Warning: Unreadable sdf file {}".format(sdf_file))
        return((np.nan,np.nan,np.nan))
    try:
        ey = getattr(sdf_handle, "Electric_Field_Ey").data
        x = getattr(sdf_handle, "Grid_Grid").data[0]

        eyargmax = np.unravel_index(np.argmax(ey), ey.shape)[0]
        eymax = np.max(ey)
        xmax = x[eyargmax]

        try:
            las_lambda = get_lambda(x, ey)
        except Exception as e:
            print(type(e))
            print(e)
            las_lambda = np.nan
    except Exception as e:
        print(e)
        print("Warning: Missing field data in {}".format(sdf_file))
        return((np.nan,np.nan,np.nan,np.nan))
    if xmax == 0:
        return((np.nan,np.nan,np.nan,np.nan))
    try: 
        bmax = np.max(getattr(sdf_handle, "Particles_Px_electron").data /
                      getattr(sdf_handle, "Particles_Gamma_electron").data)
        bmax = bmax / (sc.m_e * sc.c)
    except Exception as err:
        bmax = np.nan

    return((xmax,eymax,bmax, las_lambda))

def main():
    workdir = os.getcwd()

    nthreads = parse_input()

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

    if nthreads > 1:
        with mp.Pool(nthreads) as worker_pool:
            results = np.asarray([i for i in TextProgress(worker_pool.imap(sdf_getdata, sdf_files)
                                              ,length=len(sdf_files))])
            worker_pool.close()
            worker_pool.join()
    else:
        results = np.asarray(list(map(sdf_getdata, TextProgress(sdf_files))))

    print(results)
    print(results.shape)

    results = results[np.argsort(results[:,0])]
    results[:,1] = results[:,1] * results[:,3] * sc.e/(sc.m_e * 2 * np.pi * sc.c**2)
    header = "z (m) a_0 beta_e_max las_lambda"
    np.savetxt("a0.txt", results, header=header)

    beta_max = np.max(results[:,2])
    print("beta_max: {}".format(beta_max))

    fig = mf.Figure(figsize=(4,3))
    canvas = mplbea.FigureCanvasAgg(fig)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(results[:,0], results[:,1])
    ax2.plot(results[:,0], results[:,3]*1e9,color='green')
#    ax2.set_ylim(0.8,1.0)

    ax2.text(0.9, 0.2
            ,r"$\beta_{{max}}={}$".format(beta_max)
            ,verticalalignment="bottom"
            ,horizontalalignment="right"
            ,transform=ax2.transAxes)

    fig.savefig('a0.png')
   
    

if __name__ == "__main__":
    main()
