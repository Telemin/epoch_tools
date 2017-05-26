#!/usr/bin/env python3

import sys
import os
import h5py
import numpy as np
import scipy.interpolate as si

def main():

    if len(sys.argv) != 3:
        print("Expected exactly two command line arguments:\n"
              "\t{} inputfile rate".format(sys.argv[0]))
        return(-1)

    try:
        rate = int(sys.argv[2])
    except:
        print("Interpolation rate must be an integer!")
        return(-1)

    try:
        traj_in = h5py.File(sys.argv[1],'r')
    except:
        print("\"{}\" does not exist or is not a valid hdf5 file."
              "".format(sys.argv[1]))
        return(-1)

    out_fname = "{}_interp.h5".format(sys.argv[1][:-3])
    traj_out = h5py.File(out_fname, 'w')

    print("Interpolating {} particle(s)".format(len(traj_in.keys())))
    for part in traj_in.values():
        oldct = np.asarray(part['t'])
        newct = np.linspace(oldct.min(),oldct.max(),rate*(oldct.size -1))
        print(oldct.shape, newct.shape)

        newx1 = si.UnivariateSpline(oldct, part['x1'], k=3, s=0)(newct)
        newx2 = si.UnivariateSpline(oldct, part['x2'], k=3, s=0)(newct)
        newx3 = si.UnivariateSpline(oldct, part['x3'], k=3, s=0)(newct)
        newp1 = si.UnivariateSpline(oldct, part['p1'], k=3, s=0)(newct)
        newp2 = si.UnivariateSpline(oldct, part['p2'], k=3, s=0)(newct)
        newp3 = si.UnivariateSpline(oldct, part['p3'], k=3, s=0)(newct)
        newene = np.sqrt(1 + newp1**2 + newp2**2 + newp3**2) - 1
        newq = np.ones(newct.size)*part['q'][0]

        newpart = traj_out.create_group(part.name)
        newpart.create_dataset('t', data=newct)
        newpart.create_dataset('x1', data=newx1)
        newpart.create_dataset('x2', data=newx2)
        newpart.create_dataset('x3', data=newx3)
        newpart.create_dataset('p1', data=newp1)
        newpart.create_dataset('p2', data=newp2)
        newpart.create_dataset('p3', data=newp3)
        newpart.create_dataset('ene', data=newene)
        newpart.create_dataset('q', data=newq)

    traj_out.close()


if __name__ == "__main__":
    main()
