#!/usr/bin/env python3

import numpy as np
import scipy.interpolate as si
import pickle
import sys
from pyutils.calculus import NonUniformCentralDifference as NuCent

def main():
    with open(sys.argv[1], 'rb') as pp:
        bubble_dict = pickle.load(pp)
        bx = bubble_dict[0.][:,0]
        bt = bubble_dict[0.][:,1]
        bv_un, bt_un = NuCent(bx,bt)
        bv_un = si.interp1d(bt_un,bv_un,bounds_error=False,fill_value='extrapolate')(bt)

        np.savetxt(sys.argv[2],np.asarray([bx,bt,bv_un]).T)

if __name__ == "__main__":
    main()
