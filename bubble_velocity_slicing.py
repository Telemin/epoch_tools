#!/usr/bin/env python3

import argparse
import os

import epochpy.particles as ep
import epochpy.tracking as et
import epochpy.bubble as eb
import epochpy.fields as ef
import epochpy.header as eh

import numpy as np
import pickle

c_0 = 299792458

def __argument_parse():

# Set up parser and global options
  parser = argparse.ArgumentParser(description='Slice across bubble to find'
    'velocity at different radii')
  parser.add_argument('-n',type=int,default=5,dest="slice_n",
    metavar="N", help="Take N slices across radius (2N-1 total)")
  parser.add_argument('-o',dest='output',type=str,default="bubble_slices.pickle")
  parser.add_argument('radius',type=float,default=10e-6,
    help="Bubble radius over which to slice")
  method_parser = parser.add_mutually_exclusive_group(required=False)
  method_parser.set_defaults(method='efield')
  method_parser.add_argument('--density'
                            ,dest='method'
                            ,action='store_const'
                            ,const='density')
  method_parser.add_argument('--efield'
                            ,dest='method'
                            ,action='store_const'
                            ,const='efield')
  return(parser.parse_args())


def __main():
    args = __argument_parse()

    slice_points = np.linspace(0, args.radius, args.slice_n) 
    slice_points = np.concatenate((-1*slice_points[1:],slice_points))

    print("Sampling at radii: {}".format(slice_points))

    sdf_list = [f for f in os.listdir(os.getcwd()) if f.endswith('.sdf')]
    zeropoints = {sp :eb.find_zero_points(sdf_list,y_pos=sp,method=args.method) for sp in
        slice_points}
    
    with open(args.output, 'wb') as bp:
        pickle.dump(zeropoints, bp, protocol=2)

if __name__== "__main__":
    __main()
