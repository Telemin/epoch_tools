#!/usr/bin/env python3
# vim: set fileencoding=UTF-8

import re
import multiprocessing as mp
import os
import sys
import shutil
from distutils import spawn
import argparse
import functools

import epochtools_common as ec

import matplotlib
matplotlib.use('Agg')  
from matplotlib.colors import LogNorm, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
import sdf
import pickle

def argument_parse():

# Set up parser and global options
    parser = argparse.ArgumentParser(description='Find density FFT')
    parser.add_argument('--exclude','-e',dest='exclude',action='append',
        metavar="patt", help="exclude filenames matching regex 'patt', " 
        "may be used more than once")
    parser.add_argument('--freshen','-f', dest='freshen', action='store_true',
        help="Re-analyze all files")
    parser.add_argument('--ires', '-i', dest='ires', type=int, nargs=2,
        metavar=(r'x(px)', r'y(px)'), help='Output image dimensions (px)')
    parser.add_argument('--numprocs','-n', dest='procs', type=int, default=1,
        help="Number of analysis threads to use")
    parser.add_argument('--window-clip','-c',dest='clip', type=float, nargs=4,
        metavar=("xmin","xmax","ymin","ymax"), help="Clip total window size, (expressed as"
        "fractional position, from bottom right)")
    parser.add_argument('xvarname', type=str)
    parser.add_argument('yvarname', type=str)
    parser.add_argument('selvarname', type=str)
    parser.add_argument('pidlistfile',type=str)
    parser.add_argument('--outputprefix','-p',dest='outputPrefix',type=str,
        metavar="prefix", help="prefix to apply to output file(s)")
    parser.add_argument('--outputdir','-d',dest='outputDir',type=str,
        metavar="dir", help="directory to hold output files, defaults to {mode}")

    return(parser.parse_args())


def debug(*debuginfo):
    if args.verbose:
        print(''.join(map(str,debuginfo)))

def clamp(val, minval, maxval):
    return(max(min(val,maxval),minval))

def midpoints(x):
    return((x[:-1] + x[1:])/2)

def phase_space_image_highlight(filename, xvarname, yvarname, iddataname,
    pidlist, outputPrefix='', outputDir=''):

    FileInUse = ec.check_file_in_use(filename)
    savepath = os.path.join(outputDir,(''.join((outputPrefix,filename[:-4],'.png'))))
    if FileInUse:
        print("{0} in use by PID {1}, skipping".format(filename,FileInUse))
        return
    elif os.path.exists(os.path.join(os.getcwd(),outputDir,''.join((filename[:-4],'.png')))):
        print("{0} already analyzed, skipping".format(filename))
        return
    else:
        print("Analyzing {0}".format(filename))

    varnames = (xvarname, yvarname)
    psvars = []
    try:
        filedata = sdf.read(filename)
        iddata = getattr(filedata, iddataname).data
        for varname in varnames:
            if varname.endswith('_x') or varname.endswith('_y'):
                if varname.endswith('_x'):
                    psvars.append(getattr(filedata, varname[:-2]).data[0])
                else:
                    psvars.append(getattr(filedata, varname[:-2]).data[1])
            else:
                psvars.append(getattr(filedata, varname).data)
    except Exception as err:
        print(err)
        print("Problem analyzing {0}, file corrupted, missing data"
            "or incorrect name specified".format(filename))
        return

    iddata_idx = np.argsort(iddata)
    iddata_sort_idx = np.searchsorted(iddata[iddata_idx], pidlist)
    
    pid_m_index = np.take(iddata_idx, iddata_sort_idx, mode='clip')
    mask = iddata[pid_m_index] != pidlist

    pid_index = np.ma.array(pid_m_index, mask=mask).compressed()

    selected_vars = None
    if pid_index.shape[0] != 0:
        selected_vars = [psvars[0][pid_index],psvars[1][pid_index]]


    plt.figure()
    ax = []
    im = []
    ax.append(plt.subplot2grid((1,1), (0,0)))
    im.append(ax[0].scatter(psvars[0][::50], psvars[1][::50], s=0.1, color='black'))
    if selected_vars is not None:
        im.append(ax[0].scatter(selected_vars[0], selected_vars[1], s=0.1,
            color='red'))
    ax[0].set_xlim(psvars[0].min(),psvars[0].max())
    #if args.ylim:
    #    ax[0].set_ylim(args.ylim[0],args.ylim[1])
    #else:
    ax[0].set_ylim(psvars[1].min(),psvars[1].max())
    x_px = args.ires[0] if args.ires else 1920
    y_px = args.ires[1] if args.ires else 1080
    y_size=10
    x_size=y_size*x_px / y_px
    ax[0].set_title("$time\ t={:.3e}\mathrm{{s}}$".format(filedata.Header['time']))
    ax[0].set_xlabel("$x (\lambda_p)$")
    ax[0].set_ylabel("$P_x (\mathrm{{MeV}})$")
    plt.gcf().set_size_inches(x_size,y_size)
    plt.savefig(savepath)
    plt.close()
    return(savepath)
    
def main():

    global args
    args = argument_parse()

    prefix = ''
    if args.outputPrefix:
        prefix = args.outputPrefix

    output = "phase_space_highlight"
    if args.outputDir:
        output = args.outputDir

    if args.freshen == True:
        print("Re-analyzing all files...")
        ec.force_mkdir(output) 
    else:
        print("Analyzing new files...")
        ec.mkdir(output)
 
    sdfList = [f for f in os.listdir(os.getcwd()) if f.endswith('.sdf')]

    with open(args.pidlistfile, 'rb') as plf:
        pidlist = pickle.load(plf)

    assert type(pidlist) is type([]) 

    if args.exclude:
        for pattern in args.exclude:
            try:
                eP = re.compile(pattern)
            except:
                print("Skipping invalid regex pattern {0}".format(pattern))
                continue
            for sdfFile in sdfList:
                if eP.search(sdfFile):
                    sdfList.remove(sdfFile)
    
    processor = functools.partial(phase_space_image_highlight, 
        xvarname=args.xvarname, yvarname=args.yvarname,
        iddataname=args.selvarname, pidlist=pidlist, outputPrefix=prefix,
        outputDir=output)
    
    worker_pool = mp.Pool(processes=8)
    frames = worker_pool.map(processor, sdfList)
    #frames = list(map(processor, sdfList))
    newframes = filter(None, frames)
    if newframes:
        if spawn.find_executable('avconv'):
            converter = spawn.find_executable('avconv')
        elif spawn.find_executable('ffmpeg'):
            converter = spawn.find_executable('ffmpeg')
        else:
            print("Couldn't find ffmpeg/avconv :(\n"
                        "you're on your own for making the movie")
            sys.exit(1)

        conv_args = [converter,'-r', '2', '-i',
            '{0}/{1}%04d.png'.format(output,prefix), '-c:v', 'libx264', '-r', '25',
            '-pix_fmt', 'yuv420p','-y', 
            ''.join((os.path.basename(os.getcwd()),'_{0}.mp4'.format(output)))]
        spawn.spawn(conv_args)
    else:
        print("No new files, skipping movie creation")



if __name__ == '__main__':
    main()
