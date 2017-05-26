#!/usr/bin/env python
# vim: set fileencoding=UTF-8

import string
import sdf
import math
import numpy as np
import matplotlib as mpl
mpl.use('PS')
import matplotlib.pyplot as plt
import matplotlib.colors as col
from matplotlib.ticker import MaxNLocator, NullLocator, MultipleLocator

def main():
  basedir = "/home/telemin/sim/multiple_injection/"
  dira = basedir+"paper/inject_peaks/300uspaced_100x0.2a_2.0_ne0.005/results/"
  dirb = basedir+"flat_plasma_control/03_a2.0_ne0.005/results/"
  rows = 5
  cols = 2
  s_frame = 290
  step = 65
  im = []
  fontsize = 10
  cmin, cmax = (5e23,5e26)
  framelist = [ "{0:04}".format(i) for i in
      range(s_frame,s_frame+(step*rows),step) ]
  alphalist =  np.fromstring(string.ascii_lowercase,dtype='S1')[:rows*cols].reshape(cols,rows).T
  fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(7,9.9))
  for i,ax in enumerate(axes.flat):
    if i%2 == 0:
        data = sdf.read(''.join([dira,framelist[int(i/2)],'.sdf']))
    else:
        data = sdf.read(''.join([dirb,framelist[int(i/2)],'.sdf']))
    time = data.Header['time'] * 1e12
    xmin,xmax = (500,1500)
    ymin,ymax = (50,300)
    x = data.Grid_Grid.data[0][xmin:xmax]
    y = data.Grid_Grid.data[1][ymin:ymax]
    x = x * 1e6
    y = y * 1e6
    dens = data.Derived_Number_Density_electron.data.transpose()[ymin:ymax,xmin:xmax]
    print(dens.shape)
    im.append(ax.imshow(dens, vmin=cmin, vmax=cmax, norm=col.LogNorm(),
              cmap=plt.get_cmap('CMRmap'),
              extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto'))
    ax.text(0.05,0.95
            ,r'{0}) ${1:.2f}\ \mathrm{{ps}}$'.format(alphalist.ravel()[i].decode(), time)
            ,verticalalignment='top'
            ,horizontalalignment='left'
            ,transform=ax.transAxes
            ,color='white'
            ,fontsize=fontsize)
#    ax.set_xlabel('x',fontsize=6)
#    ax.set_xlabel('x ($\mu \mathrm{m}$)',fontsize=6,position=(0.2,0))
    ax.xaxis.labelpad = 0
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(NullLocator())
#    ax.set_ylabel('y',fontsize=6)
    ax.yaxis.labelpad = 0
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

  cbar = fig.colorbar(im[-1], ax=axes.ravel().tolist()
                     ,orientation='horizontal'
                     ,pad=0.08)
  cbar.ax.tick_params(labelsize=fontsize)
  
  fig.text(0.49,0.23,'x ($\mu \mathrm{m}$)',fontsize=fontsize)
  fig.text(0.07,0.59,'y ($\mu \mathrm{m}$)',fontsize=fontsize,rotation='vertical')

  plt.savefig('snapshots.png',dpi=300, bbox_inches='tight')
  plt.close()


if __name__ == "__main__":
  main()
