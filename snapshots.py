#!/usr/bin/env python2

import sdf
import math
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import matplotlib.colors as col
from matplotlib.ticker import MaxNLocator, NullLocator

def main():
  fig = plt.figure(figsize=(8.3,11))
  ax = []
  cmin, cmax = (1e25,1e27)
  framelist = [ "{0:04}".format(i) for i in range(120,310,10) ]
  hackyalphalist = ['a','b','c','d','e','f','g','h','i',
                    'j','k','l','m','n','o','p','q','r']
  for i in range(0,18):
    data = sdf.SDF(''.join([framelist[i],'.sdf'])).read()
    time = data['Header']['time'] * 1e15
    xmin,xmax = (1600,2400)
    ymin,ymax = (150,250)
    x = data['Grid/Grid/X'][xmin:xmax]
    y = data['Grid/Grid/Y'][ymin:ymax]
    x = x * 1e6
    y = y * 1e6
    dens = data['Derived/Number_Density/electron'].transpose()[ymin:ymax,xmin:xmax]

    ax_ = plt.subplot2grid((9,2),(int(i/2),int(i%2)))
    ax_.imshow(dens, vmin=cmin, vmax=cmax, norm=col.LogNorm(),
              extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto')
    ax_.set_title(r'{0}) ${1:.0f}\mathrm{{ fs}}$'.format(hackyalphalist[i], time)
                  ,position=(0.05,0.95),fontsize=6)
    ax_.set_xlabel('x',fontsize=6)
#    ax_.set_xlabel('x ($\mu \mathrm{m}$)',fontsize=6,position=(0.2,0))
    ax_.xaxis.labelpad = 0
    ax_.xaxis.set_major_locator(NullLocator())
    ax_.set_ylabel('y',fontsize=6)
#    ax_.set_ylabel('y ($\mu \mathrm{m}$)',fontsize=6)
    ax_.yaxis.labelpad = 0
    ax_.yaxis.set_major_locator(NullLocator())
    ax_.tick_params(axis='both', which='major', labelsize=6)
    ax.append(ax)

  plt.savefig('snapshots.eps',dpi=1200)
  plt.close()


if __name__ == "__main__":
  main()
