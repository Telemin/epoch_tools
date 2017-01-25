#!/usr/bin/env

import os
import sys
import shutil
from distutils import spawn 
import sdf
import numpy as np
import psutil
import math


def plasma_freq(dens):
  e = 1.602176565e-19
  e0 = 8.854187817e-12
  me = 9.10938291e-31
  fsq = dens * e * e / (me * e0)
  return(math.sqrt(fsq))

def critical_density(omega):
  e = 1.602176565e-19
  e0 = 8.854187817e-12
  me = 9.10938291e-31
  nc = omega * omega * me * e0 / (e * e)
  return(nc)

def mkdir(dirname):
  try:
    os.mkdir(dirname)
  except:
    pass
  return(dirname)


def force_mkdir(dirname):
  try:
    shutil.rmtree(dirname)
  except:
    pass
  try:
    os.mkdir(dirname)
  except:
    print('Problem creating {0} directory, check perms?'.format(dirname))
    exit()
  return(dirname)


def read_sdf(filename):
  try:
    return(sdf.SDF(filename).read())
  except:
    print("Failed to open SDF file")
    sys.exit(-1)

def ld_to_dl(ld):
  dl = {}
  for d in ld:
    for k,v in d.items():
      try:
        dl[k].append(v)
      except KeyError:
        dl[k]=[v]
  return dl

def check_file_in_use(filename):
  filepath = os.path.abspath(filename)
  if not os.path.exists(filepath):
    return False

  for proc in psutil.process_iter():
    try:
      flist = proc.open_files()
      if flist:
        for fh in flist:
          if fh.path == filepath:
            return(proc.pid)
    except psutil.NoSuchProcess:
      pass
    except psutil.AccessDenied:
      pass

  return False

def make_movie(dirname, append=""):
  print("Authoring Movie")
  if spawn.find_executable('avconv'):
    converter = spawn.find_executable('avconv')
  elif spawn.find_executable('ffmpeg'):
    converter = spawn.find_executable('ffmpeg')
  else:
    print("Couldn't find ffmpeg/avconv :(\n"
          "you're on your own for making the movie")
    os.exit(1)

  conv_args = [converter,'-r', '1', '-i', '{0}/%04d.png'.format(dirname),
              '-c:v', 'libx264', '-r', '30', '-pix_fmt', 'yuv420p', 
              ''.join((os.path.basename(os.getcwd()),'{0}.mp4'.format(append)))]
  spawn.spawn(conv_args)
 
