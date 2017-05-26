#!/usr/bin/env python3

import sys
import pickle
import os.path
import configobj
import numpy as np
import sdf
import matplotlib as mpl
import matplotlib.figure as mplf
import matplotlib.backends.backend_agg as mplbea
import matplotlib.gridspec as mplgs
import matplotlib.colors as mplc
import matplotlib.cm as mplcm
from pyutils.calculus import NonUniformCentralDifference as NuCent
from multiprocessing import Pool

m_e = 9.10938215e-31
c_0 = 299792458

def colorcycle(i):
    return(mplcm.hsv((i/10.3)%1))

def loadconfig():
    
    if len(sys.argv) != 2:
        print("Error: Expected exactly 1 argument (path to config)")
        sys.exit(-1)
    try:
        with open(sys.argv[1],'r') as cfile:
            config_data = configobj.ConfigObj(cfile.readlines())
    except OSError as err:
        print("Error: failed to open {}. No valid config provided. Quitting"
              "".format(sys.argv[1]))
        sys.exit(-1)
    except SyntaxError as err:
        print("Warning: Errors occured while parsing input data. "
              "Attempting to continue, see {}.log for details"
              "".format(os.path.basename(sys.argv[0])))
        config_data = err.config
    
    return(config_data)

def load_trajectories():
# load/create trajectory data

    # can use jobid to do data caching...

    trajectory_file = conf['trajectory_data']

    try:
        with open(trajectory_file, 'rb') as pf:
            inj_metadata, inj_trajectory_data = pickle.load(pf)
    except FileNotFoundError:
        print("Error: Failed to load trajectory data from {}".format(
            trajectory_file))
        print(err)
        sys.exit(-1)

    if 'ids' in conf:
        try:
            selected_ids = [ int(pid) for pid in conf['ids']]
        except ValueError as err:
           print("Particle ID parsing failed, valid values are a list of ids"
                  "\n or fraction of total population to use")
           sys.exit(-1)

    else:
        selected_ids = list(inj_trajectory_data.keys())
    
    if 'part_xrange' in conf:
        try:
            selected_ids2 = []
            xmin,xmax = (float(conf['part_xrange'][0]),float(conf['part_xrange'][1]))
            for pid in selected_ids:
                atmin = inj_trajectory_data[pid]['time'].argmin()
                if (inj_trajectory_data[pid]['Grid0'][atmin] > xmin and 
                    inj_trajectory_data[pid]['Grid0'][atmin] < xmax):
                    selected_ids2.append(pid)
            selected_ids = selected_ids2
        except Exception as err:
            print("Failed to select by xrange:\n{}".format(err))
     
    if 'fraction' in conf:
        try:
            frac = float(conf['fraction']) 
            if frac <= 1.0 and frac > 0:
                from random import shuffle as randomshuffle
                randomshuffle(selected_ids)
                select = int(len(selected_ids)*frac)
                selected_ids2 = selected_ids[:select]
                print("Fraction {} is {}/{} particles."
                      "".format(frac,select,len(selected_ids)))
                selected_ids = selected_ids2
                with open('selected_ids.txt','w') as f:
                    [f.write(''.join([str(pid),'\n'])) for pid in selected_ids]
            else:
                raise ValueError("Fraction outside of range")
        except Exception as err:
           print(err)
           print("Error: invalid fraction specified, expect float in range 0-1")
           sys.exit(-1)

              
    selected_particles = {}
    for pid in selected_ids:
        try:
            selected_particles[pid] = inj_trajectory_data[pid]
        except KeyError as err:
            print("Warning: No particle with ID={} found in {}".format(err.args[0],
                trajectory_file))
    
    if len(selected_particles) == 0:
        print("Error: no particle trajectories found, it seems pointless to "
              "continue...")
        sys.exit(-1)

    print("Final selection is {}/{} particles".format(len(selected_particles),
                                                     len(inj_trajectory_data)))

    return(selected_particles)

def reframe_data(keyframes, data, data_axis=np.s_[:], in_place=False):
    try:
        if data[data_axis].shape != keyframes.shape:
            raise ValueError("data[data_axis] and keyframes must have same "
                             "dimensionality")
    except AttributeError:
        raise TypeError("Expected keyframes, data to be of type np.ndarray")
   
    if in_place is True:
        data[data_axis] = data[data_axis] - keyframes
        return
    else:
        newdata = data.copy()
        newdata[data_axis] = newdata[data_axis] - keyframes

    return(newdata)


def get_sdf_list():
    try:
        sdfdir = conf['sdfdir']
        sdf_list = [f for f in os.listdir(sdfdir) if f.endswith('.sdf')]
        if len(sdf_list) < 1:
            raise ValueError("No sdf files found in sdfdir")
    except KeyError:
        print("sdfdir not specified")
        raise
    except FileNotFoundError as err:
        print("sdfdir {} not a valid directory".format(sdfdir))
        raise
    except ValueError:
        print("No sdf files found in {}".format(sdfdir))
        raise
    return(sdf_list)

def track_window(sdf_list):
    pos_pairs = []
    for sdf_file in sdf_list:
        try:
            s = sdf.read(os.path.join(conf['sdfdir'],sdf_file))
            pos_pairs.append((s.Header['time'],s.Grid_Grid.data[0][0]))
        except Exception as err:
            print(err)
            pass
    if len(pos_pairs) < 1:
        raise ValueError("No SDF data loaded")

    return(np.asarray(pos_pairs))


def calculate_window_coords(traj_data, win_data):
    traj_data.sort(order='time')
    win_data = win_data[win_data[:,0].argsort()]
    win_times = win_data[:,0]
    win_xs = win_data[:,1]

    #which starts first?
    if win_times.min() < traj_data['time'].min():
        win_amin = np.argwhere(win_times == traj_data['time'].min()).flatten()[0]
        traj_amin = 0
    else:
        win_amin = 0
        traj_amin = np.argwhere(traj_data['time'] == win_times.min()).flatten()[0]

    #which ends first?
    if win_times.max() > traj_data['time'].max():
        win_amax = np.argwhere(win_times == 
                               traj_data['time'].max()).flatten()[0] + 1
        traj_amax = len(traj_data['time'])
    else:
        win_amax = len(win_times)
        traj_amax = np.argwhere(traj_data['time'] == 
                                win_times.max()).flatten()[0] + 1

    if len(traj_data['time'][traj_amin:traj_amax]) != len(win_times[win_amin:win_amax]):
        raise ValueError("I couldn't match the trajectory data to the window"
                         "data, and everything broke.  I'm sorry...")


    reframed_x = reframe_data(win_xs[win_amin:win_amax], 
                              traj_data['Grid0'][traj_amin:traj_amax])

    return(np.row_stack((reframed_x
                         ,traj_data['Grid1'][traj_amin:traj_amax]
                         ,traj_data['time'][traj_amin:traj_amax])))


def plot_all_the_things(sdf_file, window_trajs, lab_trajs, bubb_traj):
    print("Processing {}".format(sdf_file))
    gs = mplgs.GridSpec(2,2) 
    fig = mplf.Figure((6.4,3.6),dpi=300)
    canvas = mplbea.FigureCanvasAgg(fig)
    ax = []
    for i in range(2):
        for j in range(2):
            ax.append(fig.add_subplot(gs[i,j]))

    try:
        sdfdata = sdf.read(os.path.join(conf['sdfdir'],sdf_file))
        ext = [sdfdata.Grid_Grid.data[0][0]     #xmin
              ,sdfdata.Grid_Grid.data[0][-1]    #xmax
              ,sdfdata.Grid_Grid.data[1][0]     #ymin
              ,sdfdata.Grid_Grid.data[1][-1]    #ymax
              ]
        dens = sdfdata.Derived_Number_Density_electron.data
    except Exception as err:
        print(err)
        print("Problem analyzing {0}, file corrupted or data missing"
              "".format(sdf_file))
        return(None)

    ax[0].imshow(dens.transpose(), aspect='auto'
                ,extent=ext
                ,norm=mplc.LogNorm()
                ,vmin=1e24, vmax=1e27
                ,cmap=mplcm.viridis
                ,alpha=0.8
                )
    for i,pid in enumerate(window_trajs):
        cur_traj = window_trajs[pid]
        traj_x = cur_traj[0] + ext[0]
        traj_y = cur_traj[1]
        ax[0].plot(traj_x, traj_y, color=colorcycle(i), lw=0.5, alpha=0.8)
        
        try:
            cur_arg = np.argwhere(cur_traj[2] == sdfdata.Header['time']
                                 ).flatten()[0]
        except:
            pass
        else:
            cur_x = cur_traj[0][cur_arg] + ext[0] 
            cur_y = cur_traj[1][cur_arg] 
            ax[0].scatter(cur_x, cur_y
                         ,edgecolors='none'
                         , color=colorcycle(i)
                         ,s=4
                         ,zorder=4
                         )

    ax[0].set_xlim(ext[:2])
    ax[0].set_ylim(ext[2:])

    if bubb_traj is not None:
        ax[2].plot(bubb_traj[0],bubb_traj[2]/c_0, lw=0.5, alpha=1.0)

    for i,pid in enumerate(lab_trajs): 
        cur_traj = lab_trajs[pid]
        ax[1].plot(cur_traj['time']
                  ,cur_traj['Px']/(m_e * c_0)
                  ,color=colorcycle(i),lw=0.5, alpha=1.0)
        ax[2].plot(cur_traj['time']
                  ,cur_traj['Py']/(m_e * c_0)
                  ,color=colorcycle(i),lw=0.5, alpha=1.0)
        ax[3].plot(cur_traj['time']
                  ,cur_traj['Py']/cur_traj['Px']
                  ,color=colorcycle(i),lw=0.5, alpha=1.0)

        try:
            cur_arg = np.argwhere(cur_traj['time'] == sdfdata.Header['time']
                                 ).flatten()[0]
        except:
            pass
        else:
            ax[1].scatter(cur_traj['time'][cur_arg]
                         ,cur_traj['Px'][cur_arg]/(m_e * c_0)
                         ,edgecolors='none'
                         ,color=colorcycle(i)
                         ,s=4
                         ,zorder=4
                         ,alpha=0.5
                         )
            ax[2].scatter(cur_traj['time'][cur_arg]
                         ,cur_traj['Py'][cur_arg]/(m_e * c_0)
                         ,edgecolors='none'
                         ,color=colorcycle(i)
                         ,s=4
                         ,zorder=4
                         ,alpha=0.5
                         )
            ax[3].scatter(cur_traj['time'][cur_arg]
                         ,cur_traj['Py'][cur_arg]/cur_traj['Px'][cur_arg]
                         ,edgecolors='none'
                         ,color=colorcycle(i)
                         ,s=4
                         ,zorder=4
                         ,alpha=0.5
                         )

    for i,plot in enumerate(conf['PLOTS'].values()):
        try: ax[i].set_xlim(left = float(plot['xmin']))
        except: pass
            
        try: ax[i].set_xlim(right = float(plot['xmax']))
        except: pass
            
        try: ax[i].set_ylim(bottom=float(plot['ymin']))
        except: pass

        try: ax[i].set_ylim(top = float(plot['ymax']))
        except: pass

        if 'window_clip' in plot:
            try:
                if len(plot['window_clip']) < 4:
                    raise ValueError("Expected list of length 4")
                clip_ratios = [float(i) for i in plot['window_clip']]
            except:
                print("Error: window_clip expects a list of ratios (0.0-1.0):"
                      "\nwindow_clip = xmin, xmax, ymin, ymax")

            else:
                x_ext = ax[i].get_xlim()
                x_len = x_ext[1] - x_ext[0]
                y_ext = ax[i].get_ylim()
                y_len = y_ext[1] - y_ext[0]

                nxmin = x_ext[0] + clip_ratios[0]*x_len
                nxmax = x_ext[0] + clip_ratios[1]*x_len
                nymin = y_ext[0] + clip_ratios[2]*y_len
                nymax = y_ext[0] + clip_ratios[3]*y_len
            
                ax[i].set_xlim(nxmin, nxmax) 
                ax[i].set_ylim(nymin, nymax) 
           

        if 'title' in plot:
            ax[i].set_title(plot['title'])

        if 'xlabel' in plot:
            ax[i].set_xlabel(plot['xlabel'])

        if 'ylabel' in plot:
            ax[i].set_ylabel(plot['ylabel'])
        
    
    for axis in ax:
        axis.ticklabel_format(style='sci',scilimits=(2,2))


    fig.savefig(sdf_file[:-4]+'.png', dpi=300)
#    fig.savefig('eps/'+sdf_file[:-4]+'.eps',papersize='a3',orientation='landscape')


def load_bubble_data():
    if 'bubble_data' in conf:
        try:
            with open(conf['bubble_data'],'rb') as pf:
                bubble_data = pickle.load(pf)
            bubble_data = bubble_data[0.0]
            bt = bubble_data[:,1]
            bx = bubble_data[:,0]
            bv = NuCent(bx,bt)
            print("Loaded bubble trajectory data")
            return((bt,bx,bv))
        except Exception as err:
            print(err)
            return(None)
    return(None)

def main():
    global conf
    conf = loadconfig()


    try:
        sdf_list = get_sdf_list()
    except:
        print("No sdf files available, this is fatal.")
        sys.exit(-1)

    try:
        sdf_s = int(conf['sdf_start'])
        sdf_e = int(conf['sdf_end'])
        sdf_range = {"{:04}.sdf".format(i) for i in range(sdf_s,sdf_e + 1)}
        sdf_list = list(sdf_range.intersection(sdf_list))
        sdf_list.sort()
    except:
        pass
    finally:
        print("Acting on {} sdf files".format(len(sdf_list)))


    lab_trajs = load_trajectories()
    window_trajectory = track_window(sdf_list)

    window_trajs = {pid:calculate_window_coords(lab_trajs[pid],
                    window_trajectory) for pid in lab_trajs}

    #plot all the things!!! http://memegenerator.net/instance2/3883211


    bubb_traj_data = load_bubble_data()

#    wp = Pool(processes=24)
#    res = [wp.apply_async(plot_all_the_things
#                   ,args=(sdf_file, window_trajs, lab_trajs, bubb_traj_data)
#                   ) for sdf_file in sdf_list]
#    [p.get() for p in res]

    for sdf_file in sdf_list:
        plot_all_the_things(sdf_file, window_trajs, lab_trajs, bubb_traj_data)

if __name__ == "__main__":
    main()
