#!/usr/bin/env python3
import sdf
import numpy as np
import numpy.random as nr
import pandas as pd
import logging
import scipy.constants as sc
import re
import os
import sys
import time
import configobj

_particle_regex = re.compile(r"_?Particles_?")
_grid_regex = re.compile(r"_?Grid_?")
_ID_regex = re.compile(r"_?ID_?")
_no_part_per_cell_regex = re.compile(r"^((?!Particles_Per_Cell).)*$")

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


def _create_unmangler(_species_regex):
    def unmangler(data_name,species_regex=_species_regex):
        for r in (species_regex, _particle_regex):
            data_name = r.sub('', data_name,count=1)
        return(data_name)
    return(unmangler)

def load_species_to_dataframe(sdf_file,species='electron',idx=None,
    mask_data=None,mask_field=None):
    """
    Load all particles of species from sdf_file,
    Returns a RecArray of all available particle data
    """
    logging.info("Importing from {}".format(sdf_file))
    try:
        sdf_data = sdf.read(sdf_file)
    except:
        raise

    species_regex = re.compile(r"_?{}_?".format(species))
    unmangle = _create_unmangler(species_regex)

    particle_data = dir(sdf_data)
    particle_data = filter(_particle_regex.search, particle_data)
    particle_data = filter(_no_part_per_cell_regex.search, particle_data)
    particle_data = list(filter(species_regex.search, particle_data))

    if len(particle_data) == 0:
        return None

    data_dict = {}

    if mask_data is not None:
        if mask_field is None:
            logging.warn("mask_data given but no mask_field specified")
        else:
            try:
                mask = np.in1d(getattr(sdf_data, mask_field).data, mask_data)
            except:
                logging.error("Failed to perform masking. aborted")
                raise
    else:
        mask = None

    for data_name in particle_data:
        try:
            data = getattr(sdf_data,data_name).data
        except:
            logging.warn(UserWarning("Failed to read {} from {}" \
                                      "".format(data_name,sdf_data)))
            continue
        
        if type(data) == type(tuple()):
            for data_idx in range(len(data)):
                if type(data[data_idx]) == type(np.ndarray(1)):
                    unmangled_name=''.join((unmangle(data_name),str(data_idx)))
                    if mask is not None:
                        data_dict[unmangled_name] = data[data_idx][mask]
                    else:
                        data_dict[unmangled_name] = data[data_idx]
                else:
                    logging.warn(UserWarning("{} is not a numpy array, skipping"
                                               "".format(data_name)))
                continue
        elif type(data) == type(np.ndarray(1)):
            unmangled_name = unmangle(data_name)
            if mask is not None:
                data_dict[unmangled_name] = data[mask]
            else:
                data_dict[unmangled_name] = data
        else:
            logging.warn(UserWarning("{} is not a numpy array, skipping"
                                       "".format(data_name)))
            continue
  
    if idx is not None: 
        try:
            idxdata = data_dict[idx]
        except:
            logging.error("Index data {} not available".format(idx))
            raise
        else:
            del data_dict[idx]
        species_dframe = pd.DataFrame(data_dict, index= idxdata)
    else:
        species_dframe = pd.DataFrame(data_dict)

    return(sdf_data.Header,species_dframe)


def select_pids_from_sdf(sdf_file, pid_var, compare_var, compare_func,
                         compare_thres):
    sdf_data = sdf.read(sdf_file)
    compare_data = getattr(sdf_data, compare_var).data
    pid_data = getattr(sdf_data, pid_var).data
    new_pid_data = pid_data[compare_func(compare_data, compare_thres)]
    logging.info("Selected {}/{} PIDS from {}".format(new_pid_data.shape[0],len(pid_data),sdf_file))
    return(new_pid_data)


def step_time_printer(step_text, step_times = []):
    currtime = time.time()
    try:
        print("{} in {} s".format(step_text, currtime - step_times[-1][1]))
    except:
        pass
    step_times.append((step_text,currtime))
    return(step_times)


def main():
    config_file = parse_input()
    config = loadconfig(config_file)
    #to be configised
    logging.basicConfig(level=logging.WARN)
    datadir = config['datadir']
    save_name = config['save_name']
    if save_name.endswith('.pkl'):
        save_name = save_name[:-4]
    elif save_name.endswith('.pickle'):
        save_name = save_name[:-7]
    beta_thres = float(config['beta_thres'])
    try:
        subset_size = int(config['subset_size'])
    except:
        subset_size = None
    try:
        subset_save_name = config['subset_save_name']
    except:
        subset_save_name = None
    try:
        sdfmin, sdfmax = [int(x) for x in config['sdf_select']]
    except:
        sdfmin = sdfmax = None
    select_thres = beta_thres * sc.c * sc.m_e / np.sqrt(1-np.power(beta_thres,2))
    select_var = "Particles_Px_electron"
    select_func = np.greater_equal
    sdf_list = [os.path.join(datadir,f) for f in os.listdir(datadir) if f.endswith('.sdf')]
    if sdfmin is not None:
        print("Selecting particles from {:04}.sdf to {:04}.sdf".format(sdfmin,sdfmax))
        sdf_presel_list = [os.path.join(datadir,'{:04}.sdf'.format(f)) for f in range(sdfmin,sdfmax)]
    else:
        sdf_presel_list = sdf_list
   
    step_time_printer('Initialise')

    # now we need to pull out the pids matching our selection criteria
    selected_pid_list = [ select_pids_from_sdf(sdf_file
                                              ,'Particles_ID_electron'
                                              ,'Particles_Px_electron'
                                              ,np.greater
                                              ,select_thres)
                          for sdf_file in sdf_presel_list ]
    selected_pids = np.unique(np.hstack(selected_pid_list)) 

    step_time_printer("Selected {} PIDs".format(len(selected_pids)))

    frame_dict = {header['time']:dframe for header,dframe in 
        [load_species_to_dataframe(datafile, idx='ID', mask_data=selected_pids,
         mask_field='Particles_ID_electron') for datafile in sdf_list] 
         if not dframe.empty}

    step_time_printer("Parsed {} sdf files".format(len(sdf_list)))

    data_panel = pd.Panel(frame_dict)

    step_time_printer("Data panel creation")

    data_panel = data_panel.transpose(1,0,2)

    step_time_printer("Data tranposed to PID first")

    print(data_panel)

    data_panel.to_pickle("{}.pkl".format(save_name))
    
    step_time_printer("Data saved to {}.pkl".format(save_name))

    if subset_size is not None:
        subset_save_name = "{}_subset{}.pkl".format(save_name,subset_size)
        subset = data_panel[nr.choice(data_panel.items, size=subset_size, replace=False)]
        subset.to_pickle(subset_save_name)

    step_time_printer("Subset saved to {}, size {}".format(subset_save_name,subset_size))




if __name__ == "__main__":
    main()
