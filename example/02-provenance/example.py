"""
Created in March 2021
@author: Shahriar Shadkhoo -- Caltech
"""

from ANFDM.active_net_dynamics import random_network
from ANFDM import __version__ as ANFDM_VERSION
import numpy as np
import sacred

ex = sacred.Experiment('ANFDM')
ex.observers.append(sacred.observers.FileStorageObserver('data/'))
ex.dependencies.add(sacred.dependencies.PackageDependency("ANFDM", ANFDM_VERSION))

@ex.config
def cfg():
    # Geometry parameters
    UnitCell_Geo        = 'Square'#{'Square''Hexagon''Triangular''Random'}
    lattice_shape       = [50,50]
    Global_Geometry     = 'Hexagon'#{'Rectangle'"Triangle""Hexagon""Hexagram""Circle""Ellipse"}
    lattice_dsrdr       = 0
    rounded             = 'False'
    round_coeff         = 2
    AR_xy               = [1, 0.5]
    AR_fr               = 1.
    illum_ratio         = 1
    rhoPower            = 1
    plot_full_positions = True
    plot_velocity_plot  = False
    plot_velocity_map   = True
    plot_density_map    = True
    show_plots          = False

    # Integration parameters
    rhof , rhoi = 1 , 0.5
    s0 , gamma  = 0.2 , 0.0

    T_tot , dt  = 1000 , 0.1
    N_frame     = 25
    tau_s       = 1
    mass        = 10

    dump_type   = 'json' # Can be any of 'json', 'h5', or None


@ex.automain
def main(_config, _run):
    # Initialize network
    net = random_network(_config, experiment = ex)

    # Integrate it up
    integration_method = _config.get('integration_method', 'inertial').lower()
    if integration_method == 'inertial':
        net.Inertial_Dynamics(_config['T_tot'],
                          _config['dt'],
                          _config['mass'],
                          _config['gamma'],
                          _config['rhoi'],
                          _config['s0'],
                          _config['tau_s'],
                          N_frame = _config['N_frame']
                          )
    elif integration_method == 'viscous': 
        net.Viscous_Dynamics(_config['T_tot'],
                          _config['dt'],
                          #_config['mass'],
                          _config['gamma'],
                          _config['rhoi'],
                          _config['s0'],
                          _config['tau_s'],
                          N_frame = _config['N_frame']
                          )
    else:
        raise RuntimeError(f"Don't understand integration type {integration_method}")

