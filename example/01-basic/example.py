"""
Created in March 2021
@author: Shahriar Shadkhoo -- Caltech
"""

from ANFDM.active_net_dynamics import random_network
import numpy as np

########################################## NETWORK PARAMETERS ##########################################
params = {
            'UnitCell_Geo'        : 'Square',#{'Square','Hexagon','Triangular','Random'}
            'lattice_shape'       : [50,50],
            'Global_Geometry'     : 'Rectangle',#{'Rectangle',"Triangle","Hexagon","Hexagram","Circle","Ellipse"}
            'lattice_dsrdr'       : 0,
            'rounded'             : 'False',
            'round_coeff'         : 2,
            'AR_xy'               : [1,0.5],
            'AR_fr'               : 1,
            'illum_ratio'         : 1,
            'rhoPower'            : 1,
            'plot_full_positions' : True,
            'plot_velocity_plot'  : False,
            'plot_velocity_map'   : True,
            'plot_density_map'    : True,
            'show_plots'          : False,
        }
########################################## NETWORK PARAMETERS ##########################################

# Initialize network
net = random_network(params)

# Integration parameters
rhof , rhoi = 1 , 0.5
s0 , gamma  = 0.2 , 0.0

T_tot , dt  = 100 , 0.1
N_frame     = np.min((10,np.int(T_tot/dt)))
tau_s       = 1
mass        = 10

# Integrate it up
net.Inertial_Dynamics(T_tot,dt,mass,gamma,rhoi,s0,tau_s)

