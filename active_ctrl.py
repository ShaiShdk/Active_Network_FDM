"""
Created in March 2021
@author: Shahriar Shadkhoo -- Caltech
"""

from active_net_dynamics import *
import numpy as np

rhof , rhoi = 1 , 0.5
s0 , gamma  = 0.2 , 0.0

T_tot , dt  = 100 , 0.1
N_frame     = np.min((10,np.int(T_tot/dt)))
tau_s       = 1
mass        = 10

########################################## NETWORK PARAMETERS ##########################################

lattice_shape   = [50 , 50]
UnitCell_Geo    = 'Square'              #{'Square' , 'Hexagon' , 'Triangular' , 'Random'}
Global_Geometry = 'Rectangle'           #{'Rectangle' , "Triangle", "Hexagon", "Hexagram", "Circle", "Ellipse"}

lattice_dsrdr   = 0
if UnitCell_Geo == 'Random':
    lattice_dsrdr = 0.2

net = random_network(UnitCell_Geo,lattice_shape,Global_Geometry)

net.rounded             = 'False'
net.round_coeff         = 2
net.AR_xy               = [1 , 0.5]
net.AR_fr               = 1
net.lattice_dsrdr       = lattice_dsrdr
net.illum_ratio         = 1
net.rhoPower            = 1

net.plot_full_positions = 1
net.plot_velocity_plot  = 0
net.plot_velocity_map   = 0
net.plot_density_map    = 0

activate_full           = 0

region_shape            = net.region_shape()
network_full            = net.net_gen()
active_vertices         = net.active_verts(activate_full)
active_edges            = net.active_edges(net.illumreg,activate_full)

# net.Viscous_Dynamics(T_tot,dt,gamma,rhoi,s0,tau_s)
net.Inertial_Dynamics(T_tot,dt,mass,gamma,rhoi,s0,tau_s)

