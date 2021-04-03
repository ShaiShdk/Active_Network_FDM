#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in March 2021
@author: Shahriar Shadkhoo -- Caltech
"""
"""
"""

# nx_map  = int(np.sqrt(net.Nv))
# Xmap = np.reshape(self.Xt, (nx_map,nx_map)).T
# Ymap = np.reshape(self.Yt, (nx_map,nx_map)).T
# net.ed_ver_sps.shape
# net.verts
# fsize = (5,5)
# fig1, ax1 = plt.subplots(figsize=fsize, dpi= 100, facecolor='w', edgecolor='k')
# ax1.plot(net.Xt, net.Vx, '.' , color='w' , markersize = 0.2 * fsize[0])
# ax1.plot(net.Xt[net.ver_inside] , net.Vx[net.ver_inside], '.' , color='r' , markersize = 0.2 * fsize[0])
# ax1.set_facecolor('k')
# plt.show()

# net.D2_bulk
# net.ver_active
# active_vertices
# net.ver_bulk_ind

# net.Vx.shape
# net.Vel = np.stack((net.Vx.reshape((1988,)),net.Vy.reshape((1988,))),axis=-1)
# net.Vel.shape
#
# net.Vel = np.zeros((1988,2))
# net.Vel[:,0] = net.Vx.reshape((1988,))
# net.Vel[:,1] = net.Vy.reshape((1988,))

# import IPython
# IPython.Application.instance().kernel.do_shutdown(True)

%reset -s -f

from random_network.random_network import *
# from active_net_control.active_net_dynamics import *

import numpy as np , scipy as sp , random
from scipy.spatial import Voronoi , voronoi_plot_2d
from scipy import sparse
import matplotlib.pyplot as plt
from copy import deepcopy
from math import atan2
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from time import time

ti = time()

rhof , rhoi = 1 , 0.5
s0 , gamma  = 0.2 , 0.0

T_tot , dt  = 100 , 0.1
N_frame     = np.min((10,np.int(T_tot/dt)))
tau_s       = 1
mass        = 1

noisy, dvx  = 0 , 0

########################################## NET PARAMETERS ##########################################

lattice_shape   = [31 , 31]
UnitCell_Geo    = 'Square'              #{'Square' , 'Hexagon' , 'Triangular' , 'Random'}
Global_Geometry = 'Rectangle'

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

tf = time()
print(tf-ti)
