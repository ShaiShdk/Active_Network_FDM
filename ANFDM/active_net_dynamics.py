
"""
Created in March 2021
@author: Shahriar Shadkhoo -- Caltech

Dynamics of the Active Networks

"""

import numpy as np , scipy as sp , random
from scipy.spatial import Voronoi , voronoi_plot_2d
from scipy import sparse
import matplotlib.pyplot as plt
from copy import deepcopy
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from numba import jit
import uuid
import os
from tqdm import tqdm
import logging
import json
import h5py

logging.basicConfig(level=logging.INFO)

class random_network:
    def __init__(self,params, experiment = None):

        """
            Initializes an random network with the corresponding parameters

        """

        self.sacred_experiment = experiment

        self.UnitCell_Geo       = params.get('UnitCell_Geo'   )                                           # "geometry of the unit cells of the lattice"
        self.unit_cell          = params.get('unit_cell', [1.0 , 1.0])                                    # "size of the unit cells"
        self.lattice_shape      = params.get('lattice_shape'  )                                           # "size of the lattice"
        self.Global_Geometry    = params.get('Global_Geometry')                                           # "global shape"
        self.rounded            = params.get('rounded')                                                   # "determines if the corners of the shape are rounded"
        self.round_coeff        = params.get('round_coeff')                                               # "rounding coefficient in if rounded==True"
        self.AR_xy              = params.get('AR_xy')                                                     # "aspect ratio y/x of the global shape"
        self.AR_fr              = params.get('AR_fr')                                                     # "size of the activated region over the entire network"
        self.act_xyt            = params.get('act_xyt')                                                   # "activity as a function of x,y and t"
        self.n_ver_rmv          = params.get('n_ver_rmv', 0)                                              # "number of vertices removed randomly from the network"
        self.n_edg_rmv          = params.get('n_edg_rmv', 0)                                              # "number of edges removed randomly from the network"
        self.lattice_dsrdr      = params.get('lattice_dsrdr', 0 if self.UnitCell_Geo == 'Random' else .2) # "disorder of the lattice if the unit cell is chosen to be random"
        self.illum_ratio        = params.get('illum_ratio')                                               # "ratio of the different illumination magnitudes"

        self.rhoPower           = params.get('rhoPower')                                                  # "power of density in the stress as a function of density"

        self.plot_full_positions= params.get('plot_full_positions')                                       # "type of plots"
        self.plot_velocity_plot = params.get('plot_velocity_plot')
        self.plot_velocity_map  = params.get('plot_velocity_map')
        self.plot_density_map   = params.get('plot_density_map')
        self.activate_full      = params.get('activate_full')

        self.working_dir        = params.get('working_dir_base', '/tmp/')+'ANFDM-'+str(uuid.uuid4())
        os.makedirs(self.working_dir)
        logging.info(f"Working in {self.working_dir}")
        self.show_plots         = params.get('show_plots', True)
        self.do_last_frame_plot = params.get('do_last_frame_plot', True)
        self.dump_type          = params.get('dump_type', 'json')

        # coordination number of vertices in lattice
        if self.UnitCell_Geo=='Square':
            self.bulkV_deg = 4
        elif self.UnitCell_Geo=='Triangular':
            self.bulkV_deg = 6
        else:
            self.bulkV_deg = 3
        self.region_shape()
        self.net_gen()
        self.active_verts(self.activate_full)
        self.active_edges(self.activate_full)

        self.tt = 0

    def points_initial(self,unit_cell):

        """
            Points to be converted into vertices using voronoi tessellation

        """

        l0_x = unit_cell[0]
        l0_y = unit_cell[1]
        lattice_dsrdr = self.lattice_dsrdr
        lattice_shape = self.lattice_shape
        UnitCell_Geo  = self.UnitCell_Geo
        nx            = lattice_shape[0]
        ny            = lattice_shape[1]

        if UnitCell_Geo == 'Square':
            centroid  = [[l0_x * i , l0_y * j] for i in range(nx) for j in range(ny)]
        else:
            centroid  = [[ np.sqrt(3) * l0_x * (i + (1+(-1)**(j+1))/4 + lattice_dsrdr * (2*np.random.random()-1)) , \
                          np.sqrt(3) * l0_y * (j * np.sqrt(3)/2 + lattice_dsrdr * (2*np.random.random()-1)) ] for i in range(nx) for j in range(ny)]

        if UnitCell_Geo == 'Triangular':
            ver_vor0  = Voronoi(centroid).vertices
            vor       = Voronoi(ver_vor0)
            centroid  = ver_vor0

        self.points   = np.asarray(centroid)
        pmean         = np.mean(self.points , axis=0)
        self.points  -= pmean

        return self.points

    def region_shape(self):

        """
            Implement the global geometry and points within the activated region

        """

        self.points_initial(self.unit_cell)
        if self.Global_Geometry == 'Circle' or self.Global_Geometry == 'Ellipse':
            size = np.max(self.points , axis=0) - np.min(self.points , axis=0)
            r0          = (np.min(size)/2) * self.AR_fr
            R_cent      = np.mean(self.points , axis=0)
            self.points -= R_cent
            corners     = []
            cent        = Point([0,0])
            self.geo_shape = cent.buffer(r0)

        elif self.Global_Geometry == 'Hexagon':
            size = np.max(self.points , axis=0) - np.min(self.points , axis=0)
            side_len    = (np.min(size) * np.sqrt(3)/4) * self.AR_fr
            corners     = [(-side_len , 0), (-side_len/2 , +side_len*np.sqrt(3)/2) ,\
                               (+side_len/2 , +side_len*np.sqrt(3)/2) , (+side_len , 0), \
                               (+side_len/2 , -side_len*np.sqrt(3)/2) , \
                               (-side_len/2 , -side_len*np.sqrt(3)/2)]
            self.geo_shape = Polygon(corners)

        elif self.Global_Geometry == 'Rectangle':
            size = np.max(self.points , axis=0) - np.min(self.points , axis=0)
            side_len_x  = np.min(size) * self.AR_fr * self.AR_xy[0]
            side_len_y  = np.min(size) * self.AR_fr * self.AR_xy[1]
            corners     = [(-side_len_x/2 , -side_len_y/2) , (-side_len_x/2 , +side_len_y/2), \
                            (+side_len_x/2 , +side_len_y/2) , (+side_len_x/2 , -side_len_y/2)]
            self.geo_shape = Polygon(corners)

        elif self.Global_Geometry == 'Triangle':
            size = (np.max(self.points , axis=0) - np.min(self.points , axis=0)) * self.AR_fr
            H           = np.min(size)
            side_len    = H * 2/np.sqrt(3)
            coord_ax    = size.tolist().index(H)
            cent_coord  = ( 0 , side_len * (1-np.sqrt(3))/4 )
            corners     = [(-side_len/2, -H/2 - cent_coord[1]), (0 , H/2 - cent_coord[1])\
                             , (+side_len/2 , -H/2 - cent_coord[1])]
            self.geo_shape = Polygon(corners)

        elif self.Global_Geometry == 'Hexagram':
            size = (1/2) * (np.max(self.points , axis=0) - np.min(self.points , axis=0)) * self.AR_fr
            H           = np.min(size)
            h           = H * np.sqrt(3)/3
            coord_ax    = size.tolist().index(H)
            angs        = np.asarray(list(range(0 , 360 , 30))) * np.pi/180
            dists       = np.asarray([h,H,h,H,h,H,h,H,h,H,h,H])
            Xhex        = dists * np.cos(angs)
            Yhex        = dists * np.sin(angs)
            hex_verts   = list(zip(Xhex , Yhex))
            cent_coord  = (0 , 0)
            corners     = hex_verts
            self.geo_shape = Polygon(corners)

        if self.rounded == 'True':
            self.geo_shape = self.geo_shape.buffer(self.round_coeff)

        return self.geo_shape

    def net_gen(self):

        """
            Generates the network:

            outputs: vertices, edges, unit cells (in terms of the list of vertices that belong to each region)

        """


        vor         = Voronoi(self.points)
        verts_0     = deepcopy(vor.vertices)
        verts_0     = np.around(verts_0[:], decimals=2)
        edges_0     = deepcopy(vor.ridge_vertices)
        regions_0   = deepcopy(vor.regions)
        rid_pts_0   = deepcopy(vor.ridge_points)

        edges_inf_index = list(set([edges_0.index(_) for _ in edges_0 if -1 in _]))
        rid_pts         = np.delete(rid_pts_0 , edges_inf_index , 0)

        edges_1     = [_ for _ in edges_0 if -1 not in _]

        ver_ed0     = [[] for _ in range(len(verts_0))]
        for ee in range(len(edges_1)):
            ver_ed0[edges_1[ee][0]].append(ee)
            ver_ed0[edges_1[ee][1]].append(ee)

        regions_1   = [_ for _ in regions_0 if -1 not in _]
        ver_reg0    = [[] for _ in range(len(verts_0))]
        for reg in range(len(regions_1)):
            for reg_size in range(len(regions_1[reg])):
                ver_reg0[regions_1[reg][reg_size]].append(reg)

        ver_inf     = [edges_0[_][1] for _ in edges_inf_index]

        p_cent      = deepcopy(self.points)
        [X_cent_max , Y_cent_max] = np.max(p_cent , axis = 0)
        [X_cent_min , Y_cent_min] = np.min(p_cent , axis = 0)

        ver_unwanted = []

        for vv in verts_0:
            if ( vv[0] < X_cent_min or vv[0] > X_cent_max or\
                 vv[1] < Y_cent_min or vv[1] > Y_cent_max ) :
                 ver_unwanted.append(verts_0.tolist().index(vv.tolist()))

        ver_unwanted += list(np.random.randint(0 , len(ver_ed0) , (self.n_ver_rmv,)))

        ver_unwanted = sorted(set(ver_unwanted))

        ############################## Updating Edge List ##############################

        edges_unwanted_list = list(np.asarray(ver_ed0)[ver_unwanted])
        edges_unwanted = list(set([item for sublist in edges_unwanted_list for item in sublist]))
        extra_edges_unwanted = list(np.random.randint(0 , len(edges_1) , (self.n_edg_rmv,)))

        ed_ver_0 = np.zeros((len(edges_1) , len(verts_0)))

        for ee in range(len(edges_1)):
            ed_ver_0[ee , edges_1[ee][0]] = 1
            ed_ver_0[ee , edges_1[ee][1]] = -1

        ed_ver0_sps = sparse.csr_matrix(ed_ver_0)
        Rij_0 = np.sqrt( np.sum( ed_ver0_sps.dot(verts_0)**2 , axis=1 ) )
        edges_unwanted += set(extra_edges_unwanted)
        edges_unwanted = sorted(set(edges_unwanted))

        ver_ed_un = []
        for vv in range(len(verts_0)):
            if len( set(ver_ed0[vv]) - set(edges_unwanted) ) <= 1:
                ver_ed_un.append(vv)
                edges_unwanted = list(set(edges_unwanted).union(set(ver_ed0[vv])))

        ed_aux = deepcopy(edges_1)
        edges_unsorted = [ed_aux[_] for _ in range(len(ed_aux)) if _ not in edges_unwanted]

        ver_unwanted += ver_ed_un
        ver_unwanted = sorted(set(ver_unwanted))
        v_dic = [_ for _ in sorted( set( list(range(len(verts_0))) ) - set(ver_unwanted) ) ]

        self.verts_unsorted = verts_0[v_dic]

        for ee in edges_unsorted:
            [ee[0],ee[1]] = [v_dic.index(ee[0]) , v_dic.index(ee[1])]

        def array_bool(v):
            u = np.asarray([ int(bool(_)) for _ in v.flatten() ])
            u = u.reshape(v.shape)
            return u

        if bool(edges_unwanted):

            ver_affected = np.asarray(edges_1)[edges_unwanted]
            regs_affect = np.asarray(ver_reg0)[ver_affected]
            regs_2merge = [set(_[0]).intersection(set(_[1])) for _ in regs_affect]

            regs_2merge = np.sort([list(_) for _ in regs_2merge])
            reg_bm_set_0 = set([item for sublist in regs_2merge for item in sublist])
            reg_bm_list_0 = list(reg_bm_set_0)
            node_tot_number = len(reg_bm_set_0)

            Ad = np.zeros( 2 * (node_tot_number,) )
            for rr in regs_2merge:
                deg = len(rr)
                idx_col0 = [reg_bm_list_0.index(rr[ii]) for ii in range(deg) for jj in range(ii,deg)]
                idx_col1 = [reg_bm_list_0.index(rr[jj]) for ii in range(deg) for jj in range(ii,deg)]
                Ad[idx_col0 , idx_col1] = 1

            Ad += np.transpose(Ad)

            components = []
            reg_bm_set = deepcopy(reg_bm_set_0)

            while bool(reg_bm_set):
                nodes = np.zeros((node_tot_number,))
                node_init =  random.choice(list(reg_bm_set))
                nodes[reg_bm_list_0.index(node_init)] = 1
                nodes_map = array_bool(np.dot(Ad , nodes))

                while bool( np.sum(nodes_map != nodes) ) :
                    nodes = array_bool(nodes + nodes_map)
                    nodes_map = array_bool(np.dot(Ad, nodes))

                node_inds = np.asarray(list(reg_bm_set_0))[np.asarray(np.nonzero(nodes_map))]
                components.append(list(node_inds[0]))
                reg_bm_set.difference_update(set(node_inds[0]))

            regions_bulk_merged = deepcopy(components)

        else:
            reg_bm_list_0 = []
            regions_bulk_merged = []

        self.regions = [regions_1[_] for _ in range(len(regions_1)) if -1 not in regions_1[_] \
                   if _ not in reg_bm_list_0]

        for cluster in regions_bulk_merged:
            cluster_verts = list(np.asarray(regions_0)[cluster])
            cluster_verts = list(set([_ for item in cluster_verts for _ in item])-set(ver_unwanted))
            if -1 not in cluster_verts:
                self.regions.append(cluster_verts)

        for reg in self.regions:
            for ver in reg:
                self.regions[self.regions.index(reg)][reg.index(ver)] = v_dic.index(ver)

        SortedNodes = self.partial_sort(self.verts_unsorted)
        self.verts = deepcopy(SortedNodes[:,1:3])
        self.edges = []
        for ee in edges_unsorted:
            self.edges.append( sorted( [np.where(SortedNodes[:,0] == ee[0])[0][0] ,\
                           np.where(SortedNodes[:,0] == ee[1])[0][0]] ) )

        ed_ver = np.zeros((len(self.edges) , len(self.verts)))
        for ee in range(len(self.edges)):
            ed_ver[ee , self.edges[ee][0]] = 1
            ed_ver[ee , self.edges[ee][1]] = -1

        self.ed_ver_sps = sparse.csr_matrix(ed_ver)

        self.Nv = len(self.verts)
        self.Ne = len(self.edges)

        self.ver_ed_list = [[] for _ in range(self.Nv)]
        for ee in range(self.Ne):
            self.ver_ed_list[self.edges[ee][0]].append(ee)
            self.ver_ed_list[self.edges[ee][1]].append(ee)

        self.ver_ed = ed_ver.T

        self.ver_bulk_ind = [_ for _ in range(self.Nv) if len(self.ver_ed_list[_]) == self.bulkV_deg]
        self.ver_bndr_ind = list( set([_ for _ in range(self.Nv)]) - set(self.ver_bulk_ind) )

        self.ver_ed_bulk_sps = sparse.csr_matrix(self.ver_ed[self.ver_bulk_ind])

        return self.verts , self.edges , self.regions , self.ed_ver_sps, self.verts_unsorted

    ################# Partial Sorting in X - Y Directions in order #################

    def partial_sort(self,nodes):

        """
            Partial sorting of the list of vertices in the order (x_index , y_index)
        """

        nodemap = np.ones((len(nodes) , 3))
        nodemap[:,0] = np.arange(len(nodes)).reshape(len(nodes),)
        nodemap[:,1:3] = nodes
        nodeXsort = np.sort(nodemap.view('f8,f8,f8'), order=['f1'], axis=0).view(np.float)
        Xuniq = np.sort(np.unique(nodes[:,0]))
        SortedNodes = np.zeros(nodeXsort.shape)

        xind = 0
        for xx in Xuniq:
            indXuniq = np.asarray(np.where(nodeXsort[:,1] == xx))[0].tolist()
            nodeXuniq = nodeXsort[indXuniq]
            nodeXuniqY = np.sort(nodeXuniq.view('f8,f8,f8'), order=['f2'], axis=0).view(np.float)
            SortedNodes[xind:xind+len(indXuniq)] = deepcopy(nodeXuniqY)
            xind += len(indXuniq)

        return SortedNodes

    def active_verts(self,activate_full):

        """
            list of vertices in the active region which is identified by self.illumreg

        """

        Nv = len(self.verts)

        [Xt , Yt] = np.transpose(deepcopy(self.verts)).reshape(2 , Nv)
        Xt = Xt.reshape(Nv , 1)
        Yt = Yt.reshape(Nv , 1)

        X0 = deepcopy(Xt)
        Y0 = deepcopy(Yt)

        ver_passive = [i for i in range(Nv) if (not self.geo_shape.contains(Point(self.verts[i].tolist())))]
        ver_passive = list(set(ver_passive) - set(self.ver_bndr_ind))
        ver_active  = list( set(self.ver_bulk_ind) - set(ver_passive) )

        if activate_full:
            veriactive  = [i for i in range(Nv)]

        self.ver_active  = ver_active
        self.ver_passive = ver_passive
        self.illumreg    = [i for i in ver_active if Y0[i] > 0 and np.abs(X0[i]) < (3/4)*np.max(X0[ver_active])]

        self.active_bndr = []

        return self.ver_active,self.ver_passive,self.illumreg

    def active_edges(self,activate_full, diff_illumreg=None):

        """
            list of edges in the active region which is identified by self.illumreg

        """

        if diff_illumreg is None:
            diff_illumreg = self.illumreg

        Ne = len(self.edges)

        edge_active = []
        for ed in range(Ne):
            if self.edges[ed][0] in self.ver_active and self.edges[ed][1] in self.ver_active:
                edge_active.append(ed)

        edge_illum = []
        for ed in edge_active:
            if self.edges[ed][0] in diff_illumreg and self.edges[ed][1] in diff_illumreg:
                edge_illum.append(ed)
        edge_passive = list(set([i for i in range(Ne)]) - set(edge_active))

        if activate_full:
            edge_active  = [i for i in range(Ne)]

        self.edge_active        = edge_active
        self.edge_passive       = edge_passive
        self.edge_diff_illum    = edge_illum

######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################

    def Inertial_Dynamics(self,T_tot,dt,mass,gamma,rhoi,s0,tau_s,N_frame=10, N_data_dump=None, do_data_dump=True, rhof=1):

        """
            Dynamic simulation of the inertial system: this means that instead of solving the steady state solution of velocity at each given time, the velocity field acquires a mass. When the mass is sent to infinity the contribution of inertial term vanishes and the steady state solution is retrieved.

        """
        if N_frame>int(T_tot/dt):
            logging.warning("Trying to save more often than there are timesteps in the integration. Reducing it to save every timeframe.")
            N_frame = int(T_tot/dt)

        if N_data_dump is None:
            N_data_dump = N_frame
            

        Nv = len(self.verts)
        Ne = len(self.edges)

        self.X_size = np.zeros((round(T_tot/dt), 1))
        self.Y_size = np.zeros((round(T_tot/dt), 1))

        [Xt , Yt] = np.transpose(deepcopy(self.verts)).reshape(2 , Nv)
        self.Xt = Xt.reshape(Nv , 1)
        self.Yt = Yt.reshape(Nv , 1)

        X0 = deepcopy(self.Xt)
        Y0 = deepcopy(self.Yt)

        self.Vx    = np.zeros((Nv , 1))
        self.Vy    = np.zeros((Nv , 1))
        self.Rx    = np.zeros((Nv , round(T_tot/dt)))
        self.Ry    = np.zeros((Nv , round(T_tot/dt)))
        self.rho_t = np.zeros((Ne , round(T_tot/dt)))
        self.len_tot = np.zeros((round(T_tot/dt) , 1))

        self.rho   = np.zeros((Ne , 1))
        self.rho[self.edge_active] = rhoi

        ################################################################################
        ################################### DYNAMICS ###################################
        ################################################################################

        self.ver_ed_active_sps = sparse.csr_matrix(self.ver_ed[self.ver_active])
        self.D2_active         = self.ver_ed_active_sps.dot(self.ver_ed_active_sps.T).todense()

        # for ii in range(len(self.ver_active)):
        #     self.D2_active[ii,ii] = self.bulkV_deg

        self.Greens_fct        = self.D2_active - gamma * np.eye(self.D2_active.shape[0])
        self.D2_active_inv     = sparse.csr_matrix(np.asarray(np.linalg.inv(self.Greens_fct)))

        noise_x = 0 * np.random.normal(0 , 0.02 , (len(self.ver_active) , 1))
        noise_y = 0 * np.random.normal(0 , 0.02 , (len(self.ver_active) , 1))

        self.Xt_active = self.Xt[self.ver_active] + noise_x
        self.Yt_active = self.Yt[self.ver_active] + noise_y

        self.Vx_active = self.Vx[self.ver_active] + noise_x
        self.Vy_active = self.Vy[self.ver_active] + noise_y

        nx_map = self.lattice_shape[1] - 1
        ny_map = self.lattice_shape[0] - 1

        for tt in tqdm(range(round(T_tot/dt))):
            self.tt += 1 ## if this function is called multiple times on the same class, do not reset self.tt

            self.Xij = self.ed_ver_sps.dot(self.Xt)
            self.Yij = self.ed_ver_sps.dot(self.Yt)
            self.Rij = np.sqrt(self.Xij**2 + self.Yij**2)
            xhat     = self.Xij/self.Rij
            yhat     = self.Yij/self.Rij

            s_t                     = np.zeros((len(self.rho),1))
            s_t[self.edge_active]   = s0 * (1 - np.exp(-tt/tau_s))
            s_t[self.edge_diff_illum] = self.illum_ratio * s0 * (1 - np.exp(-tt/tau_s))
            self.stress = s_t * (self.rho - rhof) * self.rho**(self.rhoPower)

            Fx_bulk  = + self.ed_ver_sps.T.dot(self.stress * xhat)[self.ver_active]
            Fy_bulk  = + self.ed_ver_sps.T.dot(self.stress * yhat)[self.ver_active]

            Fx_visc  = - self.D2_active.dot(self.Vx[self.ver_active])
            Fy_visc  = - self.D2_active.dot(self.Vy[self.ver_active])

            Fx_drag  = + gamma * self.Vx[self.ver_active]
            Fy_drag  = + gamma * self.Vy[self.ver_active]

            Vx_dot   = (Fx_bulk + Fx_visc + Fx_drag) / mass
            Vy_dot   = (Fy_bulk + Fy_visc + Fy_drag) / mass

            self.Vx[self.ver_active] += dt * Vx_dot
            self.Vy[self.ver_active] += dt * Vy_dot

            self.Xt_active    += dt * self.Vx[self.ver_active]
            self.Yt_active    += dt * self.Vy[self.ver_active]

            self.Xt[self.ver_active] = self.Xt_active
            self.Yt[self.ver_active] = self.Yt_active

            self.rho = self.rho + dt * self.dndt(self.Vx,self.Vy,xhat,yhat,self.Rij,self.rho)
            # self.rho    = rhoi/self.Rij
            self.len_tot[tt,0] = np.sum(self.Rij)

            Xmin , Xmax = np.min(self.Xt[self.ver_active]) , np.max(self.Xt[self.ver_active])
            Ymin , Ymax = np.min(self.Yt[self.ver_active]) , np.max(self.Yt[self.ver_active])

            self.Rx[:,tt]       = self.Xt.reshape((Nv,))
            self.Ry[:,tt]       = self.Yt.reshape((Nv,))
            self.rho_t[:,tt]    = self.rho.reshape((Ne,))
            self.X_size[tt]     = Xmax - Xmin
            self.Y_size[tt]     = Ymax - Ymin

            if tt%int((T_tot/dt)/N_frame) == 0:
                self.do_plots(X0, Y0, nx_map, ny_map)
            if do_data_dump and tt%int((T_tot/dt)/N_data_dump) == 0:
                self.do_data_dump()
        if self.do_last_frame_plot:
            self.do_plots(X0, Y0, nx_map, ny_map)
    def Viscous_Dynamics(self,T_tot,dt,gamma,rhoi,s0,tau_s,N_frame=10, N_data_dump=None, do_data_dump=True, rhof=1):

        """
            Direct solution of the velocity in the steady state. The only time-dependence is buried in the continuity equation
        """

        if N_frame>int(T_tot/dt):
            logging.warning("Trying to save more often than there are timesteps in the integration. Reducing it to save every timeframe.")
            N_frame = int(T_tot/dt)

        Nv = len(self.verts)
        Ne = len(self.edges)

        self.X_size = np.zeros((round(T_tot/dt), 1))
        self.Y_size = np.zeros((round(T_tot/dt), 1))

        [Xt , Yt] = np.transpose(deepcopy(self.verts)).reshape(2 , Nv)
        self.Xt = Xt.reshape(Nv , 1)
        self.Yt = Yt.reshape(Nv , 1)

        X0 = deepcopy(self.Xt)
        Y0 = deepcopy(self.Yt)

        self.Vx    = np.zeros((Nv , 1))
        self.Vy    = np.zeros((Nv , 1))
        self.Rx    = np.zeros((Nv , round(T_tot/dt)))
        self.Ry    = np.zeros((Nv , round(T_tot/dt)))
        self.rho_t = np.zeros((Ne , round(T_tot/dt)))
        self.len_tot = np.zeros((round(T_tot/dt) , 1))

        self.rho   = np.zeros((Ne , 1))
        self.rho[self.edge_active] = rhoi

        ################################################################################
        ################################### DYNAMICS ###################################
        ################################################################################

        self.ver_ed_active_sps = sparse.csr_matrix(self.ver_ed[self.ver_active])
        self.D2_active         = self.ver_ed_active_sps.dot(self.ver_ed_active_sps.T).todense()

        self.Greens_fct        = self.D2_active - gamma * np.eye(self.D2_active.shape[0])
        self.D2_active_inv     = sparse.csr_matrix(np.asarray(np.linalg.inv(self.Greens_fct)))

        noise_x = 0 * np.random.normal(0 , 0.02 , (len(self.ver_active) , 1))
        noise_y = 0 * np.random.normal(0 , 0.02 , (len(self.ver_active) , 1))

        self.Xt_active = self.Xt[self.ver_active] + noise_x
        self.Yt_active = self.Yt[self.ver_active] + noise_y

        self.Vx_active = self.Vx[self.ver_active] + noise_x
        self.Vy_active = self.Vy[self.ver_active] + noise_y

        nx_map  = int(np.sqrt(Nv))
        ny_map  = int(np.sqrt(len(self.ver_active)))

        for tt in tqdm(range(round(T_tot/dt))):
            self.tt += 1

            self.Xij = self.ed_ver_sps.dot(self.Xt)
            self.Yij = self.ed_ver_sps.dot(self.Yt)
            self.Rij = np.sqrt(self.Xij**2 + self.Yij**2)
            xhat     = self.Xij/self.Rij
            yhat     = self.Yij/self.Rij

            s_t                     = np.zeros((len(self.rho),1))
            s_t[self.edge_active]   = s0
            s_t[self.edge_diff_illum] = self.illum_ratio * s0
            self.stress = s_t * (self.rho - rhof) * self.rho**(self.rhoPower)

            F_bulk             = self.active_force(self.stress,xhat,yhat)
            self.Vx_active     = self.D2_active_inv.dot(F_bulk[0])
            self.Vy_active     = self.D2_active_inv.dot(F_bulk[1])

            self.Xt_active    += dt * self.Vx_active
            self.Yt_active    += dt * self.Vy_active

            self.Xt[self.ver_active] = self.Xt_active
            self.Yt[self.ver_active] = self.Yt_active

            self.Vx[self.ver_active] = self.Vx_active
            self.Vy[self.ver_active] = self.Vy_active

            self.rho = self.rho + dt * self.dndt(self.Vx,self.Vy,xhat,yhat,self.Rij,self.rho)
            # self.rho    = rhoi/self.Rij
            self.len_tot[tt,0] = np.sum(self.Rij)

            Xmin , Xmax = np.min(self.Xt[self.ver_active]) , np.max(self.Xt[self.ver_active])
            Ymin , Ymax = np.min(self.Yt[self.ver_active]) , np.max(self.Yt[self.ver_active])

            self.Rx[:,tt]       = self.Xt.reshape((Nv,))
            self.Ry[:,tt]       = self.Yt.reshape((Nv,))
            self.rho_t[:,tt]    = self.rho.reshape((Ne,))
            self.X_size[tt]     = Xmax - Xmin
            self.Y_size[tt]     = Ymax - Ymin

            if tt%int((T_tot/dt)/N_frame) == 0:
                self.do_plots(X0, Y0, nx_map, ny_map)
            if do_data_dump and tt%int((T_tot/dt)/N_data_dump) == 0:
                self.do_data_dump()

        self.total_time = tt

        if self.do_last_frame_plot:
            self.do_plots(X0, Y0, nx_map, ny_map)
            if do_data_dump:
                self.do_data_dump()

    def artifact_name(self,modifier, extension='png', do_running_index=True):
        if do_running_index:
            return f"{self.working_dir}/{modifier}-{self.tt:04}.{extension}"
        else:
            return f"{self.working_dir}/{modifier}.{extension}"

    def show_or_save(self, modifier):
        if self.show_plots:
            plt.show()
        else:
            save_name = self.artifact_name(modifier)
            plt.savefig(save_name)
            if self.sacred_experiment is not None:
                self.sacred_experiment.add_artifact(save_name)
        plt.close(plt.gcf())

    def do_plots(self, X0, Y0, nx_map, ny_map):
        Xframe = [1.5 * np.min(X0) , 1.5 * np.max(X0)]
        Yframe = [1.5 * np.min(Y0) , 1.5 * np.max(Y0)]

        if self.plot_full_positions:
            fscale = 7
            fig1, ax1 = plt.subplots(figsize=(fscale, fscale), dpi= 100, facecolor='w', edgecolor='k')
            ax1.plot(self.Xt , self.Yt, '.' , color='w' , markersize = 0.2 * fscale)
            ax1.plot(self.Xt[self.ver_active] , self.Yt[self.ver_active], '.' , color='c' , markersize = 0.2 * fscale)
            ax1.set_facecolor('k')
            ax1.axis('equal')
            plt.xlim(Xframe)
            plt.ylim(Yframe)
            self.show_or_save("full_positions")

        if self.plot_velocity_plot:
            logging.warning("Currently, plot velocity doesn't work. Skipping this part.")
            #plt.figure(figsize=(7,7))
            #VxMap = np.reshape(self.Vx[self.ver_active], (nx_map,ny_map)).T
            #VyMap = np.reshape(self.Vy[self.ver_active], (nx_map,ny_map)).T
            #XtMap = np.reshape(self.Xt[self.ver_active], (nx_map,ny_map)).T
            #YtMap = np.reshape(self.Yt[self.ver_active], (nx_map,ny_map)).T
            #plt.plot(XtMap[int(self.lattice_shape[0]/2),:] , VxMap[int(self.lattice_shape[0]/2),:] , '*')
            #plt.plot(YtMap[:,int(self.lattice_shape[0]/2)] , VyMap[:,int(self.lattice_shape[0]/2)] , '-')
            #plt.xlim(Xframe)
            #if self.show_plots:
            #    plt.show()
            #else:
            #    plt.savefig(self.artifact_name("velocity_plot"))

        if self.plot_velocity_map:
            if self.UnitCell_Geo == 'Square':
                fig = plt.figure(figsize=(8,4))
                plt.subplot(1,2,1)
                vmax = np.max([np.max(abs(v)) for v in [self.Vx, self.Vy]])
                VxMap = np.reshape(self.Vx, (nx_map,nx_map)).T
                plt.imshow(VxMap, cmap='RdBu', vmin = -vmax, vmax = vmax)

                plt.subplot(1,2,2)
                VyMap = np.reshape(self.Vy, (nx_map,nx_map)).T
                plt.imshow(VyMap, cmap='RdBu', vmin = -vmax, vmax = vmax)
                self.show_or_save("velocity_map")
            else:
                logging.warning("Velocity plot currently only works with Squre unit cell geometry.")
                self.plot_velocity_map = False

        if self.plot_density_map:
            if self.UnitCell_Geo == 'Square':
                fig = plt.figure(figsize=(7,7))
                rhoMap = np.reshape(self.ver_ed.dot(self.rho), (nx_map,nx_map)).T
                plt.imshow(rhoMap)
                self.show_or_save("density_map")
            else:
                logging.warning("Density plot currently only works with Squre unit cell geometry.")
                self.plot_density_map = False
    def do_data_dump(self):
        if self.dump_type is None:
            pass # Setting dump type to None will stop any data dump to occur
        elif self.dump_type.lower()=='json':
            dd = self.get_dump_dir()
            self.do_json_data_dump(dd)
        elif self.dump_type.lower()=='h5':
            dd = self.get_dump_dir()
            self.do_h5_data_dump(dd)
        else:
            raise RuntimeWarning(f"Unrecognized data dump type: {self.dump_type}... Valid options are 'json', 'h5', or None")
    def get_dump_dir(self):
        store_dir = dict()
        store_dir['rho'] = self.rho.flatten().tolist()
        store_dir['Xt'] = self.Xt.flatten().tolist()
        store_dir['Yt'] = self.Yt.flatten().tolist()
        store_dir['ver_active'] = self.ver_active
        return store_dir
    def do_json_data_dump(self, store_dir):
        save_name = self.artifact_name('data_dump', extension='json')
        with open(save_name, 'w') as f:
            json.dump(store_dir, f, indent=4)
        if self.sacred_experiment is not None:
            self.sacred_experiment.add_artifact(save_name)
    def do_h5_data_dump(self, store_dir):
        save_name = self.artifact_name('data_dump', extension='h5', do_running_index=False)
        with h5py.File(save_name, 'a') as f:
            basedir = f'iteration={self.tt}'
            f.create_group(basedir)
            for k,v in store_dir.items():
                f.create_dataset(f"{basedir}/{k}", data=v)
        if self.sacred_experiment is not None:
            self.sacred_experiment.add_artifact(save_name)


    def active_force(self,stress,xhat,yhat):

        """
            calculates the force on active vertices
        """

        self.Fx      = self.ed_ver_sps.T.dot(stress * xhat)
        self.Fy      = self.ed_ver_sps.T.dot(stress * yhat)

        self.Fx_bulk = self.Fx[self.ver_active]
        self.Fy_bulk = self.Fy[self.ver_active]

        return self.Fx_bulk , self.Fy_bulk

    def dndt(self,Vx,Vy,xhat,yhat,Rij,rho):

        """
            continuity equation
        """

        dVx     = self.ed_ver_sps.dot(Vx)
        dVy     = self.ed_ver_sps.dot(Vy)
        divV    = dVx * xhat/Rij**1 + dVy * yhat/Rij**1
        drho    = - rho * divV
        return drho

    def edge_centers(self):

        """
            Coordinates of the center of edges
        """

        endAvg  = np.abs(self.ed_ver_sps)
        edcents = endAvg.dot(self.verts)
        return edcents
