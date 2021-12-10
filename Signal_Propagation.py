"""
Created in June 2021
@author: Shahriar Shadkhoo -- CALTECH
"""

import numpy as np
from scipy.spatial.distance import pdist , squareform
from scipy.sparse import csr_matrix, triu
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import networkx as nx
from geometry_funcs import patches , lattice
from copy import deepcopy

seeding = True
seednum = 30
if seeding:
    np.random.seed(seednum)

plotimg , saveimg = True, False
unitcell  = 'square'

BoundaryCond = 'reflective_BC'

signaling = True
mechanism = 'Proximity'   # or Fisher

# Sets activity, elastic interactions, drage interaction, and (thermal/active) noise on (1) or off (0)
actv_on , elas_on , drag_on , nois_on = 0 , 1 , 1 , 1
disorder= True

T_tot , dt = 20 , 0.01                              # Total time and resolution
N_frame = np.min((50,int(T_tot/dt)))                # number of frames to plot
tseries = range(round(T_tot/dt))
Nt      = len(tseries)

rearr   = True ; Trearr  = dt
ReT     = int( np.max((1,np.ceil(Trearr/dt))) )      # Contact Rearrangement Times

Temp_v  = 1                                         # Translational temperature
Temp_r  = 1                                         # Rotational temperature
a0      = 5                                         # Activity
gamma   = 10                                         # Drag coefficient
MM      = 1                                         # Mass of particles
KK      = 10                                         # Collision rigidity
nu      = 1                                         # Signal transmission rate between particles
kappa   = .6                                         # Signal decay rate

tplot = []
if plotimg:
    tplot = list(np.unique(np.around(np.linspace(0 , Nt , N_frame+1)).astype(int)))

t_rearr = []
if rearr:
    t_rearr = np.around(np.arange(ReT , Nt-1 , ReT)).astype(int)

XY_lens = [20,20]                                         # Confining box size
dens    = 1                                               # Number density of particles

if unitcell!='square':
    XY_lens[1] *= np.sqrt(3/2)
    XY_lens[1] -= (1+int(round(XY_lens[1]*np.sqrt(dens))))%2

[R_cnts , Ntot] = lattice.Lattice_Points(XY_lens, dens , disorder=disorder, unit_cell=unitcell)

rad_avg = .5                                          # Average radius
rad_std = 0.0                                         # Standard deviation of radii
rads = np.abs(np.random.normal(rad_avg , rad_std , (Ntot,1)))
exp_rads    = np.exp(rads)
sum_rads    = np.log(np.kron(exp_rads , exp_rads)).reshape((Ntot,Ntot))[np.triu_indices(Ntot , 1)]
Rcutoff = 1                                         # Interaction radius cutoff in units of 2*radius

xcnt = R_cnts[:,0].reshape((Ntot,1))
ycnt = R_cnts[:,1].reshape((Ntot,1))

xmin0 , xmax0 , ymin0 , ymax0 = np.min(xcnt) , np.max(xcnt) , np.min(ycnt) , np.max(ycnt)

Vx  = np.zeros((Ntot,1))
Vy  = np.zeros((Ntot,1))
ang = np.random.uniform(-np.pi,+np.pi, (Ntot,1))               # Propulsion angle in the case of active case

Cs = np.zeros((Ntot,1))                                         # Concentration of signal
if signaling:                                                       # Determine the initial distribution of the signal
    cent_ind = [_ for _ in range(Ntot) if xcnt[_] < 0]
    cent_ind = [_ for _ in range(Ntot) if np.abs(xcnt[_]) < 1 and np.abs(ycnt[_]) < 1]
    Cs[cent_ind] = 1

############################################################### CONNECTIVITY FUNCTIONS ###############################################################

def diff_operators(NN,R_cnts,sum_rads,Rcutoff=Rcutoff):

    ds     = pdist(R_cnts, 'euclidean')
    ds_rad = np.heaviside( - ds + sum_rads*Rcutoff , 0 ).astype(int)
    ds_sq  = squareform(ds_rad) * np.triu(np.ones((NN,NN)),1)

    v1     = np.where(ds_sq!=0)[0]
    v2     = np.where(ds_sq!=0)[1]
    Ncnst  = len(v1)                            # number of constraints (particle-particle contacts)
    Bs     = np.zeros((Ncnst , NN))
    Bs[np.arange(Ncnst) , v1] = +1
    Bs[np.arange(Ncnst) , v2] = -1

    C_prop = deepcopy(Bs.T.dot(Bs))
    np.fill_diagonal(C_prop,0)
    C_prop = csr_matrix(C_prop)
    Bs = csr_matrix(Bs)

    G = nx.Graph()
    ed_list = list(zip(v1,v2))
    G.add_edges_from(ed_list)

    return Bs,C_prop,ed_list,G

[Bs,C_prop,ed_list,G] = diff_operators(Ntot,R_cnts,sum_rads)

l_rest = np.abs(Bs).dot(rads)                   # rest length in elastic interaction of the particle pairs in contact

################################################################### PLOT SETTINGS ####################################################################

fsize = (10,10)
marg = 1.2
txtsize = 7*fsize[0]/5
s0 = 5*fsize[0] * (10/max(XY_lens))
plot_set = {'color':'w' , 'alpha':1 , 'fontsize':txtsize}

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"]})

def box_corners(xcnt0,ycnt0,rads,ext=1):
    x_corners = [np.min(xcnt0-ext*rads) , np.max(xcnt0+ext*rads) , np.max(xcnt0+ext*rads) , np.min(xcnt0-ext*rads) , np.min(xcnt0-ext*rads)]
    y_corners = [np.min(ycnt0-ext*rads) , np.min(ycnt0-ext*rads) , np.max(ycnt0+ext*rads) , np.max(ycnt0+ext*rads) , np.min(ycnt0-ext*rads)]
    return x_corners , y_corners

[x_corners , y_corners] = box_corners(xcnt,ycnt,rads,ext=1)
xmin , xmax = min(x_corners) , max(x_corners)
ymin , ymax = min(y_corners) , max(y_corners)
XLIM , YLIM = [np.min((-5,marg*x_corners[0])) , np.max((5,marg*x_corners[1]))] , [np.min((-5,marg*y_corners[0])) , np.max((5,marg*y_corners[2]))]

color_circs = np.asarray([[0,.5,.7]]) * np.ones((Ntot,3))
color_circs[:,0] = Cs.reshape((Ntot,))

################################################################### PLOT FUNCTIONS ###################################################################

def plot_particles(Cs=Cs,tt=0):
    color_circs[:,0] = Cs.reshape((Ntot,))
    fig = plt.figure(facecolor='w',figsize=fsize,dpi=100)
    patches.circles(xcnt,ycnt,rads,color=color_circs, alpha=.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().set_facecolor('k')
    if BoundaryCond != 'Free':
        plt.plot(x_corners, y_corners, 'gray' ,linewidth = 1)
    plt.xlim(XLIM)
    plt.ylim(YLIM)
    plt.text(.9*xmin,1.07*ymax,'time $=$ '+str(np.around(tt*dt,2)),**plot_set)
    plt.text(.7*xmax,1.07*ymax,'$\phi =$ '+str(np.around(PackF,2)),**plot_set)
    plt.text(.9*xmin,1.11*ymin,'Signal. Mech.: '+mechanism,c='w',alpha=1,fontsize=txtsize*(2/3))
    plt.text(.1*xmin,1.11*ymin,'$\langle \mathcal{D} \\rangle =$ '+str(np.around(2*rad_avg,2)),**plot_set)
    plt.text(.7*xmax,1.11*ymin,'seed $=$ '+str(seednum),**plot_set)
    if saveimg:
        fname = 'Simulation_Movie_Disks/disks_t_%07d.tif'%tt
        plt.savefig(fname)
    plt.show()

def plot_graph(rs,ed_list,tt=0,plot_eds=True,plot_radius=True):
    color_circs[:,0] = Cs.reshape((Ntot,))
    xs, ys = rs[:,0] , rs[:,1]
    fig = plt.figure(facecolor='w',figsize=fsize,dpi=100)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().set_facecolor('k')
    if BoundaryCond != 'Free':
        plt.plot(x_corners, y_corners, 'gray' ,linewidth = 1)
    ed_coord = R_cnts[np.asarray(ed_list)].tolist()
    if plot_eds:
        lc = mc.LineCollection(ed_coord,color=np.abs(Bs).dot(color_circs)/2,alpha=.7,linewidths=.9)
        plt.gca().add_collection(lc)
    nodes = plt.scatter(xs,ys,color=(2/3)*color_circs,s=s0,zorder=2)
    plt.text(.9*xmin,1.07*ymax,'time $=$ '+str(np.around(tt*dt,2)),**plot_set)
    plt.text(.7*xmax,1.07*ymax,'$\phi =$ '+str(np.around(PackF,2)),**plot_set)
    plt.text(.9*xmin,1.11*ymin,'Signal. Mech.: '+mechanism,c='w',alpha=1,fontsize=txtsize*(2/3))
    plt.text(.1*xmin,1.11*ymin,'$\langle \mathcal{D} \\rangle =$ '+str(np.around(2*rad_avg,2)),**plot_set)
    plt.text(.7*xmax,1.11*ymin,'seed $=$ '+str(seednum),**plot_set)
    plt.xlim(XLIM)
    plt.ylim(YLIM)
    if saveimg:
        fname = 'Simulation_Movie_Graph/graph_t_%07d.tif'%tt
        plt.savefig(fname)
    plt.show()

def plot_timefcts(Xt,VarName=[],tseries=tseries):
    fig = plt.figure(facecolor='w',dpi=100,figsize=fsize)
    plt.scatter(dt*np.array(tseries), Xt, c = plt.cm.viridis(np.linspace(0.1,1,Nt))[::-1] , s=1)
    plt.xlabel('\it{time}',fontsize=20)
    plt.ylabel(VarName,fontsize=20)
    plt.text(0,np.min(Xt)+.95*(np.max(Xt)-np.min(Xt)),'Mechanism : '+mechanism,c='k',alpha=1,fontsize=txtsize)
    if saveimg:
        fname = f'R2_C_{dens}.png'
        plt.savefig(fname)
    plt.show()

################################################################## SETUP VARIABLES ###################################################################

N_track = Ntot
inds_tr = [_ for _ in range(N_track)]
t_track = [_ for _ in range(Nt)]
traj_x  = np.zeros((N_track , len(t_track)))
traj_y  = np.zeros((N_track , len(t_track)))
Cs_t    = np.zeros((N_track , len(t_track)))

R2_C    = np.zeros((Nt,1))
Ctot    = np.zeros((Nt,1))

PackF = np.sum((np.pi)*rads**2)/((xmax-xmin)*(ymax-ymin))           # Packing fraction of NON-overlapping particles

plot_particles(Cs , 0)
plot_graph(R_cnts, ed_list)

##################################################################### TIME LOOP ######################################################################

ti = 0

for tt in tseries:

    if tt in t_track:
        traj_x[:,ti] = xcnt[inds_tr].reshape(N_track,)
        traj_y[:,ti] = ycnt[inds_tr].reshape(N_track,)
        Cs_t[:,ti]   = Cs[inds_tr].reshape(N_track,)
        ti+=1

    if tt in t_rearr:
        [Bs,C_prop,ed_list,G] = diff_operators(Ntot,R_cnts,sum_rads)
        l_rest = np.abs(Bs).dot(rads)

    Dx = Bs.dot(xcnt)
    Dy = Bs.dot(ycnt)
    Dr = np.sqrt(Dx**2 + Dy**2)

    Fax = Fay = 0
    if actv_on:
        Fax = a0 * np.cos(ang)
        Fay = a0 * np.sin(ang)

    Fpx = Fpy = 0
    if elas_on:
        Fpx = - KK * Bs.T.dot((Dr-l_rest)* np.heaviside(-Dr+l_rest , 0)*Dx/Dr)
        Fpy = - KK * Bs.T.dot((Dr-l_rest)* np.heaviside(-Dr+l_rest , 0)*Dy/Dr)

    Fdx = Fdy = 0
    if drag_on:
        Fdx = - gamma * Vx
        Fdy = - gamma * Vy

    noise_vx = noise_vy = noise_ang = 0
    if nois_on:
        noise_vx  = Temp_v**.5 * np.random.normal(0 , 1 , (Ntot,1))
        noise_vy  = Temp_v**.5 * np.random.normal(0 , 1 , (Ntot,1))
        noise_ang = Temp_r**.5 * np.random.normal(0 , 1 , (Ntot,1))

    dCdt = 0
    if signaling:
        if mechanism == 'Fisher':
            dCdt  = -nu*Bs.T.dot(Bs.dot(Cs)) + kappa*Cs*(1-Cs)     # FISHER equation.
        else:     dCdt  = - nu*(1-Cs)*C_prop.dot(Cs) - kappa*Cs

    ax    = (Fax + Fpx + Fdx + noise_vx)/MM
    ay    = (Fay + Fpy + Fdy + noise_vy)/MM
    xcnt += Vx*dt + (ax * dt**2)/2
    ycnt += Vy*dt + (ay * dt**2)/2
    Vx   += ax * dt
    Vy   += ay * dt
    Cs   += dCdt * dt
    ang += noise_ang * dt

    if signaling:
        Ctot[tt] = np.sum(Cs)
        xC_com   = np.mean(xcnt*Cs)
        yC_com   = np.mean(ycnt*Cs)
        R2_C[tt] = np.sum( ( (xcnt - xC_com)**2 + (ycnt - yC_com)**2 ) * Cs )/np.sum(Cs)

    if BoundaryCond == 'reflective_BC':                     # Bouncing particles off the walls
        ind_ymax = np.where(ycnt+rads > ymax)[0]
        ind_xmax = np.where(xcnt+rads > xmax)[0]
        ind_ymin = np.where(ycnt-rads < ymin)[0]
        ind_xmin = np.where(xcnt-rads < xmin)[0]
        ycnt[ind_ymax] = 2*(ymax-rads[ind_ymax]) - ycnt[ind_ymax]
        ycnt[ind_ymin] = 2*(ymin+rads[ind_ymin]) - ycnt[ind_ymin]
        xcnt[ind_xmax] = 2*(xmax-rads[ind_xmax]) - xcnt[ind_xmax]
        xcnt[ind_xmin] = 2*(xmin+rads[ind_xmin]) - xcnt[ind_xmin]
        Vy[list(ind_ymax) + list(ind_ymin)]  *= -1
        Vx[list(ind_xmax) + list(ind_xmin)]  *= -1
        ang[list(ind_ymax) + list(ind_ymin)] *= -1
        ang[list(ind_xmax) + list(ind_xmin)]  = np.pi - ang[list(ind_xmax) + list(ind_xmin)]

    if tt in tplot:
        plot_particles(Cs , tt)


plot_particles(Cs , tt+1)
plot_graph(R_cnts, ed_list, tt+1)
