#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 17:44:36 2021

@author: amal
"""

import numpy as np
from scipy import integrate
from scipy.optimize.nonlin import newton_krylov
# import scipy.sparse.linalg as spla
import time
from mpi4py import MPI

COMM = MPI.COMM_WORLD  # The default communicator
SIZE = COMM.Get_size() # The number of processes
RANK = COMM.Get_rank() # The rank of processes

''' inputs '''
T=3.0 # horizon length 
N=1 # number of cars 
u_max=1.0 # free flow speed
rho_jam=1.0 # jam density
L=N # road length
CFL=0.75    # CFL<1
rho_a=0.05; rho_b=0.95; gama=0.1
# rho_a=0.2; rho_b=0.8; gama=0.15*L
costf="LWR"
# """ Viscous solution"""
EPS=0.45
mu=0.0 # viscosity coefficient 

Nx_list=[]
Nt_list=[]

""" grid discretization """
Nx=120; Nt=4*Nx # spatial-temporal grid sizes 
dx=L/Nx # spatial step size
if mu==0.0:
    dt=min(T/Nt,(CFL*dx)/u_max) # temporal step size
    eps=0.0
else:
    dt=min(T/Nt,CFL*dx/abs(u_max),EPS*(dx**2)/mu) # temporal step size
    eps=mu*dt/(dx**2) # V
x=np.linspace(0,L,Nx+1)
# t=np.linspace(0,T,Nt+1)
t=np.arange(0,T+dt,dt)
Nt=len(t)-1
print('Nx={Nx}, Nt={Nt}'.format(Nx=Nx,Nt=Nt))
print('dx={dx}, dt={dt}'.format(dx=round(dx,4),dt=round(dt,4)))


""" for MPI : Creates a division of processors in a cartesian grid """
nx=Nx; nt=Nt+1
px=int(np.sqrt(SIZE))-1 # number of processes on each line-1
pt=px # number of processes on each column-1
# print("px={px}, pt={pt}".format(px=px, pt=pt))
new_size=(px+1)*(pt+1) # the Number of processes to decompose 
# print('new_size=',new_size)
nbrx=int(Nx/(px+1)) #number of points for px (except root)
nbrt=int((Nt+1)/(pt+1)) #number of points for pt (except root)
dims=[px+1,pt+1] # The array containing the number of processes to assign to each dimension
# print('dims=',dims)
npoints=[nx,nt]
# print('npoints=',npoints)


''' functions '''
def U(rho): # Greenshields desired speed
    return u_max*(1-rho/rho_jam)

def f_mfg(u,r):
    if costf=="LWR":
        return 0.5*((U(r)-u)**2) # MFG-LWR
    elif costf=="Sep":
        return 0.5*((u/u_max)**2)-(u/u_max)+(r/rho_jam) # MFG-Separable
    elif costf=="NonSep":
        return 0.5*((u/u_max)**2)-(u/u_max)+((u*r)/(u_max*rho_jam)) # MFG-NonSeparable

def f_star_p(p,r): # 0<=u<=u_max
    if costf=="LWR":
        return U(r)-p # MFG-LWR
    elif costf=="Sep":
        return max(min(u_max*(1-p*u_max),u_max),0) # MFG-Separable
    elif costf=="NonSep":
        return max(min(u_max*(1-r/rho_jam-u_max*p),u_max),0) # MFG-NonSeparable
    
def f_star(p,r): # p=Vx
    if costf=="LWR":
        return -0.5*(p**2)+U(r)*p # MFG-LWR
    elif costf=="Sep":
        return f_star_p(p,r)*p+f_mfg(f_star_p(p,r),r) # MFG-Separable
    elif costf=="NonSep":
        return f_star_p(p,r)*p+f_mfg(f_star_p(p,r),r) # MFG-NonSeparable

def integral(a,b): 
    x2 = lambda x: rho_int(x)
    I=integrate.quad(x2, a, b)
    return I[0]

def rho_int(s): # initial density
    return rho_a+(rho_b-rho_a)*np.exp(-0.5*((s-0.5*L)/gama)**2) # 0<=rho<=rho_jam

def VT(a): # Terminal cost
    return 0.0

def r_idx(j,n):
    return (j-1)*(Nt+1)+n

def u_idx(j,n):
    return (Nt+1)*Nx+(j-1)*Nt+n

def V_idx(j,n):
    return (2*Nt+1)*Nx+(j-1)*(Nt+1)+n


def F(w):
    # FF=[F_rho,F_u,F_V,F_rho_int,F_V_ter], F_rho:0->Nt*Nx-1, F_u:Nt*Nx->2*Nt*Nx-1, F_V:2*Nt*Nx->3*Nt*Nx-1, F_rho_int:3*Nt*Nx->3*Nt*Nx+Nx-1, F_V_ter:3*Nt*Nx+Nx->3*Nt*Nx+2*Nx-1
    FF=np.zeros(3*Nt*Nx+2*Nx)
    for n in range(0,Nt):
        # F_rho , F[0]->F[Nt-1] ************** 1  
        FF[n]=w[r_idx(1,n+1)]-0.5*(w[r_idx(Nx,n)]+w[r_idx(2,n)])+(0.5*dt/dx)*(w[r_idx(2,n)]*w[u_idx(2,n)]-w[r_idx(Nx,n)]*w[u_idx(Nx,n)])
        # F_rho , F[Nt*Nx-Nt]->F[Nt*Nx-1] ********** 3 
        FF[Nt*(Nx-1)+n]=w[r_idx(Nx,n+1)]-0.5*(w[r_idx(Nx-1,n)]+w[r_idx(1,n)])+(0.5*dt/dx)*(w[r_idx(1,n)]*w[u_idx(1,n)]-w[r_idx(Nx-1,n)]*w[u_idx(Nx-1,n)])
        # F_u , F[Nt*Nx]->F[Nt*Nx+Nt-1] *********** 4 
        FF[Nt*Nx+n]=w[u_idx(1,n)]-f_star_p((w[V_idx(1,n+1)]-w[V_idx(Nx,n+1)])/dx,w[r_idx(1,n)])
        # F_u , F[2*Nt*Nx-Nt]->F[2*Nt*Nx-1] ********* 6 
        FF[2*Nt*Nx-Nt+n]=w[u_idx(Nx,n)]-f_star_p((w[V_idx(Nx,n+1)]-w[V_idx(Nx-1,n+1)])/dx,w[r_idx(Nx,n)])
        # F_V , F[2*Nt*Nx]->F[2*Nt*Nx+Nt-1] *********** 7 
        FF[2*Nt*Nx+n]=w[V_idx(1,n+1)]-w[V_idx(1,n)]+dt*f_star((w[V_idx(1,n+1)]-w[V_idx(Nx,n+1)])/dx,w[r_idx(1,n)])+eps*(w[V_idx(2,n+1)]-2*w[V_idx(1,n+1)]+w[V_idx(Nx,n+1)])
        # F_V , F[3*Nt*Nx-Nt]->F[3*Nt*Nx-1] ********** 9 
        FF[3*Nt*Nx-Nt+n]=w[V_idx(Nx,n+1)]-w[V_idx(Nx,n)]+dt*f_star((w[V_idx(Nx,n+1)]-w[V_idx(Nx-1,n+1)])/dx,w[r_idx(Nx,n)])+eps*(w[V_idx(1,n+1)]-2*w[V_idx(Nx,n+1)]+w[V_idx(Nx-1,n+1)])
    for j in range(2,Nx):
        for n in range(0,Nt):
            # F_rho , F[Nt]->F[Nt*Nx-Nt-1] ************ 2 
            FF[(j-1)*Nt+n]=w[r_idx(j,n+1)]-0.5*(w[r_idx(j-1,n)]+w[r_idx(j+1,n)])+(0.5*dt/dx)*(w[r_idx(j+1,n)]*w[u_idx(j+1,n)]-w[r_idx(j-1,n)]*w[u_idx(j-1,n)])
            # F_u , F[Nt*Nx+Nt]->F[2*Nt*Nx-Nt-1] *********** 5 
            FF[(j-1)*Nt+Nt*Nx+n]=w[u_idx(j,n)]-f_star_p((w[V_idx(j,n+1)]-w[V_idx(j-1,n+1)])/dx,w[r_idx(j,n)])
            # F_V , F[2*Nt*Nx+Nt]->F[3*Nt*Nx-Nt-1] ********* 8 
            FF[(j-1)*Nt+2*Nt*Nx+n]=w[V_idx(j,n+1)]-w[V_idx(j,n)]+dt*f_star((w[V_idx(j,n+1)]-w[V_idx(j-1,n+1)])/dx,w[r_idx(j,n)])+eps*(w[V_idx(j+1,n+1)]-2*w[V_idx(j,n+1)]+w[V_idx(j-1,n+1)])
        # F_rho_int , F[3*Nt*Nx+1]->F[3*Nt*Nx+Nx-2] ********** 11
        FF[3*Nt*Nx+j-1]=w[r_idx(j,0)]-(1/dx)*integral(x[j-1],x[j])
        # F_V_ter , F[3*Nt*Nx+Nx+1]->F[3*Nt*Nx+2*Nx-2] ********* 14
        FF[3*Nt*Nx+Nx+j-1]=w[V_idx(j,Nt)]-VT(x[j])
    # F_rho_int , F[3*Nt*Nx] ********* 10
    FF[3*Nt*Nx]=w[r_idx(1,0)]-(1/dx)*integral(x[0],x[1])
    # F_rho_int , F[3*Nt*Nx+Nx-1] ********* 12
    FF[3*Nt*Nx+Nx-1]=w[r_idx(Nx,0)]-(1/dx)*integral(x[Nx-1],x[Nx])
    # F_V_ter , F[3*Nt*Nx+Nx] *********** 13 
    FF[3*Nt*Nx+Nx]=w[V_idx(1,Nt)]-VT(x[1])
    # F_V_ter , F[3*Nt*Nx+2*Nx-1] ************** 15
    FF[3*Nt*Nx+2*Nx-1]=w[V_idx(Nx,Nt)]-VT(x[Nx])
    
    return FF


def jacobian(w): # Ignoring the forward-backward coupling  parts
    J=np.zeros((3*Nt*Nx+2*Nx,3*Nt*Nx+2*Nx))
    for n in range(0,Nt):
        for j in range(1,Nx-1):
            J[j*Nt+n,j*(Nt+1)+n+1]=1 # F_rho - rho
            J[j*Nt+n,j*(Nt+1)+n+Nt+1]=(0.5*dt/dx)*w[u_idx(j+2,n)]-0.5 # F_rho -rho
            J[j*Nt+n,j*(Nt+1)+n-Nt-1]=-(0.5*dt/dx)*w[u_idx(j,n)]-0.5 # F_rho -rho
            J[j*Nt+n,(Nt+1)*Nx+j*Nt+n+Nt]=(0.5*dt/dx)*w[r_idx(j+2,n)] # F_rho - u
            J[j*Nt+n,(Nt+1)*Nx+j*Nt+n-Nt]=-(0.5*dt/dx)*w[r_idx(j,n)] # F_rho - u
            J[Nt*Nx+j*Nt+n,(Nt+1)*Nx+j*Nt+n]=1 # F_u - u
            J[2*Nt*Nx+j*Nt+n,(2*Nt+1)*Nx+j*(Nt+1)+n]=-1 # F_V - V
            # J[2*Nt*Nx+j*Nt+n,(2*Nt+1)*Nx+j*(Nt+1)+n+1]=1 # F_V - V
            J[2*Nt*Nx+j*Nt+n,(2*Nt+1)*Nx+j*(Nt+1)+n+1]=1-2*eps # F_V - V ....
            J[2*Nt*Nx+j*Nt+n,(2*Nt+1)*Nx+j*(Nt+1)+n+Nt+2]=eps # F_V - V ....
            J[2*Nt*Nx+j*Nt+n,(2*Nt+1)*Nx+(j-1)*(Nt+1)+n+1]=eps # F_V - V ....
            
        J[n,n+1]=1 # F_rho - rho
        J[(Nx-1)*Nt+n,(Nx-1)*(Nt+1)+n+1]=1 # F_rho - rho
        J[n,n+Nt+1]=(0.5*dt/dx)*w[u_idx(2,n)]-0.5 # F_rho - rho
        J[(Nx-1)*Nt+n,(Nx-1)*(Nt+1)+n-Nt-1]=-(0.5*dt/dx)*w[u_idx(Nx-1,n)]-0.5 # F_rho - rho
        J[n,Nx*Nt+Nx-Nt+n-1]=(0.5*dt/dx)*w[u_idx(Nx,n)]-0.5 # F_rho - rho
        J[Nx*Nt-Nt+n,n]=(0.5*dt/dx)*w[u_idx(1,n)]-0.5 # F_rho - rho
        J[n,(Nt+1)*Nx+n+Nt]=(0.5*dt/dx)*w[r_idx(2,n)] # F_rho - u
        J[(Nx-1)*Nt+n,(Nt+1)*Nx+(Nx-1)*Nt+n-Nt]=-(0.5*dt/dx)*w[r_idx(Nx-1,n)] # F_rho - u
        J[n,2*Nt*Nx+Nx-Nt+n]=-(0.5*dt/dx)*w[r_idx(Nx,n)] # F_rho - u
        J[Nt*Nx+Nx+n,n]=(0.5*dt/dx)*w[r_idx(1,n)] # F_rho - u
        J[Nt*Nx+n,(Nt+1)*Nx+n]=1 # F_u -u
        J[Nt*Nx+(Nx-1)*Nt+n,(Nt+1)*Nx+(Nx-1)*Nt+n]=1 # F_u - u
        J[2*Nt*Nx+n,(2*Nt+1)*Nx+n]=-1 # F_V - V
        # J[2*Nt*Nx+n,(2*Nt+1)*Nx+n+1]=1 # F_V - V ....
        J[2*Nt*Nx+n,(2*Nt+1)*Nx+n+1]=1-2*eps  # F_V - V
        J[2*Nt*Nx+(Nx-1)*Nt+n,(2*Nt+1)*Nx+(Nx-1)*(Nt+1)+n]=-1 # F_V - V
        # J[2*Nt*Nx+(Nx-1)*Nt+n,(2*Nt+1)*Nx+(Nx-1)*(Nt+1)+n+1]=1 # F_V - V 
        J[2*Nt*Nx+(Nx-1)*Nt+n,(2*Nt+1)*Nx+(Nx-1)*(Nt+1)+n+1]=1-2*eps # F_V - V ....
        J[2*Nt*Nx+n,(2*Nt+1)*Nx+n+Nt+2]=eps # F_V - V ....
        J[2*Nt*Nx+n,(2*Nt+1)*Nx+(Nx-1)*(Nt+1)+n+1]=eps # F_V - V ....
        J[2*Nt*Nx+(Nx-1)*Nt+n,(2*Nt+1)*Nx+(Nx-2)*(Nt+1)+n+1]=eps # F_V - V ....
        J[2*Nt*Nx+(Nx-1)*Nt+n,(2*Nt+1)*Nx+n+1]=eps # F_V - V ....
    for j in range(0,Nx):
        J[3*Nt*Nx+j,(Nt+1)*j]=1 # F_rho_int - rho
        J[3*Nt*Nx+Nx+j,(2*Nt+1)*Nx+(Nt+1)*j+Nt]=1 # F_V_ter - V
    
    return J

# import pandas as pd
def get_preconditioner(a):
    Jac=jacobian(a)
    M=np.linalg.inv(Jac)
    return M


def sol_to(n_Nx,n_Nt,sol,rho,u,V):
    for j in range(0,n_Nx):
        for n in range(0,n_Nt):
            rho[j,n]=sol[j*(n_Nt+1)+n]
            u[j,n]=sol[(n_Nt+1)*n_Nx+j*n_Nt+n]
            V[j,n]=sol[(2*n_Nt+1)*n_Nx+j*(n_Nt+1)+n]
        rho[j,n_Nt]=sol[j*(n_Nt+1)+n_Nt]
        V[j,n_Nt]=sol[(2*n_Nt+1)*n_Nx+j*(n_Nt+1)+n_Nt]
    return 0

def to_sol(n_Nx,n_Nt,sol,rho,u,V):
    for j in range(0,2*n_Nx):
        for n in range(0,2*n_Nt):
            sol[j*(2*n_Nt+1)+n]=rho[j,n]
            sol[(2*n_Nt+1)*2*n_Nx+j*2*n_Nt+n]=u[j,n]
            sol[(2*2*n_Nt+1)*2*n_Nx+j*(2*n_Nt+1)+n]=V[j,n]
        sol[j*(2*n_Nt+1)+2*n_Nt]=rho[j,2*n_Nt]
        sol[(2*2*n_Nt+1)*2*n_Nx+j*(2*n_Nt+1)+2*n_Nt]=V[j,2*n_Nt]
    return 0


import scipy.interpolate as interpolate
# from scipy.interpolate import barycentric_interpolate
# from scipy.interpolate import griddata
def interpol(n,new_n,data): # 1D interpolation
    
    """" Go from a coarse grid Nt*Nx to a finer grid spacing (2*Nt)*(2*Nx) """""
    i = np.indices(data.shape)[0]/(n-1)  # [0, ..., 1]
    new_i = np.linspace(0, 1, new_n)
    # new_data=griddata(i, data, new_i, method="cubic")  # method{‘linear’, ‘nearest’, ‘cubic’}
    # Create a linear interpolation function based on the original data
    linear_interpolation_func = interpolate.interp1d(i, data, kind='linear') # ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’.
    new_data = linear_interpolation_func(new_i)
    # new_data=barycentric_interpolate(i, data, new_i) #  polynomial interpolation

    return new_data

def multigrid(old_Nt,new_Nt,w):
    rho=np.zeros((Nx,old_Nt+1))
    u=np.zeros((Nx,old_Nt))
    V=np.zeros((Nx,old_Nt+1))
    sol_to(old_Nt,w,rho,u,V)
    new1_rho=np.zeros((2*Nx,old_Nt+1))
    new1_u=np.zeros((2*Nx,old_Nt))
    new1_V=np.zeros((2*Nx,old_Nt+1))
    for n in range(old_Nt):
        new1_rho[:,n]=interpol(Nx,2*Nx,rho[:,n])
        new1_u[:,n]=interpol(Nx,2*Nx,u[:,n])
        new1_V[:,n]=interpol(Nx,2*Nx,V[:,n])
    new1_rho[:,old_Nt]=interpol(Nx,2*Nx,rho[:,old_Nt])
    new1_V[:,old_Nt]=interpol(Nx,2*Nx,V[:,old_Nt])
    new_rho=np.zeros((2*Nx,2*new_Nt+1))
    new_u=np.zeros((2*Nx,2*new_Nt))
    new_V=np.zeros((2*Nx,2*new_Nt+1))
    for j in range(2*Nx):
        new_rho[j,:]=interpol(old_Nt+1,2*new_Nt+1,new1_rho[j,:])
        new_u[j,:]=interpol(old_Nt,2*new_Nt,new1_u[j,:])
        new_V[j,:]=interpol(old_Nt+1,2*new_Nt+1,new1_V[j,:])
        
    new_w = np.zeros(3*(2*new_Nt)*(2*Nx)+2*(2*Nx))
    to_sol(new_Nt,new_w,new_rho,new_u,new_V)
    
    return new_w

""" For MPI """

def create_2d_cart(dims,npoints): # return communicator (cart2d) with new cartesian topology
                                                                                                                                                                                                                                              
    periods = tuple([True, False]) # True : periodic, False : non-periodic Cartesian topology
    reorder = False # the rank of the processes in the new communicator (COMM_2D) is the same as in the old communicator (COMM). 
    
    if (RANK == 0):
        print("Exécution avec",SIZE," MPI processes\n"
                "Taille du domaine : nx=",npoints[0], " nt=",npoints[1],"\n"
                "Dimension pour la topologie :",dims[0]," along x", dims[1]," along t\n"
                "-----------------------------------------") 

    cart2d = COMM.Create_cart(dims = dims, periods = periods, reorder = reorder)
    
    return cart2d

nb_neighbours = 4
N = 0 # hight
E = 1 # right
S = 2 # low
W = 3 # left

def create_neighbours(cart2d): # Find processor neighbors

    neighbour = np.zeros(nb_neighbours, dtype=np.int8)
    # Outputs : rank of source, destination processes
    neighbour[W],neighbour[E] = cart2d.Shift(direction=0,disp=1) # direction 0: <->
    neighbour[S],neighbour[N] = cart2d.Shift(direction=1,disp=1) # direction 1 : |
    
    print("I am", RANK," my neighbours are : N", neighbour[N]," E",neighbour[E] ," S ",neighbour[S]," W",neighbour[W])

    return neighbour

def Coords_2D(cart2d):

    coord2d = cart2d.Get_coords(RANK)
    # print("I’m rank :",RANK," my 2d coords are",coord2d)
    
    sy = int((coord2d[1] * npoints[1]) / dims[1]) + 1
    
    sx = int((coord2d[0] * npoints[0]) / dims[0]) + 1

    ex = int(((coord2d[0] + 1) * npoints[0]) / dims[0])
    ey = int(((coord2d[1] + 1) * npoints[1]) / dims[1])

    print("Rank in the topology :",RANK," Local Grid Index :", sx, " to ",ex," along x, ",
          sy, " to", ey," along t")
    
    return coord2d, sx, ex, sy, ey


def create_derived_type(sx, ex, sy, ey):
    
    '''Creation of the type_line derived datatype to exchange points
     with northern to southern neighbours '''
    ## Row of a matrix is not contiguous in memory
    ## In N x M matrix, each element of a row is separated by N elements
    type_ligne = MPI.DOUBLE.Create_contiguous(ey-sy + 1) # count = N = ey-sy + 1
    type_ligne.Commit() # A new datatype must be committed before using it
    
    '''Creation of the type_column derived datatype to exchange points
     with western to eastern neighbours '''
    ## A vector type describes a series of blocks, all of equal size (blocklen), spaced with a constant stride.
    ## In N x M matrix, N= and M=block_count
    # block_count : The number of blocks to create.
    # blocklen : The number of (same) elements in each block
    # stride : Distance between the start of each block, expressed in number of elements.
    type_column = MPI.DOUBLE.Create_vector(ex-sx + 1, 1, ey-sy + 3) # block_count = ex-sx + 1 ; blocklen = 1 ; stride = ey-sy + 3
    type_column.Commit()

    return type_ligne, type_column

def IDX(i, j): 
        return ( ((i)-(sx-1))*(ey-sy+3) + (j)-(sy-1) )

def communications(u, sx, ex, sy, ey, type_column, type_ligne):

    #[data, count, datatype]
    ''' Envoyer au voisin N et recevoir du voisin S '''#type_ligne
    COMM.Send([u[IDX(sx, sy) : ], 1, type_ligne], dest=neighbour[N]) #IDX(sx, sy)XX
    COMM.Recv([u[IDX(ex+1, sy) : ], 1, type_ligne], source=neighbour[S]) #IDX(ex, sy+1)XX

    ''' Envoyer au voisin S et recevoir du voisin N '''#type_ligne
    COMM.Send([u[IDX(ex, sy) : ], 1, type_ligne], dest=neighbour[S]) #IDX(ex, sy)XX
    COMM.Recv([u[IDX(sx-1, sy) : ], 1, type_ligne], source=neighbour[N]) #IDX(sx, sy-1)XX

    ''' Envoyer au voisin W et recevoir du voisin E '''#type_column
    COMM.Send([u[IDX(sx, sy) : ], 1, type_column], dest=neighbour[W]) #IDX(sx, sy)XX
    COMM.Recv([u[IDX(sx, ey+1) : ], 1, type_column], source=neighbour[E]) #IDX(sx+1, ey)XX

    ''' Envoyer au voisin E et recevoir du voisin W ''' #type_column
    COMM.Send([u[IDX(sx, ey) : ], 1, type_column], dest=neighbour[E]) #IDX(sx, ey)XX
    COMM.Recv([u[IDX(sx, sy-1) : ], 1, type_column], source=neighbour[W]) #IDX(sx-1, sy)XX

    

def initialization(rho0, u0, V0, sx, ex, sy, ey):

    ''' Espacement de la grille dans chaque dimension'''
    
    SIZE = (ex-sx+3) * (ey-sy+3)
    
    rho       = np.zeros(SIZE)
    u       = np.zeros(SIZE)
    V   = np.zeros(SIZE)
    
    '''Initialisation de rho, u et V '''
    for i in range(sx, ex+1): # x axis
        for j in range(sy, ey+1): # y axis
        
            rho[IDX(i, j)] = rho0[i-1,j-1]
            u[IDX(i, j)] = u0[i-1,j-1]
            V[IDX(i, j)] = V0[i-1,j-1]
                   
    # print("I’m rank :",RANK, "sx",sx,"ex",ex, "sy",sy, "ey",ey," new_u",u)
            
    return rho, u, V

''' Calcul de l’erreur globale (maximum des erreurs locales)'''
def global_error(u, u_new): 
   
    local_error = 0
     
    for i in range(sx, ex+1, 1):
        for y in range(sy, ey+1, 1):
            temp = np.fabs( u[IDX(i, y)] - u_new[IDX(i, y)]  )
            if local_error < temp:
                local_error = temp;
    
    return local_error
        

""" Solve in finer grid """

guess0 = np.zeros(3*Nt*Nx+2*Nx)
# guess0=np.loadtxt('Sol0_LWR_T3_N1.dat')
rho0=np.zeros((Nx,Nt+1))
u0=np.zeros((Nx,Nt+1))
V0=np.zeros((Nx,Nt+1))
sol_to(Nt,guess0,rho0,u0,V0)
# if RANK==0:
# print('rho0=',rho0)
# print('u0=',u0)
# print('V0=',V0)

cart2D=create_2d_cart()
neighbour = create_neighbours(cart2D)
coord2D, sx, ex, sy, ey = Coords_2D(cart2D)
type_ligne, type_column = create_derived_type(sx, ex, sy, ey)
rho, u, V = initialization(rho0, u0, V0, sx, ex, sy, ey)
rho_new = rho.copy() 
u_new = u.copy() 
V_new = V.copy()

''' Stepping time '''
it = 0
convergence = False
it_max = 100000
epsilon = 2.e-16

''' spend time '''
t1 = MPI.Wtime()
while (not(convergence) and (it < it_max)):
    it = it+1;

    rho_temp = rho.copy() 
    u_temp = u.copy() 
    V_temp = V.copy() 
    rho = rho_new.copy()
    u = u_new.copy() 
    V = V_new.copy() 
    rho_new = rho_temp.copy()
    u_new = u_temp.copy()
    V_new = V_temp.copy()
    
    ''' Échange des interfaces à la n itération '''
    communications(rho, sx, ex, sy, ey, type_column, type_ligne)
    communications(u, sx, ex, sy, ey, type_column, type_ligne)
    communications(V, sx, ex, sy, ey, type_column, type_ligne)
 
    '''Calcul de rho_new, u_new et V_new à l’itération n 1 '''
    nbx=ex-sx+1
    nby=ey-sy+1
    guess=np.zeros(3*nbx*nby+2*nbx)
    to_sol(nbx,nby,guess,rho,u,V)
    
    t0 = time.process_time()   ###
    prec=get_preconditioner(guess)
    t1 = time.process_time()   ###
    print("Time spent (anal_precond) :",t1-t0)
    
    t0 = time.process_time()   ###
    # sol0 = newton_gmres(guess0, F, prec, 0.001, 1e-6)
    sol= newton_krylov(F, guess, method='gmres', verbose=1, inner_M=prec)
    # print('sol=',sol)
    t1 = time.process_time()   ###
    print("Time spent (gmres) :",t1-t0)
    
    sol_to(nbx,nby,sol,rho_new,u_new,V_new)
    
    ''' Computation of the global error '''
    r_local_error = global_error(rho, rho_new);
    r_diffnorm = COMM.allreduce(np.array(r_local_error), op=MPI.MAX )  
    u_local_error = global_error(u, u_new);
    u_diffnorm = COMM.allreduce(np.array(u_local_error), op=MPI.MAX )
    V_local_error = global_error(V, V_new);
    V_diffnorm = COMM.allreduce(np.array(V_local_error), op=MPI.MAX )
   
    ''' Stop si nous avons obtenu la précision de la machine '''
    convergence = (r_diffnorm < epsilon and u_diffnorm < epsilon and V_diffnorm < epsilon)
    
    ''' Print diffnorm poue le processus 0 '''
    if ((RANK == 0) and ((it % 100) == 0)):
        print("Iteration", it, " global_error = ", max(r_diffnorm,u_diffnorm,V_diffnorm));
    
        
''' temps écoulé '''
t2 = MPI.Wtime()

if (RANK == 0):
    ''' Print temp de convergence pour le processus 0 '''
    print("convergence après:",it, 'l iterations in', t2-t1,'secs')

    '''Comparer avec la solution exacte du processus 0 '''
    # results(u, u_exact)
    # plot_2d(u)


np.savetxt('Sol0_LWR_T3_N1.dat', sol)

""" solve in finer grid 1 """
# Nx=15; Nt=60
# sol0=np.loadtxt('Sol0_LWR_T3_N1.dat')
Nxx=2*Nx; Ntt=2*Nt
dx=L/Nxx # spatial step size
if mu==0.0:
    dt=min(T/Ntt,CFL*dx/abs(u_max)) # temporal step size
    eps=0.0
else:
    dt=min(T/Ntt,CFL*dx/abs(u_max),EPS*(dx**2)/mu) # temporal step size
    eps=mu*dt/(dx**2) # V
x=np.linspace(0,L,Nxx+1)
t=np.arange(0,T+dt,dt)
Nt=int((len(t)-1)/2)
guess1=multigrid(int(Ntt/2),Nt,sol)
np.savetxt('Guess1_LWR_T3_N1.dat', guess1)




















# """ solve in finer grid 1 """
# # Nx=15; Nt=60
# # sol0=np.loadtxt('Sol0_LWR_T3_N1.dat')
# Nxx=2*Nx; Ntt=2*Nt
# dx=L/Nxx # spatial step size
# if mu==0.0:
#     dt=min(T/Ntt,CFL*dx/abs(u_max)) # temporal step size
#     eps=0.0
# else:
#     dt=min(T/Ntt,CFL*dx/abs(u_max),EPS*(dx**2)/mu) # temporal step size
#     eps=mu*dt/(dx**2) # V
# x=np.linspace(0,L,Nxx+1)
# t=np.arange(0,T+dt,dt)
# Nt=int((len(t)-1)/2)
# guess1=multigrid(int(Ntt/2),Nt,sol0)
# np.savetxt('Guess1_LWR_T3_N1.dat', guess1)
# Nx=2*Nx; Nt=2*Nt
# print('Nx={Nx}, Nt={Nt}'.format(Nx=Nx,Nt=Nt))
# print('dx={dx}, dt={dt}'.format(dx=round(dx,3),dt=round(dt,3)))
# t0 = time.process_time()   ###
# # guess= np.zeros(3*Nt*Nx+2*Nx)
# prec=get_preconditioner(guess1)
# t1 = time.process_time()   ###
# print("Time spent (jax_precond) :",t1-t0)
# t0 = time.process_time()   ###
# sol1 = newton_krylov(F, guess1, method='gmres', verbose=1, inner_M=prec)
# t1 = time.process_time()   ###
# print("Time spent (gmres) :",t1-t0)
# np.savetxt('Sol1_LWR_T3_N1.dat', sol1)
# # print('sol1=',sol1)
# Nx_list.append(Nx)
# Nt_list.append(Nt)

# """ solve in finer grid 2 """
# # Nx=30; Nt=180
# # sol1=np.loadtxt('Sol1_LWR_T3_N1.dat')
# Nxx=2*Nx; Ntt=2*Nt
# dx=L/Nxx # spatial step size
# dx=L/Nxx # spatial step size
# if mu==0.0:
#     dt=min(T/Ntt,CFL*dx/abs(u_max)) # temporal step size
#     eps=0.0
# else:
#     dt=min(T/Ntt,CFL*dx/abs(u_max),EPS*(dx**2)/mu) # temporal step size
#     eps=mu*dt/(dx**2) # V
# x=np.linspace(0,L,Nxx+1)
# t=np.arange(0,T+dt,dt)
# Nt=int((len(t)-1)/2)
# guess2=multigrid(int(Ntt/2),Nt,sol1)
# np.savetxt('Guess2_LWR_T3_N1.dat', guess2)
# Nx=2*Nx; Nt=2*Nt
# print('Nx={Nx}, Nt={Nt}'.format(Nx=Nx,Nt=Nt))
# print('dx={dx}, dt={dt}'.format(dx=round(dx,3),dt=round(dt,3)))
# t0 = time.process_time()   ###
# prec=get_preconditioner(guess2)
# t1 = time.process_time()   ###
# print("Time spent (jax_precond) :",t1-t0)
# t0 = time.process_time()   ###
# sol2 = newton_krylov(F, guess2, method='gmres', verbose=1, inner_M=prec)
# t1 = time.process_time()   ###
# print("Time spent (gmres) :",t1-t0)
# np.savetxt('Sol2_LWR_T3_N1.dat', sol2)
# # print('sol2=',sol2)
# Nx_list.append(Nx)
# Nt_list.append(Nt)

# """ solve in finer grid 3 """
# # Nx=60; Nt=720
# # sol2=np.loadtxt('Sol2_LWR_T3_N1.dat')
# Nxx=2*Nx; Ntt=2*Nt
# dx=L/Nxx # spatial step size
# dx=L/Nxx # spatial step size
# if mu==0.0:
#     dt=min(T/Ntt,CFL*dx/abs(u_max)) # temporal step size
#     eps=0.0
# else:
#     dt=min(T/Ntt,CFL*dx/abs(u_max),EPS*(dx**2)/mu) # temporal step size
#     eps=mu*dt/(dx**2) # V
# x=np.linspace(0,L,Nxx+1)
# t=np.arange(0,T+dt,dt)
# Nt=int((len(t)-1)/2)
# guess3=multigrid(int(Ntt/2),Nt,sol2)
# np.savetxt('Guess3_LWRL_T3_N1.dat', guess3)
# Nx=2*Nx; Nt=2*Nt
# print('Nx={Nx}, Nt={Nt}'.format(Nx=Nx,Nt=Nt))
# print('dx={dx}, dt={dt}'.format(dx=round(dx,3),dt=round(dt,3)))
# t0 = time.process_time()   ###
# prec=get_preconditioner(guess3)
# t1 = time.process_time()   ###
# print("Time spent (jax_precond) :",t1-t0)
# t0 = time.process_time()   ###
# sol3 = newton_krylov(F, guess3, method='gmres', verbose=1, inner_M=prec)
# t1 = time.process_time()   ###
# print("Time spent (gmres) :",t1-t0)
# np.savetxt('Sol3_LWR_T3_N1.dat', sol3)
# # # print('sol3=',sol3)
# Nx_list.append(Nx)
# Nt_list.append(Nt)

# # np.savetxt('Nx_Nt_LWR_T3_N1.dat', [Nx_list,Nt_list])




