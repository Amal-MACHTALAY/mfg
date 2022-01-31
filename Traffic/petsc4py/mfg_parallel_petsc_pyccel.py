#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 00:13:39 2022

@author: kissami
"""

##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 15:43:29 2021
@author: amal
"""

import numpy as np
import time
from modules import (compute_jacobian, compute_FF)
from tools import initialguess
from psydac.ddm.partition import mpi_compute_dims

# import time
from mpi4py import MPI

COMM = MPI.COMM_WORLD  # The default communicator
SIZE = COMM.Get_size() # The number of processes
RANK = COMM.Get_rank() # The rank of processes

'''************************ inputs ************************************'''
T=3.0 # horizon length  
u_max=1.0 # free flow speed
rho_jam=1.0 # jam density
L=1 # road length
CFL=0.75    # CFL<1
rho_a=0.05; rho_b=0.95; gama=0.1 
mu=0.0 # viscosity coefficient 
EPS=0.45

####################### grid's inputs
multip=6 # mutiple for interpolation
tol = 1e-6
Nx=30; Nt=60; use_interp = 0 # spatial-temporal grid sizes, use interpolation
if use_interp :
    Nx=Nx*multip; Nt=Nt*multip
dx=L/Nx # spatial step size
if mu==0.0:
    dt=min(T/Nt,(CFL*dx)/u_max) # temporal step size
    eps=0.0
else:
    dt=min(T/Nt,CFL*dx/abs(u_max),EPS*(dx**2)/mu) # temporal step size
    eps=mu*dt/(dx**2) 
x=np.linspace(0,L,Nx+1)
t=np.arange(0,T+dt,dt)
Nt=len(t)-1

def formFunction(snes, w, F, Nt, Nx, dt, dx, eps, u_max, rho_jam, x):
    
    FF = F.array
    w = w.array
    
    compute_FF(w, FF, Nt, Nx, dt, dx, eps, u_max, rho_jam, x)
    

row = np.zeros(10*Nt*Nx+2*Nx, dtype=np.int64); col = np.zeros(10*Nt*Nx+2*Nx, dtype=np.int64); 
data = np.zeros(10*Nt*Nx+2*Nx);

def formJacobian(snes, w, J, P):
    P.zeroEntries()
    
    compute_jacobian(w.array, row, col, data, Nt, Nx, dt, dx, eps)
    
    P.setType("mpiaij")
    P.setFromOptions()
    P.setPreallocationNNZ(10)
    P.setOption(option=19, flag=0)
    
    for i in range(len(data)):
        P.setValues(row[i], col[i], data[i], addv=False)
    
    P.assemble()
    
    if J != P:
        J.assemble()
            
    return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

def formInitguess(snes, w):
    
    print(w.array)
    w.array = initialguess(Nt, Nx, multip)
    
    print(w.array)

############################MPI###############################################
""" for MPI : Creates a division of processors in a cartesian grid """
# nbr_x=Nx+1; nbr_t=Nt+1 # spatial-temporal grid sizes 
# nx=nbr_x-1; nt=nbr_t # number of points for MPI
# px=int(np.sqrt(SIZE))-1 # number of processes on each line-1
# pt=px # number of processes on each column-1
# # print("px={px}, pt={pt}".format(px=px, pt=pt))
# new_size=(px+1)*(pt+1) # the Number of processes to decompose 
# # print('new_size=',new_size)
# nbrx=int(nx/(px+1)) #number of points for px (except root)
# nbrt=int(nt/(pt+1)) #number of points for pt (except root)
# dims=[px+1,pt+1] # The array containing the number of processes to assign to each dimension
# # print('dims=',dims)
npoints=[Nx,Nt]
# print('npoints=',npoints)

nb_neighbours = 4
N = 0; E = 1; S = 2; W = 3

neighbour = np.zeros(nb_neighbours, dtype=np.int8)

p1 = [2,2]
P1 = [False, False]
reorder = True

def create_2d_cart(npoints, p1, P1, reorder):
    
    # Store input arguments                                                                                                                                                                                                                                               
    npts    = tuple(npoints)
    pads    = tuple(p1)
    periods = tuple(P1)
    reorder = reorder
    
    nprocs, block_shape = mpi_compute_dims(SIZE, npts, pads )
    
    dims = nprocs
    
    if (RANK == 0):
        print("Execution poisson with",SIZE," MPI processes\n"
               "Size of the domain : ntx=",npoints[0], " nty=",npoints[1],"\n"
               "Dimension for the topology :",dims[0]," along x", dims[1]," along y\n"
               "-----------------------------------------")  
    
    cart2d = COMM.Create_cart(
            dims    = dims,
            periods = periods,
            reorder = reorder
            )
    
    return dims, cart2d

def create_2dCoords(cart2d, npoints, dims):

    coord2d = cart2d.Get_coords(RANK)
    
    ''' Computation of the local grid boundary coordinates (global indexes) '''
    sx = int((coord2d[0]*npoints[0])/dims[0]+1);
    ex = int(((coord2d[0]+1)*npoints[0])/dims[0]);
    
    sy = int((coord2d[1]*npoints[1])/dims[1]+1);
    ey = int(((coord2d[1]+1)*npoints[1])/dims[1]);

    print("Rank in the topology :",RANK," Local Grid Index :", sx, " to ", \
          ex," along x, ", sy, " to", ey," along y")
    
    return sx, ex, sy, ey

def create_neighbours(cart2d):

    neighbour[N],neighbour[S] = cart2d.Shift(direction=0,disp=1)
    neighbour[W],neighbour[E] = cart2d.Shift(direction=1,disp=1)
    
    print("Process", RANK," neighbour: N", neighbour[N]," E",neighbour[E] ,\
          " S ",neighbour[S]," W",neighbour[W])
    
    return neighbour

def create_derived_type(sx, ex, sy, ey):
    type_ligne = MPI.DOUBLE.Create_contiguous(ey-sy+1)
    type_ligne.Commit()
    
    type_column = MPI.DOUBLE.Create_vector(ex-sx+1, 1, ey-sy+3)
    type_column.Commit()
     
    return type_ligne, type_column

def IDX(i, j): 
    return ( ((i)-(sx-1))*(ey-sy+3) + (j)-(sy-1) )

def communications(u, sx, ex, sy, ey, type_column, type_ligne):
    
    COMM.Send([u[IDX(sx, sy):],1,type_ligne], dest=neighbour[N])
    COMM.Recv([u[IDX(ex+1, sy):],1,type_ligne],source=neighbour[S])
    
    COMM.Send([u[IDX(ex, sy):],1,type_ligne], dest=neighbour[S])
    COMM.Recv([u[IDX(sx-1, sy):],1,type_ligne],source=neighbour[N])
    
    COMM.Send([u[IDX(sx, sy):],1,type_column], dest=neighbour[W])
    COMM.Recv([u[IDX(sx, ey+1):],1,type_column],source=neighbour[E])
    
    COMM.Send([u[IDX(sx, ey):],1,type_column], dest=neighbour[E])
    COMM.Recv([u[IDX(sx, sy-1):],1,type_column],source=neighbour[W])

dims, cart2d   = create_2d_cart(npoints, p1, P1, reorder)
neighbour      = create_neighbours(cart2d)
coord2d = cart2d.Get_coords(RANK)
sx, ex, sy, ey = create_2dCoords(cart2d, npoints, dims)


# global_shap=(3*Nt*Nx+2*Nx,3*Nt*Nx+2*Nx)

Nx = ex-sx+1
Nt = ey-sy+1

local_shap=(3*Nt*Nx+2*Nx,3*Nt*Nx+2*Nx)


import sys; sys.exit()

# """************************ solve in grid 1***************************** """
from petsc4py import PETSc

t0 = time.process_time()   ###
# shap=(3*Nt*Nx+2*Nx,3*Nt*Nx+2*Nx)

# create nonlinear solver
snes = PETSc.SNES()
snes.create()
da = PETSc.DMDA().create(dim = 1,
                         boundary_type=(PETSc.DMDA.BoundaryType.NONE,),
                         sizes = (local_shap[0],), dof = 1, stencil_width = 1)

da.setFromOptions()
da.setUp()
snes.setDM(da)

F = da.createLocalVec()
F.setFromOptions()

# print(len(F.array), local_shap[0])

b = None
xx = da.createGlobalVector()

args = [Nt, Nx, dt, dx, eps, u_max, rho_jam, x]
snes.setFunction(formFunction, F, args)
snes.setJacobian(formJacobian)


print(F.getSize())
# if use_interp:
#     #snes.setInitialGuess(formInitguess)
#     X = initialguess(Nt, Nx, multip)#snes.getInitialGuess()[0](snes, xx)
#     xx.setArray(X)


snes.getKSP().setType('fgmres')

ksp = snes.getKSP()
pc = ksp.getPC()
pc.setFactorSolverType("mumps")
opts = PETSc.Options()
opts["ksp_rtol"] = tol
opts["pc_type"] = "lu"
ksp.setInitialGuessNonzero(True)
ksp.setFromOptions()


snes.setTolerances(rtol = tol)
snes.setFromOptions()


# t0 = time.process_time()   ###
snes.solve(b, xx)
t1 = time.process_time()   ###
time2=t1-t0
print("Time spent:",time2)


its = snes.getIterationNumber()
lits = snes.getLinearSolveIterations()

print ("Number of SNES iterations = :", its)
print ("Number of Linear iterations =" , lits)

# litspit = lits/float(its)
# print ("Average Linear its / SNES = %e", float(litspit))


if not use_interp:
    import os
    filename = ("sol.dat")
    if os.path.exists(filename):
        os.remove(filename)
    
    with open(filename, "a") as text_file:
        text_file.write(str(Nx))
        text_file.write("\n")
        text_file.write(str(Nt))
        text_file.write("\n")
        np.savetxt(text_file, xx.array)


# #Free petsc elements
# xx.destroy()      
# F.destroy()                                     
# snes.destroy()
# ksp.destroy()