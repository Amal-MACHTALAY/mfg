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
# from psydac.ddm.partition import mpi_compute_dims

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
Nx=3; Nt=5; use_interp = 0 # spatial-temporal grid sizes, use interpolation

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

shap=(Nt, Nx)

from petsc4py import PETSc

t0 = time.process_time()   ###

# create nonlinear solver
snes = PETSc.SNES()
snes.create()
da = PETSc.DMDA().create(dim = 2,
                         boundary_type=(PETSc.DMDA.BoundaryType.NONE,),
                         sizes = (shap), dof = 1, stencil_width = 0)

da.setFromOptions()
da.setUp()
Ntloc = da.getRanges()[0][1] - da.getRanges()[0][0]
Nxloc = da.getRanges()[1][1] - da.getRanges()[1][0]

daa = PETSc.DMDA().create(dim = 1,
                         boundary_type=(PETSc.DMDA.BoundaryType.NONE,),
                         sizes = (3*Nt*Nx+2*Nx,), dof = 1, stencil_width = 0,
                         setup=False)

daa.setFromOptions()
daa.setUp()
snes.setDM(daa)



row = np.zeros(10*Ntloc*Nxloc+2*Nxloc, dtype=np.int64); col = np.zeros(10*Ntloc*Nxloc+2*Nxloc, dtype=np.int64); 
data = np.zeros(10*Ntloc*Nxloc+2*Nxloc);

def formInitguess(snes, w):
    
    w.array = initialguess(Nt, Nx, multip)

def formJacobian(snes, w, J, P):
    
    ww = w.array    
    sendcounts = np.array(COMM.allgather(len(ww)))
    
    if RANK == 0:
        recvbuf1 = np.empty(sum(sendcounts), dtype=np.double)
    else:
        recvbuf1 = None
    
    COMM.Gatherv(sendbuf=ww, recvbuf=(recvbuf1, sendcounts), root=0)  
    
    ww = COMM.bcast(recvbuf1, root=0)
   
    compute_jacobian(ww, row, col, data, Nt, Nx, dt, dx, eps, np.array(da.ranges))
    
    P.setType("mpiaij")
    P.setFromOptions()
    P.setPreallocationNNZ(10)
    P.setOption(option=19, flag=0)
    
    for i in range(len(data)):
        P.setValues(row[i], col[i], data[i], addv=False)
        
    P.assemble()
    
    if J != P:
        J.assemble()
        
    # print(P.view())
    # import sys; sys.exit()
            
    return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

# """************************ solve in grid 1***************************** """
def formFunction(snes, w, F, Nt, Nx, dt, dx, eps, u_max, rho_jam, x):
    
    FF = F.getArray()
    ww = w.getArray()
    
    sendcounts = np.array(COMM.allgather(len(ww)))
    
    if RANK == 0:
        recvbuf1 = np.empty(sum(sendcounts), dtype=np.double)
        recvbuf2 = np.empty(sum(sendcounts), dtype=np.double)
    else:
        recvbuf1 = None
        recvbuf2 = None
        
    
    COMM.Gatherv(sendbuf=ww, recvbuf=(recvbuf1, sendcounts), root=0)  
    COMM.Gatherv(sendbuf=FF, recvbuf=(recvbuf2, sendcounts), root=0)

    # COMM.gatherv([ww, sendcounts, MPI.DOUBLE], recvbuf1, root = 0)
    # COMM.gatherv([FF, sendcounts, MPI.DOUBLE], recvbuf2, root = 0)
    ww = COMM.bcast(recvbuf1, root=0)
    FF = COMM.bcast(recvbuf2, root=0)
    
    # xlocal = daa.getLocalVec()
    # daa.localToGlobal(w, xlocal)
    
    # flocal = daa.getLocalVec()
    # daa.localToGlobal(F, flocal)
    # # print(F.getSize())
    # # print(w.getSize())
    
    # xx = daa.getVecArray(xlocal)
    # ff = daa.getVecArray(flocal)
    
    # print(xx.)
    # ww = daa.getLocalVec()
    # # ww = daa.getGlobalVec()
    
    # # print(ww.sizes, w.sizes)
    # print(len(w))
    
    # print(da.ranges, da.getRanges())
    compute_FF(ww, FF, Nt, Nx, dt, dx, eps, u_max, rho_jam, x, np.array(da.ranges), RANK)
    # FFlocal = da.createLocalVec()
    # daa.globalToLocal(F, FFlocal)
    
    
    # FF = FFlocal
    # # FF = COMM.Allreduce(FF)
    # the 'totals' array will hold the sum of each 'data' array
    totals = np.zeros_like(FF)
    
    # # use MPI to get the totals 
    COMM.Allreduce(
        [FF, MPI.DOUBLE],
        [totals, MPI.DOUBLE],
        op = MPI.MIN,
    )
    # # # rhs0 = COMM.allreduce(FF, op=MPI.SUM)

    FF = totals
    FFlocal = F.getArray()
    COMM.Scatterv([FF, sendcounts, MPI.DOUBLE], FFlocal, root = 0)
    
    wwlocal = w.getArray()
    COMM.Scatterv([ww, sendcounts, MPI.DOUBLE], wwlocal, root = 0)
    
    # FF = FFlocal
    # w  = wwlocal 
    # if RANK == 0:
    # for i in range(len(FF)):
    #     if FF[i] !=0:
    #         print(i, FF[i], RANK)
    
    # print(F.view())
    # print(FFlocal)
    # print(xx.array)
    # import sys; sys.exit()
    
    # return FF
    
    
F = daa.createGlobalVector()

args = [Nt, Nx, dt, dx, eps, u_max, rho_jam, x]
snes.setFunction(formFunction, F, args)
snes.setJacobian(formJacobian)#, mat, mat)

b = None
xx = daa.createGlobalVector()

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

snes.solve(b, xx)
t1 = time.process_time()   ###
time2=t1-t0


if RANK == 0:
    print("Time spent:",time2)
    its = snes.getIterationNumber()
    lits = snes.getLinearSolveIterations()
    
    print ("Number of SNES iterations = :", its)
    print ("Number of Linear iterations =" , lits)
    
# print(xx.view())

# litspit = lits/float(its)
# print ("Average Linear its / SNES = %e", float(litspit))


# if not use_interp:
#     import os
#     filename = ("sol.dat")
#     if os.path.exists(filename):
#         os.remove(filename)
    
#     with open(filename, "a") as text_file:
#         text_file.write(str(Nx))
#         text_file.write("\n")
#         text_file.write(str(Nt))
#         text_file.write("\n")
#         np.savetxt(text_file, xx.array)


# #Free petsc elements
# xx.destroy()      
# F.destroy()                                     
# snes.destroy()
# ksp.destroy()