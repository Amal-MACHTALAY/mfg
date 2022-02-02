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
Nx=2; Nt=5; use_interp = 0 # spatial-temporal grid sizes, use interpolation

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
Ntloc = da.getRanges()[0][1] - da.getRanges()[0][0] + 1
Nxloc = da.getRanges()[1][1] - da.getRanges()[1][0] + 1 

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
    
    sendcounts = np.array(COMM.allgather(len(w.array)))
    
    ww = np.empty(sum(sendcounts), dtype=np.double)
    
    COMM.Allgatherv(sendbuf=w.array, recvbuf=(ww, sendcounts))  
    
    # if RANK == 0:
    #     for i in range(len(ww)):
    #         print("ww =", "{:.2f}".format(ww[i]), i)
    
    # print("len",len(row))
    compute_jacobian(ww, row, col, data, Nt, Nx, dt, dx, eps, np.array(da.ranges))
    
    P.setType("mpiaij")
    P.setFromOptions()
    P.setPreallocationNNZ(20)
    P.setOption(option=19, flag=0)
    
    for i in range(len(data)):
        P.setValues(row[i], col[i], data[i], addv=False)
        
    P.assemble()
    
    if J != P:
        J.assemble()
        
    # print(w.view())
    # print(P.view())
    # import sys; sys.exit()
            
    return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

def my_sum(a, b, mpi_datatype):
    # a = np.frombuffer(aa, dtype=np.double)
    # b = np.frombuffer(bb, dtype=np.double)
    
    # bb = max(abs(a), abs(b))
    for i in range(len(a)):
        # b[i] = max(abs(a[i]), abs(b[i]))
        if abs(a[i]) > abs(b[i]):
            b[i] = a[i]
        
        elif abs(a[i]) < abs(b[i]):
            b[i] = b[i]
            
        else:
            b[i] = b[i]
            
my_op = MPI.Op.Create(my_sum)
# """************************ solve in grid 1***************************** """
def formFunction(snes, w, F, Nt, Nx, dt, dx, eps, u_max, rho_jam, x):
    
    # FF = F.getArray()
    # ww = w.getArray()
    
    
    sendcounts = np.array(COMM.allgather(len(w.array)))
    
    ww = np.empty(sum(sendcounts), dtype=np.double)
    FF = np.empty(sum(sendcounts), dtype=np.double)
    
    COMM.Allgatherv(sendbuf=w.array, recvbuf=(ww, sendcounts))  
    COMM.Allgatherv(sendbuf=F.array, recvbuf=(FF, sendcounts))  
    
    # if RANK == 0:
    #     recvbuf1 = np.zeros(sum(sendcounts), dtype=np.double)
    #     recvbuf2 = np.zeros(sum(sendcounts), dtype=np.double)
    # else:
    #     recvbuf1 = None
    #     recvbuf2 = None
        
    
    # COMM.Gatherv(sendbuf=ww, recvbuf=(recvbuf1, sendcounts), root=0)  
    # COMM.Gatherv(sendbuf=FF, recvbuf=(recvbuf2, sendcounts), root=0)

    # wwGlobal = COMM.bcast(recvbuf1, root=0)
    # FFGlobal = COMM.bcast(recvbuf2, root=0)
    
    
    # if RANK == 0:
    #     print(ww)
    
    
    compute_FF(ww, FF, Nt, Nx, dt, dx, eps, u_max, rho_jam, x, np.array(da.ranges), RANK)
    # print("ff =", "{:.2f}".format(FF[0]), 0, RANK)
    # if RANK == 0:
    #     for i in range(len(FF)):
    #         if FF[i] != 0:
    #             print(FF[i], i, RANK)
    
    totals = np.zeros_like(FF)
    
    # use MPI to get the totals 
    COMM.Allreduce(
        [FF, MPI.DOUBLE],
        [totals, MPI.DOUBLE],
        op = my_op,
    )
    FF = totals
    
    # if RANK == 0:
    #     for i in range(len(FF)):
    #         print("ff =", "{:.2f}".format(FF[i]), i)
   
    
    # # if RANK == 0:
    # for i in range(len(FFGlobal)):
    #     if FFGlobal[i] != 0:
    #         print(i, FFGlobal[i])
    
    # print("\n")
    # totals = np.zeros_like(ww)
    
    # # use MPI to get the totals 
    # COMM.Allreduce(
    #     [ww, MPI.DOUBLE],
    #     [totals, MPI.DOUBLE],
    #     op = my_op,
    # )
    # ww = totals
    
    FFlocal = F.getArray()
    COMM.Scatterv([FF, sendcounts, MPI.DOUBLE], FFlocal, root = 0)
    
    
    
    # wwlocal = w.getArray()
    # COMM.Scatterv([ww, sendcounts, MPI.DOUBLE], wwlocal, root = 0)
    
    # FF = FFlocal
    # ww  = wwlocal 
    

    
    # print(w.view())
    # print(F.view())
    
    
    # import sys; sys.exit()
    
F = daa.createGlobalVector()
# F = PETSc.Vec()
# F.create()
# F.setSizes(3*Nt*Nx+2*Nx)
# F.setFromOptions()

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

# print(snes.view())

print(xx.view())
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