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

# -ksp_type gmres -pc_type lu -ksp_monitor -snes_converged_reason -ksp_converged_reason
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
Nx=15; Nt=60; use_interp = 0 # spatial-temporal grid sizes, use interpolation

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

if RANK == 0:
    print('Nx={Nx}, Nt={Nt}'.format(Nx=Nx,Nt=Nt))
    print('dx={dx}, dt={dt}'.format(dx=round(dx,4),dt=round(dt,4)))

shap=(Nt, Nx)

import petsc4py
import sys
petsc4py.init(sys.argv)
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
data = np.zeros(10*Ntloc*Nxloc+2*Nxloc, np.double);

def formInitguess(snes, w):
    
    w.array = initialguess(Nt, Nx, multip)

def formJacobian(snes, w, J, P, sendcounts, ww):
    # ts = time.process_time()###
    J.zeroEntries()
    
    COMM.Allgatherv(sendbuf=w.array, recvbuf=(ww, sendcounts))  
    
    compute_jacobian(ww, row, col, data, Nt, Nx, dt, dx, eps, np.array(da.ranges))
    
    for i in range(len(data)):
        J.setValues(row[i], col[i], data[i], addv=False)
    
    J.assemblyBegin(P.AssemblyType.FINAL)
    J.assemblyEnd(P.AssemblyType.FINAL)

    
    if P != J:
        P.assemblyBegin(P.AssemblyType.FINAL)
        P.assemblyEnd(P.AssemblyType.FINAL)
    
    # te = time.process_time()###
    
    # times=te-ts
    
    # print(times, "Jacobian")

    return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

def my_sum(aa, bb, mpi_datatype):
    a = np.frombuffer(aa, dtype=np.double)
    b = np.frombuffer(bb, dtype=np.double)
    
    
    for i in range(len(a)):
        if abs(a[i]) > abs(b[i]):
            b[i] = a[i]
            
my_op = MPI.Op.Create(my_sum)
# """************************ solve in grid 1***************************** """
def formFunction(snes, w, F, Nt, Nx, dt, dx, eps, u_max, rho_jam, x, sendcounts):
    
    ww = np.empty(sum(sendcounts), dtype=np.double)
    FF = np.empty(sum(sendcounts), dtype=np.double)
    
    COMM.Allgatherv(sendbuf=w.array, recvbuf=(ww, sendcounts))  

    compute_FF(ww, FF, Nt, Nx, dt, dx, eps, u_max, rho_jam, x, np.array(da.ranges), RANK)
    
    totals = np.empty(sum(sendcounts), dtype=np.double)
    
    # ts = time.process_time()###
    
    # use MPI to get the totals 
    req = COMM.Iallreduce(
            [FF, MPI.DOUBLE],
            [totals, MPI.DOUBLE],
            op = my_op,
            )
    req.wait()
    FF = totals
    
    # te = time.process_time()###
    
    # times=te-ts
    
    # print(times, daa.getOwnershipRanges())

    COMM.Scatterv([FF, sendcounts, MPI.DOUBLE], F.array, root = 0)
    
    
F = daa.createGlobalVector()
sendcounts = np.array(COMM.allgather(len(F.array)))
ww = np.empty(sum(sendcounts), dtype=np.double)

args = [Nt, Nx, dt, dx, eps, u_max, rho_jam, x, sendcounts]
snes.setFunction(formFunction, F, args)

mat = PETSc.Mat().create()
mat.setSizes((3*Nt*Nx+2*Nx,3*Nt*Nx+2*Nx))
mat.setType("mpiaij")
mat.setFromOptions()
mat.setPreallocationNNZ(100)

args = [sendcounts, ww]
snes.setJacobian(formJacobian, mat, mat, args)
# # snes.setUseMF(True)
b = None
xx = daa.createGlobalVector()


ksp = snes.getKSP()
ksp = PETSc.KSP().create()
pc = ksp.getPC()
pc.setFactorSolverType("mumps")
opts = PETSc.Options()
ksp.setTolerances(rtol=tol)
opts["pc_type"] = "lu"
opts["mat_mumps_icntl_23"] = 1000
ksp.setInitialGuessNonzero(True)
ksp.setFromOptions()


# snes.setUseMF(True)
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
    
print(xx.view())

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