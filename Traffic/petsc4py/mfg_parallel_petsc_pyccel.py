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
# mpirun -n 2 python mfg_parallel_petsc_pyccel.py -ksp_rmonitor -snes_converged_reason -ksp_converged_reason 
# -snes_monitor -snes_linesearch_monitor -pc_factor_mat_solver_type superlu_dist -snes_lag_jacobian -2 -snes_lag_preconditioner -2 

# mpirun -n 2 python mfg_parallel_petsc_pyccel.py -ksp_rmonitor -snes_converged_reason -ksp_converged_reason
# -snes_monitor -snes_linesearch_monitor -pc_factor_mat_solver_type superlu_dist -snes_linesearch_type l2 -snes_lag_jacobian -2



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

Nx=10; Nt=5; use_interp = 1 # spatial-temporal grid sizes, use interpolation
multip=2 # mutiple for interpolation
tol = 1e-6
    
for i in range(1, 4):
    if i == 1:
        use_interp = 0
    else:
        use_interp = 1
    ####################### grid's inputs
    
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
    
    def formInitguess(snes, w, sendcounts, ww):
        if RANK == 0:
            ww = initialguess(Nt, Nx, multip)
        COMM.Scatterv([ww, sendcounts, MPI.DOUBLE], w.array, root = 0)
    
    def formJacobian(snes, w, J, P, sendcounts, ww):
        J.zeroEntries()
        
        COMM.Allgatherv(sendbuf=w.array, recvbuf=(ww, sendcounts))  
        
        compute_jacobian(ww, row, col, data, Nt, Nx, dt, dx, eps, np.array(da.ranges))
        
        
        for i in range(len(data)):
            J.setValues(row[i], col[i], data[i], addv=False)
        
        J.assemble()
        if P != J:
            P.assemble()
        return PETSc.Mat.Structure.SAME_NONZERO_PATTERN
    
    # """************************ solve in grid 1***************************** """
    def formFunction(snes, w, F, Nt, Nx, dt, dx, eps, u_max, rho_jam, x, sendcounts, ww, FF):
        
        # ts = time.process_time()###
        COMM.Allgatherv(sendbuf=w.array, recvbuf=(ww, sendcounts))  
        compute_FF(ww, FF, Nt, Nx, dt, dx, eps, u_max, rho_jam, x, np.array(da.ranges), RANK)
        signe = np.sign(FF) 
        FF = np.fabs(FF)
        
        sig = np.empty(sum(sendcounts), dtype=np.double)
        req1 = COMM.Ireduce([signe, MPI.DOUBLE],
                            [sig, MPI.DOUBLE],
                            op = MPI.SUM, 
                            root=0)
        req1.wait()
        totals = np.empty(sum(sendcounts), dtype=np.double)
        
        # use MPI to get the totals 
        req2 = COMM.Ireduce([FF, MPI.DOUBLE],
                    [totals, MPI.DOUBLE],
                    op = MPI.MAX, 
                    root=0)
        req2.wait()
        # FF = totals
        if RANK ==0:
            FF = totals*np.sign(sig)
        
        COMM.Scatterv([FF, sendcounts, MPI.DOUBLE], F.array, root = 0)
        
    F = daa.createGlobalVector()
    sendcounts = np.array(COMM.allgather(len(F.array)))
    ww = np.empty(sum(sendcounts), dtype=np.double)
    FF = np.empty(sum(sendcounts), dtype=np.double)
        
    args = [Nt, Nx, dt, dx, eps, u_max, rho_jam, x, sendcounts, ww, FF]
    snes.setFunction(formFunction, F, args)
    
    mat = PETSc.Mat().create()
    mat.setSizes((3*Nt*Nx+2*Nx,3*Nt*Nx+2*Nx))
    mat.setType("mpiaij")
    mat.setFromOptions()
    mat.setPreallocationNNZ(100)
    # mat.setOption(option=19, flag=0)
    
    
    args = [sendcounts, ww]
    snes.setJacobian(formJacobian, mat, mat, args)
    
    
    b = None
    xx = daa.createGlobalVector()
    
    if use_interp:
        snes.setInitialGuess(formInitguess, args)
    
    # snes.setType("nasm")
    snes.getKSP().setType('fgmres')
    
    ksp = snes.getKSP()
    ksp = PETSc.KSP().create()
    pc = ksp.getPC()
    opts = PETSc.Options()
    ksp.setTolerances(rtol=tol)
    opts["pc_type"] = "lu"
    opts["mat_mumps_icntl_23"] = 16000
    # ksp.setInitialGuessNonzero(True)
    ksp.setFromOptions()
    
    
    # snes.setUseMF(True)
    snes.setTolerances(max_it=1000, rtol = tol)
    snes.setFromOptions()
    
    t0 = time.process_time()   ###
    snes.solve(b, xx)
    t1 = time.process_time()   ###
    time2=t1-t0
    
    if RANK == 0:
        print("Time spent:",time2)
        its = snes.getIterationNumber()
        lits = snes.getLinearSolveIterations()
        
        print ("Number of SNES iterations = :", its)
        print ("Number of Linear iterations =" , lits)
        # print("norm", xx.norm())
        
    if RANK == 0:
        recvbuf = np.empty(sum(sendcounts), dtype=np.double)
    else:
        recvbuf = None
    
    COMM.Gatherv(sendbuf=xx.array, recvbuf=(recvbuf, sendcounts), root=0)
    if RANK == 0:
        # if not use_interp:
        import os
        filename = ("sol.dat")
        if os.path.exists(filename):
            os.remove(filename)
        
        with open(filename, "a") as text_file:
            text_file.write(str(Nx))
            text_file.write("\n")
            text_file.write(str(Nt))
            text_file.write("\n")
            np.savetxt(text_file, recvbuf)
    
    
    #Free petsc elements
    xx.destroy()      
    F.destroy()                                     
    snes.destroy()
    ksp.destroy()
