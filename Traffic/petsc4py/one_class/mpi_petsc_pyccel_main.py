##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 15:43:29 2021
@author: amal
"""

import numpy as np
# import timeit
from modules import (compute_jacobian, compute_FF)
from tools import initialguess, solutions
# import time
from mpi4py import MPI

COMM = MPI.COMM_WORLD  # The default communicator
SIZE = COMM.Get_size() # The number of processes
RANK = COMM.Get_rank() # The rank of processes

'''************************ inputs ************************************'''
text = "sol_petsc_grid{}.npz"
T=3.0 # horizon length  
u_max=1.0 # free flow speed
rho_jam=1.0 # jam density
L=1 # road length
CFL=0.75    # CFL<1
rho_a=0.05; rho_b=0.95; gama=0.1 
mu=0.0 # viscosity coefficient 
EPS=0.45

Nx=15; Nt=60; use_interp = 1 # spatial-temporal grid sizes, use interpolation
multip=2 # mutiple for interpolation
tol = 6e-6
    
for i in range(1, 8):
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
    
    numbr = 14
    row = np.zeros(numbr*Ntloc*Nxloc+2*Nxloc, dtype=np.int64); col = np.zeros(numbr*Ntloc*Nxloc+2*Nxloc, dtype=np.int64); 
    data = np.zeros(numbr*Ntloc*Nxloc+2*Nxloc, np.double);
    
    # print(RANK,da.getRanges()[0][0], da.getRanges()[0][1],da.getRanges()[1][0], da.getRanges()[1][1],Ntloc,Nxloc,Nt,Nx)
    
    def formInitguess(snes, w, sendcounts, ww):
        if RANK == 0:
            ww = initialguess(Nt, Nx, multip, text.format(i-1))
            # print(len(ww),3*Nt*Nx+2*Nx)
            # print(ww[:10])
        COMM.Scatterv([ww, sendcounts, MPI.DOUBLE], w.array, root = 0)
    
    def formJacobian(snes, w, J, P, sendcounts, ww):
        J.zeroEntries()
        
        COMM.Allgatherv(sendbuf=w.array, recvbuf=(ww, sendcounts))  
        
        compute_jacobian(ww, row, col, data, u_max, rho_jam, Nt, Nx, dt, dx, eps, np.array(da.ranges))
        
        # shap=(3*Nt*Nx+2*Nx,3*Nt*Nx+2*Nx)
        # from scipy.sparse import csc_matrix
        # Jac1 = csc_matrix((data, (row, col)),shape = shap)
        
        for i in range(len(data)):
            J.setValues(row[i], col[i], data[i], addv=False)
        
        J.assemble()
        if P != J:
            P.assemble()
        return PETSc.Mat.Structure.SAME_NONZERO_PATTERN
    
    # """************************ solve in grid 1***************************** """
    def formFunction(snes, w, F, Nt, Nx, dt, dx, eps, u_max, rho_jam, x, sendcounts, ww, FF):
        
        # ts = timeit.default_timer()###
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
    mat.setType("mpiaij")    # mpiaij , seqaij , seqdense , mpidense , seqbaij , mpibaij 
    mat.setFromOptions()
    mat.setPreallocationNNZ(5)
    # mat.setOption(option=19, flag=0)
    
    # ww[1:]=0.
    args = [sendcounts, ww]
    snes.setJacobian(formJacobian, mat, mat, args)
    
    
    b = None
    xx = daa.createGlobalVector()
    
    if use_interp:
        snes.setInitialGuess(formInitguess, args)
    
    # snes.setType("nasm")
    
                                                          # KSP : Krylov-subspace-preconditioner (linear solver)
    snes.getKSP().setType('gmres')                       # Krylov subspace technique : 'gmres', 'fgmres' , 'lgmres', 'bcgs', ...
    
    ksp = snes.getKSP() #
    ksp = PETSc.KSP().create()
    pc = ksp.getPC()
    opts = PETSc.Options()
    ksp.setTolerances(rtol=tol, atol=6.e-06, max_it=50000)                          
                                                # rtol : the relative convergence tolerance 
                                                # abstol (atol) : the absolute convergence tolerance of the residual norm
                                                # maxits (max_it) : maximum number of iterations to use 
    opts["pc_type"] = "lu"                     # "lu", "ilu", "gamg" ,"hypre", "bjacobi","jacobi", "sor", "asm",  "gasm", "telescope"
    opts["mat_mumps_icntl_23"] = 64000          # max size of the working memory (MB) that can allocate per processor 
    ksp.setInitialGuessNonzero(True)
    ksp.setFromOptions()
    
    
    # snes.setUseMF(True)
    snes.setTolerances(max_it=50000, rtol = tol)
    snes.setFromOptions()
    
    t0 = MPI.Wtime()   ###
    snes.solve(b, xx)
    t1 = MPI.Wtime()    ### timeit.default_timer()   ###
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
        rho, u, V, Q = solutions(recvbuf,Nt,Nx)
        np.savez(text.format(i), Nx=Nx, Nt=Nt, dx=dx, dt=dt, mu=mu, solution=recvbuf, t_points=t, x_points=x, rho=rho, u=u, V=V, Q=Q)
        # if not use_interp:
        import os
        filename = (text.format(i))
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