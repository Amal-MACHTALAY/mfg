#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 18:59:43 2022

@author: amal
"""

import numpy as np
from scipy.optimize.nonlin import newton_krylov
from inputs import inputs, f_starp, f_star, rho1_int, rho2_int, VT
from indx_funcs import solutions
from newton_func import newton_func
from jacobian_exact import jacobian
from multi_grid import multigrid
from precond import get_preconditioner
import time



T,L,l1,l2,u1_max,u2_max,rho1_jam,rho2_jam, CFL, EPS, nb_grid, mu, Nx0, Nt0, multip, use_precond, use_multigrid = inputs()

#---------For save_text
nnfile_name = 'data_inputs.npz'

#---------Coarse grid
Nx=Nx0; Nt=Nt0 # spatial-temporal grid sizes 

np.savez(nnfile_name, T=T, L=L, l1=l1, u1_max=u1_max, u2_max=u2_max, rho1_jam=rho1_jam, rho2_jam=rho2_jam, CFL=CFL, EPS=EPS, nb_grid=nb_grid, Nx0=Nx, Nt0=Nt, multip=multip, 
         use_precond=use_precond, use_multigrid=use_multigrid)

# import os
import sys 
stdoutOrigin=sys.stdout 
sys.stdout = open("outputs.dat", "w")

for i in range(0,nb_grid):
    if i == 0:
        use_interp = False
    else:
        use_interp = True
        npzfile = np.load('data_solutions_grid{}.npz'.format(i-1))
        sol=npzfile['solution']
        old_Nx=npzfile['Nx']; old_Nt=npzfile['Nt']
        Nx=old_Nx*multip; Nt=old_Nt*multip

    dx=L/Nx # spatial step size
    if mu[i]==0.0:
        dt=min(T/Nt,(CFL*dx)/max(u1_max,u2_max)) # temporal step size
        eps=0.0
    else:
        dt=min(T/Nt,CFL*dx/max(u1_max,u2_max),EPS*(dx**2)/mu[i]) # temporal step size
        eps=mu[i]*dt/(dx**2) # V
    x=np.linspace(0,L,Nx+1)
    t=np.arange(0,T+dt,dt)
    Nt=multip*int((len(t)-1)/multip)
    print('Nx={Nx}, Nt={Nt}, nu={nu}'.format(Nx=Nx,Nt=Nt,nu=mu[i]))
    print('dx={dx}, dt={dt}'.format(dx=round(dx,4),dt=round(dt,4)))
    
    #---------------------MultiGrid
    if use_interp==True and use_multigrid==True:
            t0 = time.process_time()   ###
            guess=multigrid(int(Nt/multip),old_Nt,old_Nx,sol,multip)   
            t1 = time.process_time()   ###
            print("Time spent (multigrid) :",t1-t0)
    elif use_interp==False: guess = np.zeros(3*Nt*(2*Nx)+2*(2*Nx))
    #---------------------preconditionning
    prec=None
    if use_precond==True:
        t0 = time.process_time()   ###
        prec=get_preconditioner(guess,Nt,Nx,dt,dx,eps,jacobian,u1_max,u2_max)
        t1 = time.process_time()   ###
        time1=t1-t0
        print("Time spent (anal_precond) :",time1)
    #---------------------Newton-GMRES
    t0 = time.process_time()   ###
    newton_F = lambda w : newton_func(w,f_starp,f_star,rho1_int,rho2_int,Nt,Nx,dt,dx,eps,x,VT,u1_max,u2_max)
    sol = newton_krylov(newton_F, guess, method='lgmres', verbose=1, inner_M=prec) # f_tol=2e-08 (default 6e-06), maxiter=500 (default 50000)
    t1 = time.process_time()   ###
    time2=t1-t0
    print("Time spent (gmres) :",time2)
    cpu_time=time1+time2
    print("CPU time :",cpu_time)
    #------Save : Nx,Nt,sol
    s_rho1,s_u1,s_V1,s_Q1,s_rho2,s_u2,s_V2,s_Q2 = solutions(sol,Nt,Nx)
    g_rho1,g_u1,g_V1,g_Q1,g_rho2,g_u2,g_V2,g_Q2 = solutions(guess,Nt,Nx)
    np.savez('data_solutions_grid{}.npz'.format(i), Nx=Nx, Nt=Nt, dx=dx, dt=dt, mu=mu[i], guess=guess, solution=sol,
             g_rho1=g_rho1 ,g_u1=g_u1, g_V1=g_V1, g_Q1=g_Q1, g_rho2=g_rho2, g_u2=g_u2, g_V2=g_V2, g_Q2=g_Q2,
             s_rho1=s_rho1, s_u1=s_u1, s_V1=s_V1, s_Q1=s_Q1, s_rho2=s_rho2, s_u2=s_u2, s_V2=s_V2, s_Q2=s_Q2, x_points=x, t_points=t)
    
sys.stdout.close()
sys.stdout=stdoutOrigin
print('End')



