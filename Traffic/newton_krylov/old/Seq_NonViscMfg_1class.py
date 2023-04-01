##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 15:43:29 2021

@author: amal
"""


import numpy as np
from scipy import integrate
from scipy.optimize.nonlin import newton_krylov
import scipy.sparse.linalg as spla
from scipy.sparse import csc_matrix
import time

''' inputs '''
T=3.0 # horizon length  
u_max=1.0 # free flow speed
rho_jam=1.0 # jam density
L=1  #L=N # road length
CFL=0.75    # CFL<1
rho_a=0.05; rho_b=0.95; gama=0.1
# rho_a=0.2; rho_b=0.8; gama=0.15*L
# """ Viscosity condition"""
EPS=0.45 

''' functions '''
def U(rho): # Greenshields desired speed
    return u_max*(1-rho/rho_jam)

def f_mfg_LWR(u,r):
    return 0.5*((U(r)-u)**2) # MFG-LWR

def f_mfg_Sep(u,r):
    return 0.5*((u/u_max)**2)-(u/u_max)+(r/rho_jam) # MFG-Separable

def f_mfg_NonSep(u,r):
    return 0.5*((u/u_max)**2)-(u/u_max)+((u*r)/(u_max*rho_jam)) # MFG-NonSeparable

def f_star_p_LWR(p,r): # 0<=u<=u_max
    return U(r)-p # MFG-LWR

def f_star_p_Sep(p,r): # 0<=u<=u_max
    return max(min(u_max*(1-p*u_max),u_max),0) # MFG-Separable
    
def f_star_p_NonSep(p,r): # 0<=u<=u_max
    return max(min(u_max*(1-r/rho_jam-u_max*p),u_max),0) # MFG-NonSeparable
    
def f_star_LWR(p,r): # p=Vx
    return -0.5*(p**2)+U(r)*p # MFG-LWR
    
def f_star_Sep(p,r): # p=Vx
    return f_star_p_Sep(p,r)*p+f_mfg_Sep(f_star_p_Sep(p,r),r) # MFG-Separable
    
def f_star_NonSep(p,r): # p=Vx
    return f_star_p_NonSep(p,r)*p+f_mfg_NonSep(f_star_p_NonSep(p,r),r) # MFG-NonSeparable

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

def Fr_idx(j,n):
    return (j-1)*Nt+n

def Fu_idx(j,n):
    return Nt*Nx+(j-1)*Nt+n

def FV_idx(j,n):
    return 2*Nt*Nx+(j-1)*Nt+n

def Frint_idx(j):
    return 3*Nt*Nx+(j-1)

def FVter_idx(j):
    return 3*Nt*Nx+Nx+(j-1)


def Fct(w,f_star_p,f_star):
    # FF=[F_rho,F_u,F_V,F_rho_int,F_V_ter], F_rho:0->Nt*Nx-1, F_u:Nt*Nx->2*Nt*Nx-1, F_V:2*Nt*Nx->3*Nt*Nx-1, 
    # F_rho_int:3*Nt*Nx->3*Nt*Nx+Nx-1, F_V_ter:3*Nt*Nx+Nx->3*Nt*Nx+2*Nx-1
    FF=np.zeros(3*Nt*Nx+2*Nx)
    for n in range(0,Nt):
        # F_rho , F[0]->F[Nt-1] ************** 1  
        FF[Fr_idx(1,n)]=w[r_idx(1,n+1)]-0.5*(w[r_idx(Nx,n)]+w[r_idx(2,n)])\
            +(0.5*dt/dx)*(w[r_idx(2,n)]*w[u_idx(2,n)]-w[r_idx(Nx,n)]*w[u_idx(Nx,n)])
        # F_rho , F[Nt*Nx-Nt]->F[Nt*Nx-1] ********** 3 
        FF[Fr_idx(Nx,n)]=w[r_idx(Nx,n+1)]-0.5*(w[r_idx(Nx-1,n)]+w[r_idx(1,n)])\
            +(0.5*dt/dx)*(w[r_idx(1,n)]*w[u_idx(1,n)]-w[r_idx(Nx-1,n)]*w[u_idx(Nx-1,n)])
        # F_u , F[Nt*Nx]->F[Nt*Nx+Nt-1] *********** 4 
        FF[Fu_idx(1,n)]=w[u_idx(1,n)]-f_star_p((w[V_idx(1,n+1)]-w[V_idx(Nx,n+1)])/dx,w[r_idx(1,n)])
        # F_u , F[2*Nt*Nx-Nt]->F[2*Nt*Nx-1] ********* 6 
        FF[Fu_idx(Nx,n)]=w[u_idx(Nx,n)]-f_star_p((w[V_idx(Nx,n+1)]-w[V_idx(Nx-1,n+1)])/dx,w[r_idx(Nx,n)])
        # F_V , F[2*Nt*Nx]->F[2*Nt*Nx+Nt-1] *********** 7 
        FF[FV_idx(1,n)]=w[V_idx(1,n+1)]-w[V_idx(1,n)]+dt*f_star((w[V_idx(1,n+1)]-w[V_idx(Nx,n+1)])/dx,w[r_idx(1,n)])\
            +eps*(w[V_idx(2,n+1)]-2*w[V_idx(1,n+1)]+w[V_idx(Nx,n+1)])
        # F_V , F[3*Nt*Nx-Nt]->F[3*Nt*Nx-1] ********** 9 
        FF[FV_idx(Nx,n)]=w[V_idx(Nx,n+1)]-w[V_idx(Nx,n)]+dt*f_star((w[V_idx(Nx,n+1)]-w[V_idx(Nx-1,n+1)])/dx,w[r_idx(Nx,n)])\
            +eps*(w[V_idx(1,n+1)]-2*w[V_idx(Nx,n+1)]+w[V_idx(Nx-1,n+1)])
    for j in range(2,Nx):
        for n in range(0,Nt):
            # F_rho , F[Nt]->F[Nt*Nx-Nt-1] ************ 2 
            FF[Fr_idx(j,n)]=w[r_idx(j,n+1)]-0.5*(w[r_idx(j-1,n)]+w[r_idx(j+1,n)])\
                +(0.5*dt/dx)*(w[r_idx(j+1,n)]*w[u_idx(j+1,n)]-w[r_idx(j-1,n)]*w[u_idx(j-1,n)])
            # F_u , F[Nt*Nx+Nt]->F[2*Nt*Nx-Nt-1] *********** 5 
            FF[Fu_idx(j,n)]=w[u_idx(j,n)]-f_star_p((w[V_idx(j,n+1)]-w[V_idx(j-1,n+1)])/dx,w[r_idx(j,n)])
            # F_V , F[2*Nt*Nx+Nt]->F[3*Nt*Nx-Nt-1] ********* 8 
            FF[FV_idx(j,n)]=w[V_idx(j,n+1)]-w[V_idx(j,n)]+dt*f_star((w[V_idx(j,n+1)]-w[V_idx(j-1,n+1)])/dx,w[r_idx(j,n)])\
                +eps*(w[V_idx(j+1,n+1)]-2*w[V_idx(j,n+1)]+w[V_idx(j-1,n+1)])
        # F_rho_int , F[3*Nt*Nx+1]->F[3*Nt*Nx+Nx-2] ********** 11
        FF[Frint_idx(j)]=w[r_idx(j,0)]-(1/dx)*integral(x[j-1],x[j])
        # F_V_ter , F[3*Nt*Nx+Nx+1]->F[3*Nt*Nx+2*Nx-2] ********* 14
        FF[FVter_idx(j)]=w[V_idx(j,Nt)]-VT(x[j])
    # F_rho_int , F[3*Nt*Nx] ********* 10
    FF[Frint_idx(1)]=w[r_idx(1,0)]-(1/dx)*integral(x[0],x[1])
    # F_rho_int , F[3*Nt*Nx+Nx-1] ********* 12
    FF[Frint_idx(Nx)]=w[r_idx(Nx,0)]-(1/dx)*integral(x[Nx-1],x[Nx])
    # F_V_ter , F[3*Nt*Nx+Nx] *********** 13 
    FF[FVter_idx(1)]=w[V_idx(1,Nt)]-VT(x[1])
    # F_V_ter , F[3*Nt*Nx+2*Nx-1] ************** 15
    FF[FVter_idx(Nx)]=w[V_idx(Nx,Nt)]-VT(x[Nx])
    
    return FF


def jacobian(w,row,col,data): # Ignoring the forward-backward coupling  parts
    # J=np.zeros((3*Nt*Nx+2*Nx,3*Nt*Nx+2*Nx))
    cmpt = 0
    row[:] = 0; col[:] = 0; data[:] = 0.
    # row = []; col = []; data = []
    for n in range(0,Nt):
        for j in range(1,Nx+1): # 1,Nx-1
            # J[Fr_idx(j,n),r_idx(j,n+1)]=1 # F_rho - rho  ## Ok
            row[cmpt]=Fr_idx(j,n); col[cmpt]=r_idx(j,n+1); data[cmpt]=1
            cmpt +=1
            # J[Fu_idx(j,n),u_idx(j,n)]=1 # F_u - u  ## Ok
            row[cmpt]=Fu_idx(j,n); col[cmpt]=u_idx(j,n); data[cmpt]=1
            cmpt +=1
            # J[FV_idx(j,n),V_idx(j,n)]=-1 # F_V - V  ## Ok
            row[cmpt]=FV_idx(j,n); col[cmpt]=V_idx(j,n); data[cmpt]=-1
            cmpt +=1
            # J[FV_idx(j,n),V_idx(j,n+1)]=1-2*eps # F_V - V  ## Ok
            row[cmpt]=FV_idx(j,n); col[cmpt]=V_idx(j,n+1); data[cmpt]=1-2*eps
            cmpt +=1
            if j!=1:
                # J[Fr_idx(j,n),r_idx(j-1,n)]=-(0.5*dt/dx)*w[u_idx(j-1,n)]-0.5 # F_rho -rho  ## Ok
                row[cmpt]=Fr_idx(j,n); col[cmpt]=r_idx(j-1,n); data[cmpt]=-(0.5*dt/dx)*w[u_idx(j-1,n)]-0.5
                cmpt +=1
                # J[Fr_idx(j,n),u_idx(j-1,n)]=-(0.5*dt/dx)*w[r_idx(j-1,n)] # F_rho - u  ## Ok
                row[cmpt]=Fr_idx(j,n); col[cmpt]=u_idx(j-1,n); data[cmpt]=-(0.5*dt/dx)*w[r_idx(j-1,n)]
                cmpt +=1
                # J[FV_idx(j,n),V_idx(j-1,n+1)]=eps # F_V - V   ## Ok
                row[cmpt]=FV_idx(j,n); col[cmpt]=V_idx(j-1,n+1); data[cmpt]=eps
                cmpt +=1
            if j==1:
                # J[Fr_idx(j,n),r_idx(Nx,n)]=(0.5*dt/dx)*w[u_idx(Nx,n)]-0.5 # F_rho - rho
                row[cmpt]=Fr_idx(j,n); col[cmpt]=r_idx(Nx,n); data[cmpt]=-(0.5*dt/dx)*w[u_idx(Nx,n)]-0.5
                cmpt +=1
                # J[Fr_idx(j,n),u_idx(Nx,n)]=-(0.5*dt/dx)*w[r_idx(Nx,n)] # F_rho - u
                row[cmpt]=Fr_idx(j,n); col[cmpt]=u_idx(Nx,n); data[cmpt]=-(0.5*dt/dx)*w[r_idx(Nx,n)]
                cmpt +=1
                # J[FV_idx(j,n),V_idx(Nx,n+1)]=eps # F_V - V
                row[cmpt]=FV_idx(j,n); col[cmpt]=V_idx(Nx,n+1); data[cmpt]=eps
                cmpt +=1
            if j!=Nx:
                # J[Fr_idx(j,n),r_idx(j+1,n)]=(0.5*dt/dx)*w[u_idx(j+1,n)]-0.5 # F_rho -rho  ## Ok
                row[cmpt]=Fr_idx(j,n); col[cmpt]=r_idx(j+1,n); data[cmpt]=(0.5*dt/dx)*w[u_idx(j+1,n)]-0.5
                cmpt +=1
                # J[Fr_idx(j,n),u_idx(j+1,n)]=(0.5*dt/dx)*w[r_idx(j+1,n)] # F_rho - u ## Ok
                row[cmpt]=Fr_idx(j,n); col[cmpt]=u_idx(j+1,n); data[cmpt]=(0.5*dt/dx)*w[r_idx(j+1,n)]
                cmpt +=1
                # J[FV_idx(j,n),V_idx(j+1,n+1)]=eps # F_V - V  ## Ok
                row[cmpt]=FV_idx(j,n); col[cmpt]=V_idx(j+1,n+1); data[cmpt]=eps
                cmpt +=1
            if j==Nx:
                # J[Fr_idx(j,n),r_idx(1,n)]=(0.5*dt/dx)*w[u_idx(1,n)]-0.5 # F_rho - rho
                row[cmpt]=Fr_idx(j,n); col[cmpt]=r_idx(1,n); data[cmpt]=(0.5*dt/dx)*w[u_idx(1,n)]-0.5
                cmpt +=1
                # J[Fr_idx(j,n),u_idx(1,n)]=(0.5*dt/dx)*w[r_idx(1,n)] # F_rho - u
                row[cmpt]=Fr_idx(j,n); col[cmpt]=u_idx(1,n); data[cmpt]=(0.5*dt/dx)*w[r_idx(1,n)]
                cmpt +=1
                # J[FV_idx(j,n),V_idx(1,n+1)]=eps # F_V - V
                row[cmpt]=FV_idx(j,n); col[cmpt]=V_idx(1,n+1); data[cmpt]=eps
                cmpt +=1
  
    for j in range(1,Nx+1):
        # J[Frint_idx(j),r_idx(j,0)]=1 # F_rho_int - rho  ## Ok
        row[cmpt]=Frint_idx(j); col[cmpt]=r_idx(j,0); data[cmpt]=1
        cmpt +=1
        # J[FVter_idx(j),V_idx(j,Nt)]=1 # F_V_ter - V ## Ok
        row[cmpt]=FVter_idx(j); col[cmpt]=V_idx(j,Nt); data[cmpt]=1
        cmpt +=1
    
    # return J
    return 0


def get_preconditioner(a):
    # Jac=jacobian(a)
    # Jac1 = csc_matrix(Jac)
    row=np.zeros(10*Nt*Nx+2*Nx)
    col=np.zeros(10*Nt*Nx+2*Nx)
    data=np.zeros(10*Nt*Nx+2*Nx)
    jacobian(a,row, col, data)
    shap=(3*Nt*Nx+2*Nx,3*Nt*Nx+2*Nx)
    Jac1 = csc_matrix((data, (row, col)),shape = shap)
    # the *incomplete LU* decomposition
    J_ilu = spla.spilu(Jac1)
    M_x = lambda r: J_ilu.solve(r)
    M = spla.LinearOperator(shap, M_x)
    # M=np.linalg.inv(Jac1)
    return M

def sol_to(old_Nt,old_Nx,sol,rho,u,V): # solution 1D to 2D
    for j in range(0,old_Nx):
        for n in range(0,old_Nt):
            rho[j,n]=sol[j*(old_Nt+1)+n]
            u[j,n]=sol[(old_Nt+1)*old_Nx+j*old_Nt+n]
            V[j,n]=sol[(2*old_Nt+1)*old_Nx+j*(old_Nt+1)+n]
        rho[j,old_Nt]=sol[j*(old_Nt+1)+old_Nt]
        V[j,old_Nt]=sol[(2*old_Nt+1)*old_Nx+j*(old_Nt+1)+old_Nt]


def to_sol(new_Nt,old_Nx,sol,rho,u,V,multip):# solution 2D to 1D
    for j in range(0,multip*old_Nx):
        for n in range(0,multip*new_Nt):
            sol[j*(multip*new_Nt+1)+n]=rho[j,n]
            sol[(multip*new_Nt+1)*multip*old_Nx+j*multip*new_Nt+n]=u[j,n]
            sol[(2*multip*new_Nt+1)*multip*old_Nx+j*(multip*new_Nt+1)+n]=V[j,n]
        sol[j*(multip*new_Nt+1)+multip*new_Nt]=rho[j,multip*new_Nt]
        sol[(2*multip*new_Nt+1)*multip*old_Nx+j*(multip*new_Nt+1)+multip*new_Nt]=V[j,multip*new_Nt]


import scipy.interpolate as interpolate
def interpol(n,new_n,data): # 1D interpolation
    
    """" Go from a coarse grid Nt*Nx to a finer grid spacing (2*Nt)*(2*Nx) """""
    i = np.indices(data.shape)[0]/(n-1)  # [0, ..., 1]
    new_i = np.linspace(0, 1, new_n)
    # Create a linear interpolation function based on the original data
    linear_interpolation_func = interpolate.interp1d(i, data, kind='cubic') 
    # ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’.
    new_data = linear_interpolation_func(new_i)
    return new_data


def multigrid(new_Nt,old_Nt,old_Nx,w,multip):
    
    rho=np.zeros((old_Nx,old_Nt+1))
    u=np.zeros((old_Nx,old_Nt))
    V=np.zeros((old_Nx,old_Nt+1))
    sol_to(old_Nt,old_Nx, w,rho,u,V)
    
    new1_rho=np.zeros((multip*old_Nx,old_Nt+1))
    new1_u=np.zeros((multip*old_Nx,old_Nt))
    new1_V=np.zeros((multip*old_Nx,old_Nt+1))
    for n in range(old_Nt):
        new1_rho[:,n]=interpol(old_Nx,multip*old_Nx,rho[:,n])
        new1_u[:,n]=interpol(old_Nx,multip*old_Nx,u[:,n])
        new1_V[:,n]=interpol(old_Nx,multip*old_Nx,V[:,n])
    new1_rho[:,old_Nt]=interpol(old_Nx,multip*old_Nx,rho[:,old_Nt])
    new1_V[:,old_Nt]=interpol(old_Nx,multip*old_Nx,V[:,old_Nt])
    new_rho=np.zeros((multip*old_Nx,multip*new_Nt+1))
    new_u=np.zeros((multip*old_Nx,multip*new_Nt))
    new_V=np.zeros((multip*old_Nx,multip*new_Nt+1))
    for j in range(multip*old_Nx):
        new_rho[j,:]=interpol(old_Nt+1,multip*new_Nt+1,new1_rho[j,:])
        new_u[j,:]=interpol(old_Nt,multip*new_Nt,new1_u[j,:])
        new_V[j,:]=interpol(old_Nt+1,multip*new_Nt+1,new1_V[j,:])
        
    new_w = np.zeros(3*(multip*new_Nt)*(multip*old_Nx)+2*(multip*old_Nx))
    to_sol(new_Nt, old_Nx, new_w,new_rho,new_u,new_V, multip)
    
    return new_w


#---------Functions     
F  = lambda x : Fct(x,f_star_p_LWR,f_star_LWR) # LWR
# F  = lambda x : Fct(x,f_star_p_Sep,f_star_Sep) # Separable
# F  = lambda x : Fct(x,f_star_p_NonSep,f_star_NonSep) # Non-Separable

#---------viscosity coefficient
mu=[0.0]*(7) # LWR  
# mu=[0.05, 0.045, 0.04, 0.035, 0.03, 0.02]  # Sep  
# mu=[0.05, 0.045, 0.04, 0.035, 0.03, 0.02] # NonSep 

#---------For save_text
guess_text='guess.dat'
sol_text='sol.dat'

#---------Coarse grid
Nx=15; Nt=60 # spatial-temporal grid sizes 
multip=2

use_precond=1 
use_multigrid=1 

import os
import sys 
stdoutOrigin=sys.stdout 
sys.stdout = open("outputs.dat", "w")

for i in range(0,5):
    if i == 0:
        use_interp = 0
    else:
        use_interp = 1
        sol=np.loadtxt(sol_text)
        old_Nx=int(sol[0]); old_Nt=int(sol[1])
        sol=sol[2:]
        Nx=old_Nx*multip; Nt=old_Nt*multip

    dx=L/Nx # spatial step size
    if mu[i]==0.0:
        dt=min(T/Nt,(CFL*dx)/u_max) # temporal step size
        eps=0.0
    else:
        dt=min(T/Nt,CFL*dx/abs(u_max),EPS*(dx**2)/mu[i]) # temporal step size
        eps=mu[i]*dt/(dx**2) # V
    x=np.linspace(0,L,Nx+1)
    t=np.arange(0,T+dt,dt)
    Nt=multip*int((len(t)-1)/multip)
    print('Nx={Nx}, Nt={Nt}, nu={nu}'.format(Nx=Nx,Nt=Nt,nu=mu[i]))
    print('dx={dx}, dt={dt}'.format(dx=round(dx,4),dt=round(dt,4)))
    
    #---------------------MultiGrid
    if use_interp==1 and use_multigrid==1:
            t0 = time.process_time()   ###
            guess=multigrid(int(Nt/multip),old_Nt,old_Nx,sol,multip)
            t1 = time.process_time()   ###
            print("Time spent (multigrid) :",t1-t0)
            if os.path.exists(guess_text):
                os.remove(guess_text)
            np.savetxt(guess_text, guess)
    elif use_interp==0: guess = np.zeros(3*Nt*Nx+2*Nx)
    #---------------------preconditionning
    if use_precond==1:
        t0 = time.process_time()   ###
        prec=get_preconditioner(guess)
        t1 = time.process_time()   ###
        time1=t1-t0
        print("Time spent (anal_precond) :",time1)
    elif use_precond==0: prec=None
    #---------------------Newton-GMRES
    t0 = time.process_time()   ###
    sol = newton_krylov(F, guess, method='lgmres', verbose=1, inner_M=prec) # f_tol=2e-08 (default 6e-06), maxiter=500 (default 50000)
    t1 = time.process_time()   ###
    time2=t1-t0
    print("Time spent (gmres) :",time2)
    cpu_time=time1+time2
    print("CPU time :",cpu_time)
    #------Save : Nx,Nt,sol
    # import os
    if os.path.exists(sol_text):
        os.remove(sol_text)
    with open(sol_text, "a") as text_file:
            text_file.write(str(Nx))
            text_file.write("\n")
            text_file.write(str(Nt))
            text_file.write("\n")
            np.savetxt(text_file, sol)
    
sys.stdout.close()
sys.stdout=stdoutOrigin
print('End')

    
    
    
