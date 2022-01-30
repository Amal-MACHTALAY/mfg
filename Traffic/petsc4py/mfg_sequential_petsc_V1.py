##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 15:43:29 2021

@author: amal
"""


import numpy as np
from scipy import integrate
import time
from numba import njit

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
multip=2 # mutiple for interpolation
Nx=15; Nt=60; use_interp = 1 # spatial-temporal grid sizes, use interpolation
if use_interp :
    Nx=15*multip; Nt=60*multip
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
print('Nx={Nx}, Nt={Nt}'.format(Nx=Nx,Nt=Nt))
print('dx={dx}, dt={dt}'.format(dx=round(dx,4),dt=round(dt,4)))

'''************************ functions **********************************'''
@njit
def U(rho:'float[:]'): # Greenshields desired speed
    return u_max*(1-rho/rho_jam)
@njit
def f_mfg_LWR(u:'float[:]',r:'float[:]'):
    return 0.5*((U(r)-u)**2) # MFG-LWR
@njit
def f_star_p_LWR(p:'float[:]',r:'float[:]'): # 0<=u<=u_max
    return U(r)-p # MFG-LWR
@njit
def f_star_LWR(p:'float[:]',r:'float[:]'): # p=Vx
    return -0.5*(p**2)+U(r)*p # MFG-LWR


def integral(a,b): 
    x2 = lambda x: rho_int(x)
    I=integrate.quad(x2, a, b)
    return I[0]

@njit
def rho_int(s:'float[:]'): # initial density
    return rho_a+(rho_b-rho_a)*np.exp(-0.5*((s-0.5*L)/gama)**2) # 0<=rho<=rho_jam
@njit
def VT(a:'float[:]'): # Terminal cost
    return 0.0
@njit
def r_idx(j:'int[:]',n:'int[:]'):
    return (j-1)*(Nt+1)+n
@njit
def u_idx(j:'int[:]',n:'int[:]'):
    return (Nt+1)*Nx+(j-1)*Nt+n
@njit
def V_idx(j:'int[:]',n:'int[:]'):
    return (2*Nt+1)*Nx+(j-1)*(Nt+1)+n
@njit
def Fr_idx(j:'int[:]',n:'int[:]'):
    return (j-1)*Nt+n
@njit
def Fu_idx(j:'int[:]',n:'int[:]'):
    return Nt*Nx+(j-1)*Nt+n
@njit
def FV_idx(j:'int[:]',n:'int[:]'):
    return 2*Nt*Nx+(j-1)*Nt+n
@njit
def Frint_idx(j:'int[:]'):
    return 3*Nt*Nx+(j-1)
@njit
def FVter_idx(j:'int[:]'):
    return 3*Nt*Nx+Nx+(j-1)

@njit
def f_star_p(p:'float[:]',r:'float[:]'): # 0<=u<=u_max
    return U(r)-p # MFG-LWR
@njit
def f_star(p:'float[:]',r:'float[:]'): # p=Vx
    return -0.5*(p**2)+U(r)*p # MFG-LWR

####################################""for interpolation""""""""""""""""""""""""""""
@njit
def sol_to(old_Nt:'int', old_Nx:'int', sol:'float[:]',rho:'float[:]',u:'float[:]',V:'float[:,:]'): # solution 1D to 2D
    for j in range(0,old_Nx):
        for n in range(0,old_Nt):
            rho[j,n]=sol[j*(old_Nt+1)+n]
            u[j,n]=sol[(old_Nt+1)*old_Nx+j*old_Nt+n]
            V[j,n]=sol[(2*old_Nt+1)*old_Nx+j*(old_Nt+1)+n]
        rho[j,old_Nt]=sol[j*(old_Nt+1)+old_Nt]
        V[j,old_Nt]=sol[(2*old_Nt+1)*old_Nx+j*(old_Nt+1)+old_Nt]
    return 0

@njit
def to_sol(new_Nt:'int', old_Nx:'int', sol:'float[:]', rho:'float[:,:]', u:'float[:,:]', V:'float[:,:]'):# solution 2D to 1D
    for j in range(0,multip*old_Nx):
        for n in range(0,multip*new_Nt):
            sol[j*(multip*new_Nt+1)+n]=rho[j,n]
            sol[(multip*new_Nt+1)*multip*old_Nx+j*multip*new_Nt+n]=u[j,n]
            sol[(2*multip*new_Nt+1)*multip*old_Nx+j*(multip*new_Nt+1)+n]=V[j,n]
        sol[j*(multip*new_Nt+1)+multip*new_Nt]=rho[j,multip*new_Nt]
        sol[(2*multip*new_Nt+1)*multip*old_Nx+j*(multip*new_Nt+1)+multip*new_Nt]=V[j,multip*new_Nt]
    return 0


import scipy.interpolate as interpolate
def interpol(n,new_n,data): # 1D interpolation
    
    """" Go from a coarse grid Nt*Nx to a finer grid spacing (2*Nt)*(2*Nx) """""
    i = np.indices(data.shape)[0]/(n-1)  # [0, ..., 1]
    new_i = np.linspace(0, 1, new_n)
    linear_interpolation_func = interpolate.interp1d(i, data, kind='linear') 
    # ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’.
    new_data = linear_interpolation_func(new_i)
    return new_data


def initialguess(X):
    
    new_Nt = int(Nt/multip)

    w = np.loadtxt("sol.dat")
    
    old_Nx = int(w[0])
    old_Nt = int(w[1])
    w = w[2:]

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
    to_sol(new_Nt, old_Nx, new_w,new_rho,new_u,new_V)
  
    X = new_w
    
    # print(X)
    
    return X

def formFunction(snes, w, F):
    FF = F.array
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

@njit
def compute_jacobian(w:'float[:]', row:'int[:]', col:'int[:]', data:'float[:]'):
    
    cmpt = 0
    for n in range(0,Nt):
        for j in range(1,Nx+1): # 1,Nx-1
            row[cmpt] = Fr_idx(j,n); col[cmpt] = r_idx(j,n+1); data[cmpt] = 1
            cmpt +=1
            row[cmpt] = Fu_idx(j,n); col[cmpt] = u_idx(j,n); data[cmpt] = 1
            cmpt +=1
            row[cmpt] = FV_idx(j,n); col[cmpt] = V_idx(j,n); data[cmpt] = -1
            cmpt +=1
            row[cmpt] = FV_idx(j,n); col[cmpt] = V_idx(j,n+1); data[cmpt] = 1-2*eps
            cmpt +=1
            
            if j!=1:
                row[cmpt] = Fr_idx(j,n); col[cmpt] = r_idx(j-1,n); data[cmpt] = -(0.5*dt/dx)*w[u_idx(j-1,n)]-0.5
                cmpt +=1
                row[cmpt] = Fr_idx(j,n); col[cmpt] = u_idx(j-1,n); data[cmpt] = -(0.5*dt/dx)*w[r_idx(j-1,n)]
                cmpt +=1
                row[cmpt] = FV_idx(j,n); col[cmpt] = V_idx(j-1,n+1); data[cmpt] = eps
                cmpt +=1
                
            if j==1:
                row[cmpt] = Fr_idx(j,n); col[cmpt] = r_idx(Nx,n); data[cmpt] = (0.5*dt/dx)*w[u_idx(Nx,n)]-0.5
                cmpt +=1
                row[cmpt] = Fr_idx(j,n); col[cmpt] = u_idx(Nx,n); data[cmpt] = -(0.5*dt/dx)*w[r_idx(Nx,n)]
                cmpt +=1
                row[cmpt] = FV_idx(j,n); col[cmpt] = V_idx(Nx,n+1); data[cmpt] = eps
                cmpt +=1
           
            if j!=Nx:
                row[cmpt] = Fr_idx(j,n); col[cmpt] = r_idx(j+1,n); data[cmpt] = (0.5*dt/dx)*w[u_idx(j+1,n)]-0.5
                cmpt +=1
                row[cmpt] = Fr_idx(j,n); col[cmpt] = u_idx(j+1,n); data[cmpt] = (0.5*dt/dx)*w[r_idx(j+1,n)]
                cmpt +=1
                row[cmpt] = FV_idx(j,n); col[cmpt] = V_idx(j+1,n+1); data[cmpt] = eps
                cmpt +=1

            if j==Nx:
                row[cmpt] = Fr_idx(j,n); col[cmpt] = r_idx(1,n); data[cmpt] = (0.5*dt/dx)*w[u_idx(1,n)]-0.5
                cmpt +=1
                row[cmpt] = Fr_idx(j,n); col[cmpt] = u_idx(1,n); data[cmpt] = (0.5*dt/dx)*w[r_idx(1,n)]
                cmpt +=1
                row[cmpt] = FV_idx(j,n); col[cmpt] = V_idx(1,n+1); data[cmpt] = eps
                cmpt +=1
                
    
    for j in range(1,Nx+1):
        row[cmpt] = Frint_idx(j); col[cmpt] = r_idx(j,0); data[cmpt] = 1
        cmpt +=1
        row[cmpt] = FVter_idx(j); col[cmpt] = V_idx(j,Nt); data[cmpt] = 1
        cmpt +=1
        

row = np.zeros(10*Nt*Nx+2*Nx); col = np.zeros(10*Nt*Nx+2*Nx); data = np.zeros(10*Nt*Nx+2*Nx);
def formJacobian(snes, w, J, P):
    # w = np.zeros(3*Nt*Nx+2*Nx)
    P.zeroEntries()

    row[:] = 0; col[:] = 0.; data[:] = 0.
    
    compute_jacobian(w, row, col, data)
    
    P.setType("mpiaij")
    P.setFromOptions()
    P.setPreallocationNNZ(10)
    # P.setOption(option=19, flag=0)
    
    
    for i in range(len(data)):
        P.setValues(row[i], col[i], data[i], addv=False)
    
    P.assemble()
    if J != P:
        J.assemble()
            
    return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

 
# """************************ solve in grid 1***************************** """
from petsc4py import PETSc

shap=(3*Nt*Nx+2*Nx,3*Nt*Nx+2*Nx)

# create nonlinear solver
snes = PETSc.SNES()
snes.create()

F = PETSc.Vec()
F.create()
F.setSizes(shap[0])
F.setFromOptions()

b = None
xx = PETSc.Vec().createSeq(shap[0]) 

snes.setFunction(formFunction, F)
snes.setJacobian(formJacobian)

if use_interp:
    # snes.setInitialGuess(initialguess)
    X = np.zeros(shap[0])
    X = initialguess(X)#snes.getInitialGuess()[0](snes, xx)
    xx.setArray(X)


# snes.setType("ngmres")
snes.getKSP().setType('lgmres')
# snes.setFromOptions()

ksp = snes.getKSP()
pc = ksp.getPC()
pc.setFactorSolverType("mumps")
opts = PETSc.Options()
opts["ksp_rtol"] = 1.0e-6
opts["pc_type"] = "lu"
ksp.setInitialGuessNonzero(True)
ksp.setFromOptions()

snes.setTolerances(rtol = 1e-8)
snes.setFromOptions()

t0 = time.process_time()   ###
snes.solve(b, xx)
t1 = time.process_time()   ###
time2=t1-t0
print("Time spent:",time2)


its = snes.getIterationNumber()
lits = snes.getLinearSolveIterations()

print ("Number of SNES iterations = :", its)
print ("Number of Linear iterations =" , lits)

litspit = lits/float(its)
print ("Average Linear its / SNES = %e", float(litspit))


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


#Free petsc elements
xx.destroy()      
F.destroy()                                     
snes.destroy()
ksp.destroy()