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
Nx=40; Nt=120 # spatial-temporal grid sizes
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
def U(rho): # Greenshields desired speed
    return u_max*(1-rho/rho_jam)
@njit
def f_mfg_LWR(u,r):
    return 0.5*((U(r)-u)**2) # MFG-LWR
@njit
def f_star_p_LWR(p,r): # 0<=u<=u_max
    return U(r)-p # MFG-LWR
@njit
def f_star_LWR(p,r): # p=Vx
    return -0.5*(p**2)+U(r)*p # MFG-LWR

def integral(a,b): 
    x2 = lambda x: rho_int(x)
    I=integrate.quad(x2, a, b)
    return I[0]
@njit
def rho_int(s): # initial density
    return rho_a+(rho_b-rho_a)*np.exp(-0.5*((s-0.5*L)/gama)**2) # 0<=rho<=rho_jam
@njit
def VT(a): # Terminal cost
    return 0.0
@njit
def r_idx(j,n):
    return (j-1)*(Nt+1)+n
@njit
def u_idx(j,n):
    return (Nt+1)*Nx+(j-1)*Nt+n
@njit
def V_idx(j,n):
    return (2*Nt+1)*Nx+(j-1)*(Nt+1)+n
@njit
def Fr_idx(j,n):
    return (j-1)*Nt+n
@njit
def Fu_idx(j,n):
    return Nt*Nx+(j-1)*Nt+n
@njit
def FV_idx(j,n):
    return 2*Nt*Nx+(j-1)*Nt+n
@njit
def Frint_idx(j):
    return 3*Nt*Nx+(j-1)
@njit
def FVter_idx(j):
    return 3*Nt*Nx+Nx+(j-1)

@njit
def f_star_p(p,r): # 0<=u<=u_max
    return U(r)-p # MFG-LWR
@njit
def f_star(p,r): # p=Vx
    return -0.5*(p**2)+U(r)*p # MFG-LWR

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

def formJacobian(snes, w, J, P):
    # w = np.zeros(3*Nt*Nx+2*Nx)
    P.zeroEntries()

    row = []; col = []; data = []
    for n in range(0,Nt):
        for j in range(1,Nx+1): # 1,Nx-1
            row.append(Fr_idx(j,n)); col.append(r_idx(j,n+1)); data.append(1)
            row.append(Fu_idx(j,n)); col.append(u_idx(j,n)); data.append(1)
            row.append(FV_idx(j,n)); col.append(V_idx(j,n)); data.append(-1)
            row.append(FV_idx(j,n)); col.append(V_idx(j,n+1)); data.append(1-2*eps)
            if j!=1:
                row.append(Fr_idx(j,n)); col.append(r_idx(j-1,n)); data.append(-(0.5*dt/dx)*w[u_idx(j-1,n)]-0.5)
                row.append(Fr_idx(j,n)); col.append(u_idx(j-1,n)); data.append(-(0.5*dt/dx)*w[r_idx(j-1,n)])
                row.append(FV_idx(j,n)); col.append(V_idx(j-1,n+1)); data.append(eps)
            if j==1:
                row.append(Fr_idx(j,n)); col.append(r_idx(Nx,n)); data.append((0.5*dt/dx)*w[u_idx(Nx,n)]-0.5)
                row.append(Fr_idx(j,n)); col.append(u_idx(Nx,n)); data.append(-(0.5*dt/dx)*w[r_idx(Nx,n)])
                row.append(FV_idx(j,n)); col.append(V_idx(Nx,n+1)); data.append(eps)
            if j!=Nx:
                row.append(Fr_idx(j,n)); col.append(r_idx(j+1,n)); data.append((0.5*dt/dx)*w[u_idx(j+1,n)]-0.5)
                row.append(Fr_idx(j,n)); col.append(u_idx(j+1,n)); data.append((0.5*dt/dx)*w[r_idx(j+1,n)])
                row.append(FV_idx(j,n)); col.append(V_idx(j+1,n+1)); data.append(eps)
            if j==Nx:
                row.append(Fr_idx(j,n)); col.append(r_idx(1,n)); data.append((0.5*dt/dx)*w[u_idx(1,n)]-0.5)
                row.append(Fr_idx(j,n)); col.append(u_idx(1,n)); data.append((0.5*dt/dx)*w[r_idx(1,n)])
                row.append(FV_idx(j,n)); col.append(V_idx(1,n+1)); data.append(eps)
  
    for j in range(1,Nx+1):
        row.append(Frint_idx(j)); col.append(r_idx(j,0)); data.append(1)
        row.append(FVter_idx(j)); col.append(V_idx(j,Nt)); data.append(1)
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

####################################""for interpolation""""""""""""""""""""""""""""
def sol_to(n_Nt,sol,rho,u,V):
    for j in range(0,Nx):
        for n in range(0,n_Nt):
            rho[j,n]=sol[j*(n_Nt+1)+n]
            u[j,n]=sol[(n_Nt+1)*Nx+j*n_Nt+n]
            V[j,n]=sol[(2*n_Nt+1)*Nx+j*(n_Nt+1)+n]
        rho[j,n_Nt]=sol[j*(n_Nt+1)+n_Nt]
        V[j,n_Nt]=sol[(2*n_Nt+1)*Nx+j*(n_Nt+1)+n_Nt]
    return 0

def to_sol(n_Nt,sol,rho,u,V):
    for j in range(0,2*Nx):
        for n in range(0,2*n_Nt):
            sol[j*(2*n_Nt+1)+n]=rho[j,n]
            sol[(2*n_Nt+1)*2*Nx+j*2*n_Nt+n]=u[j,n]
            sol[(2*2*n_Nt+1)*2*Nx+j*(2*n_Nt+1)+n]=V[j,n]
        sol[j*(2*n_Nt+1)+2*n_Nt]=rho[j,2*n_Nt]
        sol[(2*2*n_Nt+1)*2*Nx+j*(2*n_Nt+1)+2*n_Nt]=V[j,2*n_Nt]
    return 0


import scipy.interpolate as interpolate
def interpol(n,new_n,data): # 1D interpolation
    
    """" Go from a coarse grid Nt*Nx to a finer grid spacing (2*Nt)*(2*Nx) """""
    i = np.indices(data.shape)[0]/(n-1)  # [0, ..., 1]
    new_i = np.linspace(0, 1, new_n)
    linear_interpolation_func = interpolate.interp1d(i, data, kind='cubic') 
    # ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’.
    new_data = linear_interpolation_func(new_i)
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
####################################################################
 
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

J = PETSc.Mat().create()
J.setSizes(shap)
J.setFromOptions()
J.setUp()

w = np.zeros(3*Nt*Nx+2*Nx)

snes.setFunction(formFunction, F)
snes.setJacobian(formJacobian)

snes.getKSP().setType('fgmres')

snes.setFromOptions()



ksp = snes.getKSP()
pc = ksp.getPC()
opts = PETSc.Options()
opts["ksp_rtol"] = 1.0e-6
opts["pc_type"] = "lu"
ksp.setFromOptions()

snes.setTolerances(rtol = 1e-6)
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


# """************************ solve in grid 2***************************** """
Nxx=2*Nx; Ntt=2*Nt
# print('Nx={Nx}, Nt={Nt}'.format(Nx=Nxx,Nt=Ntt))
dx=L/Nxx # spatial step size
# print('dx={dx}, dt={dt}'.format(dx=round(dx,3),dt=round(dt,3)))
if mu==0.0:
    dt=min(T/Ntt,(CFL*dx)/u_max) # temporal step size
    eps=0.0
else:
    dt=min(T/Ntt,CFL*dx/abs(u_max),EPS*(dx**2)/mu) # temporal step size
    eps=mu*dt/(dx**2) # V
x=np.linspace(0,L,Nxx+1)
# t=np.linspace(0,T,Ntt+1)
t=np.arange(0,T+dt,dt)
Nt=int((len(t)-1)/2)
# t0 = time.process_time()   ###
guess=multigrid(int(Ntt/2),Nt,xx)
# t1 = time.process_time()   ###
# print("Time spent (multigrid) :",t1-t0)


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

J = PETSc.Mat().create()
J.setSizes(shap)
J.setFromOptions()
J.setUp()

w = np.zeros(3*Nt*Nx+2*Nx)

snes.setFunction(formFunction, F)
snes.setJacobian(formJacobian)

snes.getKSP().setType('fgmres')

snes.setFromOptions()



ksp = snes.getKSP()
pc = ksp.getPC()
opts = PETSc.Options()
opts["ksp_rtol"] = 1.0e-6
opts["pc_type"] = "lu"
ksp.setFromOptions()

snes.setTolerances(rtol = 1e-6)
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

