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
from scipy.interpolate import griddata
from scipy.sparse import csc_matrix
import time

''' inputs '''
T=3.0 # horizon length 
N=1 # number of cars 
u_max=1.0 # free flow speed
rho_jam=1.0 # jam density
L=N # road length
CFL=0.75    # CFL<1
rho_a=0.05; rho_b=0.95; gama=0.1
# rho_a=0.2; rho_b=0.8; gama=0.15*L
# """ Non-viscous solution"""
ep1=0.0  # rho
ep2=0.0  # V
# """ Viscous solution"""
# EPS=0.45
# mu=0.05 # viscosity coefficient 

Nx_list=[]
Nt_list=[]
costf="LWR"

''' functions '''
def U(rho): # Greenshields desired speed
    return u_max*(1-rho/rho_jam)

def f_mfg(u,r):
    if costf=="LWR":
        return 0.5*((U(r)-u)**2) # MFG-LWR
    elif costf=="Sep":
        return 0.5*((u/u_max)**2)-(u/u_max)+(r/rho_jam) # MFG-Separable
    elif costf=="NonSep":
        return 0.5*((u/u_max)**2)-(u/u_max)+((u*r)/(u_max*rho_jam)) # MFG-NonSeparable

def f_star_p(p,r): # 0<=u<=u_max
    if costf=="LWR":
        return U(r)-p # MFG-LWR
    elif costf=="Sep":
        return max(min(u_max*(1-p*u_max),u_max),0) # MFG-Separable
    elif costf=="NonSep":
        return max(min(u_max*(1-r/rho_jam-u_max*p),u_max),0) # MFG-NonSeparable
    
def f_star(p,r): # p=Vx
    if costf=="LWR":
        return -0.5*(p**2)+U(r)*p # MFG-LWR
    elif costf=="Sep":
        return f_star_p(p,r)*p+f_mfg(f_star_p(p,r),r) # MFG-Separable
    elif costf=="NonSep":
        return f_star_p(p,r)*p+f_mfg(f_star_p(p,r),r) # MFG-NonSeparable

def integral(a,b): 
    x2 = lambda x: rho_int(x)
    I=integrate.quad(x2, a, b)
    return I[0]

def rho_int(s): # initial density
    return rho_a+(rho_b-rho_a)*np.exp(-0.5*((s-0.5*L)/gama)**2) # 0<=rho<=rho_jam

def VT(a): # Terminal cost
    return 0.0

def F(w):
    # FF=[F_rho,F_u,F_V,F_rho_int,F_V_ter], F_rho:0->Nt*Nx-1, F_u:Nt*Nx->2*Nt*Nx-1, F_V:2*Nt*Nx->3*Nt*Nx-1, F_rho_int:3*Nt*Nx->3*Nt*Nx+Nx-1, F_V_ter:3*Nt*Nx+Nx->3*Nt*Nx+2*Nx-1
    FF=np.zeros(3*Nt*Nx+2*Nx)
    for n in range(0,Nt):
        # F_rho , F[0]->F[Nt-1] ************** 1
        FF[n]=w[n+1]-0.5*w[n+Nt+1]+(0.5*dt/dx)*w[n+Nt+1]*w[n+(Nt+1)*Nx+Nt]+ep1*(w[n+Nt+1]-2*w[n])
        # F_rho , F[Nt*Nx-Nt]->F[Nt*Nx-1] ********** 3
        FF[Nt*(Nx-1)+n]=w[(Nt+1)*(Nx-1)+n+1]-0.5*w[(Nt+1)*(Nx-2)+n]-(0.5*dt/dx)*w[(Nt+1)*(Nx-2)+n]*w[(Nt+1)*Nx+(Nx-2)*Nt+n]+ep1*(-2*w[(Nt+1)*(Nx-1)+n]+w[(Nt+1)*(Nx-2)+n])
        # F_u , F[Nt*Nx]->F[Nt*Nx+Nt-1] *********** 4
        FF[Nt*Nx+n]=w[(Nt+1)*Nx+n]-f_star_p(w[(2*Nt+1)*Nx+n+1]/dx,w[n])
        # F_u , F[2*Nt*Nx-Nt]->F[2*Nt*Nx-1] ********* 6
        FF[2*Nt*Nx-Nt+n]=w[(Nt+1)*Nx+(Nx-1)*Nt+n]-f_star_p((w[(2*Nt+1)*Nx+(Nx-1)*(Nt+1)+n+1]-w[(2*Nt+1)*Nx+(Nx-2)*(Nt+1)+n+1])/dx,w[(Nx-1)*(Nt+1)+n])
        # F_V , F[2*Nt*Nx]->F[2*Nt*Nx+Nt-1] *********** 7
        FF[2*Nt*Nx+n]=w[(2*Nt+1)*Nx+n+1]-w[(2*Nt+1)*Nx+n]+dt*f_star(w[(2*Nt+1)*Nx+n+1]/dx,w[n])+ep2*(w[(2*Nt+1)*Nx+Nt+n+2]-2*w[(2*Nt+1)*Nx+n+1])
        # F_V , F[3*Nt*Nx-Nt]->F[3*Nt*Nx-1] ********** 9
        FF[3*Nt*Nx-Nt+n]=w[(2*Nt+1)*Nx+(Nx-1)*(Nt+1)+n+1]-w[(2*Nt+1)*Nx+(Nx-1)*(Nt+1)+n]+dt*f_star((w[(2*Nt+1)*Nx+(Nx-1)*(Nt+1)+n+1]-w[(2*Nt+1)*Nx+(Nx-2)*(Nt+1)+n+1])/dx,w[(Nx-1)*(Nt+1)+n])+ep2*(-2*w[(2*Nt+1)*Nx+(Nx-1)*(Nt+1)+n+1]+w[(2*Nt+1)*Nx+(Nx-2)*(Nt+1)+n+1])
    for j in range(2,Nx):
        for n in range(0,Nt):
            # F_rho , F[Nt]->F[Nt*Nx-Nt-1] ************ 2
            FF[(j-1)*Nt+n]=w[(j-1)*(Nt+1)+n+1]-0.5*(w[(j-2)*(Nt+1)+n]+w[j*(Nt+1)+n])+(0.5*dt/dx)*(w[j*(Nt+1)+n]*w[(Nt+1)*Nx+j*Nt+n]-w[(j-2)*(Nt+1)+n]*w[(Nt+1)*Nx+(j-2)*Nt+n])+ep1*(w[j*(Nt+1)+n]-2*w[(j-1)*(Nt+1)+n]+w[(j-2)*(Nt+1)+n])
            # F_u , F[Nt*Nx+Nt]->F[2*Nt*Nx-Nt-1] *********** 5
            FF[(j-1)*Nt+Nt*Nx+n]=w[(Nt+1)*Nx+(j-1)*Nt+n]-f_star_p((w[(2*Nt+1)*Nx+(j-1)*(Nt+1)+n+1]-w[(2*Nt+1)*Nx+(j-2)*(Nt+1)+n+1])/dx,w[(j-1)*(Nt+1)+n])
            # F_V , F[2*Nt*Nx+Nt]->F[3*Nt*Nx-Nt-1] ********* 8
            FF[(j-1)*Nt+2*Nt*Nx+n]=w[(2*Nt+1)*Nx+(j-1)*(Nt+1)+n+1]-w[(2*Nt+1)*Nx+(j-1)*(Nt+1)+n]+dt*f_star((w[(2*Nt+1)*Nx+(j-1)*(Nt+1)+n+1]-w[(2*Nt+1)*Nx+(j-2)*(Nt+1)+n+1])/dx,w[(j-1)*(Nt+1)+n])+ep2*(w[(2*Nt+1)*Nx+j*(Nt+1)+n+1]-2*w[(2*Nt+1)*Nx+(j-1)*(Nt+1)+n+1]+w[(2*Nt+1)*Nx+(j-2)*(Nt+1)+n+1])
        # F_rho_int , F[3*Nt*Nx+1]->F[3*Nt*Nx+Nx-2] ********** 11
        FF[3*Nt*Nx+j-1]=w[(j-1)*(Nt+1)]-(1/dx)*integral(x[j-1],x[j])
        # F_V_ter , F[3*Nt*Nx+Nx+1]->F[3*Nt*Nx+2*Nx-2] ********* 14
        FF[3*Nt*Nx+Nx+j-1]=w[(2*Nt+1)*Nx+(j-1)*(Nt+1)+Nt]-VT(x[j])
    # F_rho_int , F[3*Nt*Nx] ********* 10
    FF[3*Nt*Nx]=w[0]-(1/dx)*integral(x[0],x[1])
    # F_rho_int , F[3*Nt*Nx+Nx-1] ********* 12
    FF[3*Nt*Nx+Nx-1]=w[(Nx-1)*(Nt+1)]-(1/dx)*integral(x[Nx-1],x[Nx])
    # F_V_ter , F[3*Nt*Nx+Nx] *********** 13 
    FF[3*Nt*Nx+Nx]=w[(2*Nt+1)*Nx+Nt]-VT(x[1])
    # F_V_ter , F[3*Nt*Nx+2*Nx-1] ************** 15
    FF[3*Nt*Nx+2*Nx-1]=w[(2*Nt+1)*Nx+(Nx-1)*(Nt+1)+Nt]-VT(x[Nx])
    
    return FF



def jacobian(w): # Ignoring the forward-backward coupling  parts
    J=np.zeros((3*Nt*Nx+2*Nx,3*Nt*Nx+2*Nx))
    for n in range(0,Nt):
        for j in range(1,Nx-1):
            J[j*Nt+n,j*(Nt+1)+n+1]=1
            J[j*Nt+n,j*(Nt+1)+n+Nt+1]=(0.5*dt/dx)*w[(Nt+1)*Nx+(j+1)*Nt+n]-0.5
            J[j*Nt+n,j*(Nt+1)+n-Nt-1]=-(0.5*dt/dx)*w[(Nt+1)*Nx+(j-1)*Nt+n]-0.5
            J[j*Nt+n,(Nt+1)*Nx+j*Nt+n+Nt]=(0.5*dt/dx)*w[(j+1)*(Nt+1)+n]
            J[j*Nt+n,(Nt+1)*Nx+j*Nt+n-Nt]=-(0.5*dt/dx)*w[(j-1)*(Nt+1)+n]
            J[Nt*Nx+j*Nt+n,(Nt+1)*Nx+j*Nt+n]=1
            J[2*Nt*Nx+j*Nt+n,(2*Nt+1)*Nx+j*(Nt+1)+n]=-1
            J[2*Nt*Nx+j*Nt+n,(2*Nt+1)*Nx+j*(Nt+1)+n+1]=1
            
        J[n,n+1]=1
        J[(Nx-1)*Nt+n,(Nx-1)*(Nt+1)+n+1]=1
        J[n,n+Nt+1]=(0.5*dt/dx)*w[(Nt+1)*Nx+n]-0.5
        J[(Nx-1)*Nt+n,(Nx-1)*(Nt+1)+n-Nt-1]=-(0.5*dt/dx)*w[(Nt+1)*Nx+((Nx-1)-1)*Nt+n]-0.5
        J[n,(Nt+1)*Nx+n+Nt]=(0.5*dt/dx)*w[n]
        J[(Nx-1)*Nt+n,(Nt+1)*Nx+(Nx-1)*Nt+n-Nt]=-(0.5*dt/dx)*w[((Nx-1)-1)*(Nt+1)+n]
        J[Nt*Nx+n,(Nt+1)*Nx+n]=1
        J[Nt*Nx+(Nx-1)*Nt+n,(Nt+1)*Nx+(Nx-1)*Nt+n]=1
        J[2*Nt*Nx+n,(2*Nt+1)*Nx+n]=-1
        J[2*Nt*Nx+n,(2*Nt+1)*Nx+n+1]=1
        J[2*Nt*Nx+(Nx-1)*Nt+n,(2*Nt+1)*Nx+(Nx-1)*(Nt+1)+n]=-1
        J[2*Nt*Nx+(Nx-1)*Nt+n,(2*Nt+1)*Nx+(Nx-1)*(Nt+1)+n+1]=1
    for j in range(0,Nx):
        J[3*Nt*Nx+j,(Nt+1)*j]=1
        J[3*Nt*Nx+Nx+j,(2*Nt+1)*Nx+(Nt+1)*j+Nt]=1
    
    return J


# import pandas as pd
# def get_preconditioner(a):
#     Jac=jacobian(a)
#     Jac1 = csc_matrix(Jac)
#     # df = pd.DataFrame(data=Jac.astype(float))
#     # df.to_csv('/home/amal/Documents/Newton-gmres-solver/prec.csv', sep=' ', header=False, float_format='%.2f', index=False)
#     # the *incomplete LU* decomposition
#     J_ilu = spla.spilu(Jac1)
#     M_x = lambda r: J_ilu.solve(r)
#     M = spla.LinearOperator(Jac.shape, M_x)

#     return M

xglo = None

def get_preconditioner(a):
    ...
    Jac=jacobian(a)
    J1 = csc_matrix(Jac)
    J1_ilu = spla.spilu(J1)  # better spsolve ?

    M = spla.LinearOperator(shape=Jac.shape, matvec=J1_ilu.solve)

    #.......................................................................
    def xglobal( x, F ):
        """ newton_krylov calls this at each iteration: xglo = x """
        global xglo
        xglo = x.copy()
        print( "update: x %s  F %s " % (np.sum(x), np.sum(F)) )  # test
            # nu.asum: array summary, size min av max

    M.update = xglobal

    return M
    


def interpol(Nt,Nt_mul,Nx,Nx_mul,w): # 1D interpolation
    
    """" Go from a coarse grid Nt*Nx to a finer grid spacing (Nt_mul*Nt)*(Nx_mul*Nx) """""

    n=w.shape[0] # n=3Nt*Nx+2Nx
    i = np.indices(w.shape)[0]/(n-1)  # [0, ..., 1]
    new_n = 3*(Nt_mul*Nt)*(Nx_mul*Nx)+2*(Nx_mul*Nx)
    print('n={n}, new_n={new_n}'.format(n=n,new_n=new_n))
    new_i = np.linspace(0, 1, new_n)
    new_w=griddata(i, w, new_i, method="nearest")  # method{‘linear’, ‘nearest’, ‘cubic’}
    
    return Nt_mul*Nt, Nx_mul*Nx, new_w




""" solve in coarse grid """
Nx=30; Nt=120 # spatial-temporal grid sizes 
dx=L/(Nx-1) # spatial step size
dt=min(T/Nt,(CFL*dx)/u_max) # temporal step size
# dt=min(T/Nt,CFL*dx/abs(u_max),EPS*(dx**2)/mu) # temporal step size
# ep1=-mu*dt/(dx**2)  # rho
# ep2=mu*dt/(dx**2) # V
print('Nx={Nx}, Nt={Nt}'.format(Nx=Nx,Nt=Nt))
print('dx={dx}, dt={dt}'.format(dx=round(dx,4),dt=round(dt,4)))
x=np.linspace(0,L,Nx+1)
# t=np.linspace(0,T,Nt+1)
t=np.arange(0,T,dt)
Nt=len(t)
print(Nt)
guess0 = np.zeros(3*Nt*Nx+2*Nx)
t0 = time.process_time()   ###
prec=get_preconditioner(guess0)
t1 = time.process_time()   ###
print("Time spent (anal_precond) :",t1-t0)
t0 = time.process_time()   ###
sol0 = newton_krylov(F, guess0, method='gmres', verbose=1, inner_M=prec)
t1 = time.process_time()   ###
print("Time spent (gmres) :",t1-t0)
np.savetxt('Sol0_LWR_T3_N1.dat', sol0)
# print('sol0=',sol0)
Nx_list.append(Nx)
Nt_list.append(Nt)

""" solve in finer grid 1 """
# Nt=60; Nx=240
# sol0=np.loadtxt('/home/amal/Documents/Newton-gmres-solver/Sol0_LWR_T3_N1.dat')
Nxx=2*Nx; Ntt=2*Nt
dx=L/Nxx # spatial step size
dt=min(T/Ntt,CFL*dx/abs(u_max)) # temporal step size
# dt=min(T/Nt,CFL*dx/abs(u_max),EPS*(dx**2)/mu) # temporal step size
# ep1=-mu*dt/(dx**2)  # rho
# ep2=mu*dt/(dx**2) # V
x=np.linspace(0,L,Nxx+1)
t=np.arange(0,T,dt)
Nt=int(len(t)/2)
Nt, Nx, guess1=interpol(Nt,2,Nx,2,sol0)
np.savetxt('Guess1_LWR_T3_N1.dat', guess1)
print('Nx={Nx}, Nt={Nt}'.format(Nx=Nx,Nt=Nt))
print('dx={dx}, dt={dt}'.format(dx=round(dx,3),dt=round(dt,3)))
t0 = time.process_time()   ###
# guess= np.zeros(3*Nt*Nx+2*Nx)
prec=get_preconditioner(guess1)
t1 = time.process_time()   ###
print("Time spent (jax_precond) :",t1-t0)
t0 = time.process_time()   ###
sol1 = newton_krylov(F, guess1, method='gmres', verbose=1, inner_M=prec)
t1 = time.process_time()   ###
print("Time spent (gmres) :",t1-t0)
np.savetxt('Sol1_LWR_T3_N1.dat', sol1)
# print('sol1=',sol1)
Nx_list.append(Nx)
Nt_list.append(Nt)

""" solve in finer grid 2 """
# Nt=120; Nx=480
# sol1=np.loadtxt('/home/amal/Documents/Newton-gmres-solver/Sol1_LWR_T3_N1.dat')
Nxx=2*Nx; Ntt=2*Nt
dx=L/Nxx # spatial step size
dt=min(T/Ntt,CFL*dx/abs(u_max)) # temporal step size
# dt=min(T/Nt,CFL*dx/abs(u_max),EPS*(dx**2)/mu) # temporal step size
# ep1=-mu*dt/(dx**2)  # rho
# ep2=mu*dt/(dx**2) # V
x=np.linspace(0,L,Nxx+1)
t=np.arange(0,T,dt)
Nt=int(len(t)/2)
Nt, Nx, guess2=interpol(Nt,2,Nx,2,sol1)
np.savetxt('Guess2_LWR_T3_N1.dat', guess2)
print('Nx={Nx}, Nt={Nt}'.format(Nx=Nx,Nt=Nt))
print('dx={dx}, dt={dt}'.format(dx=round(dx,3),dt=round(dt,3)))
t0 = time.process_time()   ###
prec=get_preconditioner(guess2)
t1 = time.process_time()   ###
print("Time spent (jax_precond) :",t1-t0)
t0 = time.process_time()   ###
sol2 = newton_krylov(F, guess2, method='gmres', verbose=1, inner_M=prec)
t1 = time.process_time()   ###
print("Time spent (gmres) :",t1-t0)
np.savetxt('Sol2_LWR_T3_N1.dat', sol2)
# print('sol2=',sol2)
Nx_list.append(Nx)
Nt_list.append(Nt)

""" solve in finer grid 3 """
# Nt=240; Nx=960
# sol2=np.loadtxt('/home/amal/Documents/Newton-gmres-solver/Sol2_LWR_T3_N1.dat')
Nxx=2*Nx; Ntt=2*Nt
dx=L/Nxx # spatial step size
dt=min(T/Ntt,CFL*dx/abs(u_max)) # temporal step size
# dt=min(T/Nt,CFL*dx/abs(u_max),EPS*(dx**2)/mu) # temporal step size
# ep1=-mu*dt/(dx**2)  # rho
# ep2=mu*dt/(dx**2) # V
x=np.linspace(0,L,Nxx+1)
t=np.arange(0,T,dt)
Nt=int(len(t)/2)
Nt, Nx, guess3=interpol(Nt,2,Nx,2,sol2)
np.savetxt('Guess3_LWR_T3_N1.dat', guess3)
print('Nx={Nx}, Nt={Nt}'.format(Nx=Nx,Nt=Nt))
print('dx={dx}, dt={dt}'.format(dx=round(dx,3),dt=round(dt,3)))
t0 = time.process_time()   ###
prec=get_preconditioner(guess3)
t1 = time.process_time()   ###
print("Time spent (jax_precond) :",t1-t0)
t0 = time.process_time()   ###
sol3 = newton_krylov(F, guess3, method='gmres', verbose=1, inner_M=prec)
t1 = time.process_time()   ###
print("Time spent (gmres) :",t1-t0)
np.savetxt('Sol3_LWR_T3_N1.dat', sol3)
# # print('sol3=',sol3)
Nx_list.append(Nx)
Nt_list.append(Nt)

np.savetxt('Nx_Nt_LWR_T3_N1.dat', [Nx_list,Nt_list])




