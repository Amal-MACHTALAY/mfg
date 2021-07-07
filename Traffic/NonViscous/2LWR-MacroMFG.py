#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 16:55:07 2021

@author: amal
"""

import numpy as np
from scipy import integrate
import numdifftools as nd
from scipy.optimize.nonlin import newton_krylov
import scipy.sparse.linalg as spla
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time

''' inputs '''
''' population 1 : cars
    population 2 : trucks '''
T=3.0 # horizon length 
# number of vehicles : 2N
N=20 
# average length of the vehicles in the j-th population
l1=1; l2=3; l=l1+l2
# free flow speed
u1_max=1.0; u2_max=0.6
# jam density
rho1_jam=1.0; rho2_jam=1.0 
L=2*N # road length
CFL=0.75    # CFL<1
# rho_a=0.05; rho_b=0.95; gama=0.1
rho_a=0.2; rho_b=0.8; gama=0.15*L
# """ Non-viscous solution"""
ep1=0.0  # rho
ep2=0.0  # V
# """ Viscous coeff. """
# EPS=0.45
# mu=0.1 # viscosity coefficient 
beta=1

''' functions '''
# Greenshields desired speed
def U(u_max,rho1,rho2): # u_max= u1_max or u2_max
    return u_max*(1-(rho1/rho1_jam)*(l1/l)-(rho2/rho2_jam)*(l2/l))

# Cost functional
def f_mfg(u,rho1,rho2): # u= u1 or u2
    return 0.5*((U(rho1,rho2)-u)**2) # 2-MFG-LWR

def f_star_p(u_max,p,rho1,rho2): # # u_max= u1_max or u2_max
    return U(u_max,rho1,rho2)-p 
    
def f_star(u_max,p,rho1,rho2): # p=Vx
    return -0.5*(p**2)+U(u_max,rho1,rho2)*p 

def integral(a,b): 
    x2 = lambda x: rho_int(x)
    I=integrate.quad(x2, a, b)
    return I[0]

def rho_int(s): # initial density
    return rho_a+(rho_b-rho_a)*np.exp(-0.5*((s-0.5*L)/gama)**2) # 0<=rho<=rho_jam

def VT(a): # Terminal cost
    return 0.0

def F(w):
    Nxx=2*Nx
    FF=np.zeros(3*Nt*Nxx+2*Nxx)
    for n in range(0,Nt): 
        # F_rho1 , F[0]->F[Nt-1] ************** 1 (1)
        FF[n]=w[n+1]-0.5*w[n+Nt+1]+(0.5*dt/dx)*w[n+Nt+1]*w[n+(Nt+1)*Nxx+Nt]+ep1*(w[n+Nt+1]-2*w[n])
        # F_rho1 , F[Nt*Nx-Nt]->F[Nt*Nx-1] ********** 3 (1)
        FF[Nt*(Nx-1)+n]=w[(Nt+1)*(Nx-1)+n+1]-0.5*w[(Nt+1)*(Nx-2)+n]-(0.5*dt/dx)*w[(Nt+1)*(Nx-2)+n]*w[(Nt+1)*Nxx+(Nx-2)*Nt+n]+ep1*(-2*w[(Nt+1)*(Nx-1)+n]+w[(Nt+1)*(Nx-2)+n])
        # F_rho2 , F[Nt*Nx]->F[Nt*Nx+Nt-1] ************** 4 (2)
        FF[Nt*Nx+n]=w[Nx*(Nt+1)+n+1]-0.5*w[(Nx+1)*(Nt+1)+n]+(0.5*dt/dx)*w[(Nx+1)*(Nt+1)+n]*w[n+(Nt+1)*Nxx+(Nx+1)*Nt]+ep1*(w[(Nx+1)*(Nt+1)+n]-2*w[Nx*(Nt+1)+n])
        # F_rho2 , F[2*Nt*Nx-Nt]->F[2*Nt*Nx-1] ********** 6 (2)
        FF[Nt*(Nxx-1)+n]=w[(Nt+1)*(Nxx-1)+n+1]-0.5*w[(Nt+1)*(Nxx-2)+n]-(0.5*dt/dx)*w[(Nt+1)*(Nxx-2)+n]*w[(Nt+1)*Nxx+(Nxx-2)*Nt+n]+ep1*(-2*w[(Nt+1)*(Nxx-1)+n]+w[(Nt+1)*(Nxx-2)+n])
        # F_u1 , F[2*Nt*Nx]->F[2*Nt*Nx+Nt-1] *********** 7 (1)
        FF[Nt*Nxx+n]=w[(Nt+1)*Nxx+n]-beta*f_star_p(u1_max,w[(2*Nt+1)*Nxx+n+1]/dx,w[n],w[Nx*(Nt+1)+n])
        # F_u1 , F[3*Nt*Nx-Nt]->F[3*Nt*Nx-1] ********* 9 (1)
        FF[(Nx-1)*Nt+Nt*Nxx+n]=w[(Nt+1)*Nxx+(Nx-1)*Nt+n]-beta*f_star_p(u1_max,(w[(2*Nt+1)*Nxx+(Nx-1)*(Nt+1)+n+1]-w[(2*Nt+1)*Nxx+(Nx-2)*(Nt+1)+n+1])/dx,w[(Nx-1)*(Nt+1)+n],w[(Nxx-1)*(Nt+1)+n])
        # F_u2 , F[3*Nt*Nx]->F[3*Nt*Nx+Nt-1] *********** 10 (2)
        FF[Nt*Nx+Nt*Nxx+n]=w[(Nt+1)*Nxx+Nt*Nx+n]-beta*f_star_p(u2_max,w[(2*Nt+1)*Nxx+Nx*(Nt+1)+n+1]/dx,w[n],w[Nx*(Nt+1)+n])
        # F_u2 , F[4*Nt*Nx-Nt]->F[4*Nt*Nx-1] ********* 12 (2)
        FF[(Nxx-1)*Nt+Nt*Nxx+n]=w[(Nt+1)*Nxx+(Nxx-1)*Nt+n]-beta*f_star_p(u2_max,(w[(2*Nt+1)*Nxx+(Nxx-1)*(Nt+1)+n+1]-w[(2*Nt+1)*Nxx+(Nxx-2)*(Nt+1)+n+1])/dx,w[(Nx-1)*(Nt+1)+n],w[(Nxx-1)*(Nt+1)+n])
        # F_V1 , F[4*Nt*Nx]->F[4*Nt*Nx+Nt-1] *********** 13 (1)
        FF[2*Nt*Nxx+n]=w[(2*Nt+1)*Nxx+n+1]-w[(2*Nt+1)*Nxx+n]+beta*dt*f_star(u1_max,w[(2*Nt+1)*Nxx+n+1]/dx,w[n],w[Nx*(Nt+1)+n])+ep2*(w[(2*Nt+1)*Nxx+Nt+n+2]-2*w[(2*Nt+1)*Nxx+n+1])
        # F_V1 , F[5*Nt*Nx-Nt]->F[5*Nt*Nx-1] ********** 15 (1)
        FF[5*Nt*Nx-Nt+n]=w[(2*Nt+1)*Nxx+(Nx-1)*(Nt+1)+n+1]-w[(2*Nt+1)*Nxx+(Nx-1)*(Nt+1)+n]+beta*dt*f_star(u1_max,(w[(2*Nt+1)*Nxx+(Nx-1)*(Nt+1)+n+1]-w[(2*Nt+1)*Nxx+(Nx-2)*(Nt+1)+n+1])/dx,w[(Nx-1)*(Nt+1)+n],w[(Nxx-1)*(Nt+1)+n])+ep2*(-2*w[(2*Nt+1)*Nxx+(Nx-1)*(Nt+1)+n+1]+w[(2*Nt+1)*Nxx+(Nx-2)*(Nt+1)+n+1])
        # F_V2 , F[5*Nt*Nx]->F[5*Nt*Nx+Nt-1] *********** 16 (2)
        FF[Nt*Nx+2*Nt*Nxx+n]=w[(2*Nt+1)*Nxx+Nx*(Nt+1)+n+1]-w[(2*Nt+1)*Nxx+Nx*(Nt+1)+n]+beta*dt*f_star(u2_max,w[(2*Nt+1)*Nxx+Nx*(Nt+1)+n+1]/dx,w[n],w[Nx*(Nt+1)+n])+ep2*(w[(2*Nt+1)*Nxx+(Nx+1)*(Nt+1)+n+1]-2*w[(2*Nt+1)*Nxx+Nx*(Nt+1)+n+1])
        # F_V2 , F[6*Nt*Nx-Nt]->F[6*Nt*Nx-1] ********** 18 (2)
        FF[2*Nt*Nxx+(Nxx-1)*Nt+n]=w[(2*Nt+1)*Nxx+(Nxx-1)*(Nt+1)+n+1]-w[(2*Nt+1)*Nxx+(Nxx-1)*(Nt+1)+n]+beta*dt*f_star(u2_max,(w[(2*Nt+1)*Nxx+(Nxx-1)*(Nt+1)+n+1]-w[(2*Nt+1)*Nxx+(Nxx-2)*(Nt+1)+n+1])/dx,w[(Nx-1)*(Nt+1)+n],w[(Nxx-1)*(Nt+1)+n])+ep2*(-2*w[(2*Nt+1)*Nxx+(Nxx-1)*(Nt+1)+n+1]+w[(2*Nt+1)*Nxx+(Nxx-2)*(Nt+1)+n+1])
        
        
    for j in range(2,Nx):
        for n in range(0,Nt):
            # F_rho1 , F[Nt]->F[Nt*Nx-Nt-1] ************ 2 (1)
            FF[(j-1)*Nt+n]=w[(j-1)*(Nt+1)+n+1]-0.5*(w[(j-2)*(Nt+1)+n]+w[j*(Nt+1)+n])+(0.5*dt/dx)*(w[j*(Nt+1)+n]*w[(Nt+1)*Nxx+j*Nt+n]-w[(j-2)*(Nt+1)+n]*w[(Nt+1)*Nxx+(j-2)*Nt+n])+ep1*(w[j*(Nt+1)+n]-2*w[(j-1)*(Nt+1)+n]+w[(j-2)*(Nt+1)+n])      
            # F_u1 , F[2*Nt*Nx+Nt]->F[3*Nt*Nx-Nt-1] *********** 8 (1)
            FF[(j-1)*Nt+Nt*Nxx+n]=w[(Nt+1)*Nxx+(j-1)*Nt+n]-beta*f_star_p(u1_max,(w[(2*Nt+1)*Nxx+(j-1)*(Nt+1)+n+1]-w[(2*Nt+1)*Nxx+(j-2)*(Nt+1)+n+1])/dx,w[(j-1)*(Nt+1)+n],w[(Nx+j-1)*(Nt+1)+n])
            # F_V1 , F[4*Nt*Nx+Nt]->F[5*Nt*Nx-Nt-1] ********* 14 (1)
            FF[(j-1)*Nt+2*Nt*Nxx+n]=w[(2*Nt+1)*Nxx+(j-1)*(Nt+1)+n+1]-w[(2*Nt+1)*Nxx+(j-1)*(Nt+1)+n]+beta*dt*f_star(u1_max,(w[(2*Nt+1)*Nxx+(j-1)*(Nt+1)+n+1]-w[(2*Nt+1)*Nxx+(j-2)*(Nt+1)+n+1])/dx,w[(j-1)*(Nt+1)+n],w[(Nx+j-1)*(Nt+1)+n])+ep2*(w[(2*Nt+1)*Nxx+j*(Nt+1)+n+1]-2*w[(2*Nt+1)*Nxx+(j-1)*(Nt+1)+n+1]+w[(2*Nt+1)*Nxx+(j-2)*(Nt+1)+n+1])
        
        
        # F_rho1_int , F[6*Nt*Nx+1]->F[6*Nt*Nx+Nx-2] ********** 20 (1)
        FF[6*Nt*Nx+j-1]=w[(j-1)*(Nt+1)]-(1/dx)*integral(x[j-1],x[j])
        # F_V1_ter , F[3*Nt*Nx+Nx+1]->F[3*Nt*Nx+2*Nx-2] ********* 26 (1)
        FF[3*Nt*Nx+Nx+j-1]=w[(2*Nt+1)*Nx+(j-1)*(Nt+1)+Nt]-VT(x[j])
    # F_rho1_int , F[6*Nt*Nx] ********* 19 (1)
    FF[6*Nt*Nx]=w[0]-(1/dx)*integral(x[0],x[1])
    # F_rho1_int , F[6*Nt*Nx+Nx-1] ********* 21 (1)
    FF[6*Nt*Nx+Nx-1]=w[(Nx-1)*(Nt+1)]-(1/dx)*integral(x[Nx-1],x[Nx])
    
    # F_V1_ter , F[3*Nt*Nx+Nx] *********** 25 (1)
    FF[3*Nt*Nx+Nx]=w[(2*Nt+1)*Nx+Nt]-VT(x[1])
    # F_V1_ter , F[3*Nt*Nx+2*Nx-1] ************** 27 (1)
    FF[3*Nt*Nx+2*Nx-1]=w[(2*Nt+1)*Nx+(Nx-1)*(Nt+1)+Nt]-VT(x[Nx])
        
    for j in range(Nx+2,Nxx):
        for n in range(0,Nt):
            # F_rho2 , F[Nt*Nx+Nt]->F[2*Nt*Nx-Nt-1] ************ 5 (2)
            FF[(j-1)*Nt+n]=w[(j-1)*(Nt+1)+n+1]-0.5*(w[(j-2)*(Nt+1)+n]+w[j*(Nt+1)+n])+(0.5*dt/dx)*(w[j*(Nt+1)+n]*w[(Nt+1)*Nxx+j*Nt+n]-w[(j-2)*(Nt+1)+n]*w[(Nt+1)*Nxx+(j-2)*Nt+n])+ep1*(w[j*(Nt+1)+n]-2*w[(j-1)*(Nt+1)+n]+w[(j-2)*(Nt+1)+n])      
            # F_u2 , F[3*Nt*Nx+Nt]->F[4*Nt*Nx-Nt-1] *********** 11 (2)
            FF[(j-1)*Nt+Nt*Nxx+n]=w[(Nt+1)*Nxx+(j-1)*Nt+n]-beta*f_star_p(u2_max,(w[(2*Nt+1)*Nxx+(j-1)*(Nt+1)+n+1]-w[(2*Nt+1)*Nxx+(j-2)*(Nt+1)+n+1])/dx,w[(j-Nx-1)*(Nt+1)+n],w[(j-1)*(Nt+1)+n])
            # F_V2 , F[5*Nt*Nx+Nt]->F[6*Nt*Nx-Nt-1] ********* 17 (2)
            FF[(j-1)*Nt+2*Nt*Nxx+n]=w[(2*Nt+1)*Nxx+(j-1)*(Nt+1)+n+1]-w[(2*Nt+1)*Nxx+(j-1)*(Nt+1)+n]+beta*dt*f_star(u2_max,(w[(2*Nt+1)*Nxx+(j-1)*(Nt+1)+n+1]-w[(2*Nt+1)*Nxx+(j-2)*(Nt+1)+n+1])/dx,w[(j-Nx-1)*(Nt+1)+n],w[(j-1)*(Nt+1)+n])+ep2*(w[(2*Nt+1)*Nxx+j*(Nt+1)+n+1]-2*w[(2*Nt+1)*Nxx+(j-1)*(Nt+1)+n+1]+w[(2*Nt+1)*Nxx+(j-2)*(Nt+1)+n+1])
        
        
        # F_rho2_int , F[6*Nt*Nx+Nx]->F[6*Nt*Nx+2*Nx-2] ********** 23 (2)
        FF[3*Nt*Nx+j-1]=w[(j-1)*(Nt+1)]-(1/dx)*integral(x[j-1],x[j])
        # F_V2_ter , F[3*Nt*Nx+Nx+1]->F[3*Nt*Nx+2*Nx-2] ********* 29 (2)
        FF[3*Nt*Nx+Nx+j-1]=w[(2*Nt+1)*Nx+(j-1)*(Nt+1)+Nt]-VT(x[j])
    # F_rho2_int , F[6*Nt*Nx+Nx-1] ********* 22 (2)
    FF[6*Nt*Nx+Nx-1]=w[Nx]-(1/dx)*integral(x[Nx-1],x[Nx])
    # F_rho2_int , F[6*Nt*Nx+Nx-1] ********* 24 (2)
    FF[3*Nt*Nx+Nx-1]=w[(Nx-1)*(Nt+1)]-(1/dx)*integral(x[Nx-1],x[Nx])
    
    # F_V2_ter , F[3*Nt*Nx+Nx] *********** 28 (2) 
    FF[3*Nt*Nx+Nx]=w[(2*Nt+1)*Nx+Nt]-VT(x[1])
    # F_V2_ter , F[3*Nt*Nx+2*Nx-1] ************** 30 (2)
    FF[3*Nt*Nx+2*Nx-1]=w[(2*Nt+1)*Nx+(Nx-1)*(Nt+1)+Nt]-VT(x[Nx])
    
    return FF


def get_preconditioner(a):
    beta=0 # Ignoring the forward-backward coupling  parts
    Jac=nd.Jacobian(F)
    J1=Jac(a)
    # the *incomplete LU* decomposition
    J1_ilu = spla.spilu(J1)
    # matrix-vector product -> LinearOperator 
    M_x = lambda r: J1_ilu.solve(r)
    M = spla.LinearOperator(J1.shape, M_x)

    return M
    

def interpol(Nt,Nt_mul,Nx,Nx_mul,w): # 1D interpolation
    
    """" Go from a coarse grid Nt*Nx to a finer grid spacing (Nt_mul*Nt)*(Nx_mul*Nx) """""

    n=w.shape[0] # n=3Nt*Nx+2Nx
    i = np.indices(w.shape)[0]/(n-1)  # [0, ..., 1]
    new_n = 3*(Nt_mul*Nt)*(Nx_mul*Nx)+2*(Nx_mul*Nx)
    print('n={n}, new_n={new_n}'.format(n=n,new_n=new_n))
    new_i = np.linspace(0, 1, new_n)
    new_w=griddata(i, w, new_i, method="cubic")  # method{‘linear’, ‘nearest’, ‘cubic’}
    
    return Nt_mul*Nt, Nx_mul*Nx, new_w

def solution(sol,rho1,u1,V1,Q1,rho2,u2,V2,Q2):
    for j in range(1,Nx+1):
        for n in range(0,Nt):
            rho1[j,n]=sol[(j-1)*(Nt+1)+n]
            u1[j,n]=sol[(Nt+1)*2*Nx+(j-1)*Nt+n]
            V1[j,n]=sol[(2*Nt+1)*2*Nx+(j-1)*(Nt+1)+n]
            Q1[j,n]=rho1[j,n]*u1[j,n]
        rho1[j,Nt]=sol[(j-1)*(Nt+1)+Nt]
        V1[j,Nt]=sol[(2*Nt+1)*2*Nx+(j-1)*(Nt+1)+Nt]
    for j in range(Nx+1,2*Nx+1):
        for n in range(0,Nt):
            rho2[j-Nx,n]=sol[(j-1)*(Nt+1)+n]
            u2[j-Nx,n]=sol[(Nt+1)*2*Nx+(j-1)*Nt+n]
            V2[j-Nx,n]=sol[(2*Nt+1)*2*Nx+(j-1)*(Nt+1)+n]
            Q2[j-Nx,n]=rho2[j,n]*u2[j,n]
        rho2[j-Nx,Nt]=sol[(j-1)*(Nt+1)+Nt]
        V2[j-Nx,Nt]=sol[(2*Nt+1)*2*Nx+(j-1)*(Nt+1)+Nt]
    for n in range(0,Nt+1): # periodic boundary conditions
        rho1[0,n]=rho1[Nx,n]
        V1[0,n]=V1[Nx,n]
        rho2[0,n]=rho2[Nx,n]
        V2[0,n]=V2[Nx,n]
    for n in range(0,Nt):
        u1[0,n]=f_star_p(u1_max,V1[0,n+1]/dx,rho1[0,n],rho2[0,n])
        Q1[0,n]=rho1[0,n]*u1[0,n]
        u2[0,n]=f_star_p(u2_max,V2[0,n+1]/dx,rho1[0,n],rho2[0,n])
        Q2[0,n]=rho2[0,n]*u2[0,n]
    return 0

def plotting(text,t,x,rho1,u1,V1,Q1,rho2,u2,V2,Q2,Nx_list,Error_list,fig1,fig2,fig3,fig4):
    tt, xx = np.meshgrid(t, x)
    fig1 = plt.figure(figsize=(6, 5), dpi=100)
    ax1 = fig1.gca(projection='3d')
    ax1.plot_surface(xx, tt, rho1, cmap=cm.viridis)
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$t$')
    ax1.set_zlabel('density-1')
    ax1.invert_xaxis()
#     ax.text2D(0.05, 0.95, text, transform=ax1.transAxes)
    plt.savefig(fig1)
    
    fig2 = plt.figure(figsize=(6, 5), dpi=100)
    ax2 = fig2.gca(projection='3d')
    ax2.plot_surface(xx, tt, rho2, cmap=cm.viridis)
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$t$')
    ax2.set_zlabel('density-2')
    ax2.invert_xaxis()
#     ax.text2D(0.05, 0.95, text, transform=ax2.transAxes)
    plt.savefig(fig2)

    plt.figure(figsize=(20, 5))
    plt.subplot(1,3,1)
    plt.plot(x,rho1[:,0],'b',label='density-1')
    plt.plot(x,u1[:,0],'g',label='speed-1')
    plt.plot(x,V1[:,0],'r',label='Optimal cost-1')
    plt.plot(x,rho2[:,0],'b+',label='density-2')
    plt.plot(x,u2[:,0],'g+',label='speed-2')
    plt.plot(x,V2[:,0],'r+',label='Optimal cost-2')
    plt.legend()
    plt.grid()
    plt.title("t=0.0")
    plt.xlabel('x')
    plt.subplot(1,3,2)
    plt.plot(x,rho1[:,int(Nt/2)],'b',label='density-1')
    plt.plot(x,u1[:,int(Nt/2)],'g',label='speed-1')
    plt.plot(x,V1[:,int(Nt/2)],'r',label='Optimal cost-1')
    plt.plot(x,rho2[:,int(Nt/2)],'b+',label='density-2')
    plt.plot(x,u2[:,int(Nt/2)],'g+',label='speed-2')
    plt.plot(x,V2[:,int(Nt/2)],'r+',label='Optimal cost-2')
    plt.grid()
    plt.legend()
    plt.title("t={t}".format(t=round(t[int(Nt/2)],3)))
    plt.xlabel('x')
    plt.subplot(1,3,3)
    plt.plot(x,rho1[:,Nt-1],'b',label='density-1')
    plt.plot(x,u1[:,Nt-1],'g',label='speed-1')
    plt.plot(x,V1[:,Nt-1],'r',label='Optimal cost-1')
    plt.plot(x,rho2[:,Nt-1],'b+',label='density-2')
    plt.plot(x,u2[:,Nt-1],'g+',label='speed-2')
    plt.plot(x,V2[:,Nt-1],'r+',label='Optimal cost-2')
    plt.grid()
    plt.legend()
    plt.title("t={t}".format(t=round(t[Nt-1],3)))
    plt.xlabel('x')
#     plt.subplot(2,3,5)
#     plt.plot(Nx_list,Error_list)
#     plt.xlabel('Spatial grid size')
#     plt.ylabel('error')
#     plt.grid()
#     plt.title("convergence of solution algorithm")
    plt.savefig(fig3)
    plt.figure()
    plt.plot(rho1[:,Nt-1],Q1[:,Nt-1],label='flow-density-1')
    plt.plot(rho2[:,Nt-1],Q2[:,Nt-1],label='flow-density-2')
    plt.xlabel('density')
    plt.ylabel('Flow')
    plt.grid()
    plt.title("Fundamental diagram (T={T})".format(T=T))
    plt.savefig(fig4)
    return 0

def convergence(guess,sol,o):
    rho=np.zeros((Nx+1,Nt+1))
    u=np.zeros((Nx+1,Nt))
    V=np.zeros((Nx+1,Nt+1))
    Q=np.zeros((Nx+1,Nt))
    solution(guess,rho,u,V,Q)
    rho_mfg=np.zeros((Nx+1,Nt+1))
    u_mfg=np.zeros((Nx+1,Nt))
    V_mfg=np.zeros((Nx+1,Nt+1))
    Q_mfg=np.zeros((Nx+1,Nt))
    solution(sol,rho_mfg,u_mfg,V_mfg,Q_mfg)
    error=np.linalg.norm(rho_mfg-rho,ord=o)/np.linalg.norm(rho_mfg,ord=o)+np.linalg.norm(u_mfg-u,ord=o)/np.linalg.norm(u_mfg,ord=o)
    return error