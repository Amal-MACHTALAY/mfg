# -*- coding: utf-8 -*-
"""
Created on Mon May 10 02:25:47 2021

@author: MACHTALAY AMAL
"""
import numpy as np
from scipy import integrate
import numdifftools as nd
import scipy.sparse.linalg as spla
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def U(rho): # Greenshields desired speed
    return u_max*(1-rho/rho_jam)

def f_mfg(u,r):
    if costf=="LWR":
        return 0.5*((U(r)-u)**2) # MFG-LWR
    if costf=="Sep":
        return 0.5*((u/u_max)**2)-(u/u_max)+(r/rho_jam) # MFG-Separable
    if costf=="NonSep":
        return 0.5*((u/u_max)**2)-(u/u_max)+((u*r)/(u_max*rho_jam)) # MFG-NonSeparable

def f_star(p,r,u): # p=Vx
    if costf=="LWR":
        return -0.5*(p**2)+U(r)*p # MFG-LWR
    if costf=="Sep":
        return f_star_p(p,r)*p+f_mfg(f_star_p(p,r),r) # MFG-Separable
    if costf=="NonSep":
        return f_star_p(p,r)*p+f_mfg(f_star_p(p,r),r) # MFG-NonSeparable

def f_star_p(p,r): # 0<=u<=u_max
    if costf=="LWR":
#         return max(min(U(r)-p,u_max),0) # MFG-LWR 
        return U(r)-p # MFG-LWR
    if costf=="Sep":
        return max(min(u_max*(1-p*u_max),u_max),0) # MFG-Separable
    if costf=="NonSep":
        return max(min(u_max*(1-r/rho_jam-u_max*p),u_max),0) # MFG-NonSeparable

def rho_int(s): # initial density
    return rho_a+(rho_b-rho_a)*np.exp(-0.5*((s-0.5*L)/gama)**2) # 0<=rho<=rho_jam

def VT(a): # Terminal cost
    return 0

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
        FF[2*Nt*Nx-Nt+n]=w[2*Nt*Nx+Nx-Nt+n]-f_star_p((w[3*Nt*Nx+2*Nx-Nt+n]-w[3*Nt*Nx+2*Nx-2*Nt+n-1])/dx,w[Nt*Nx+Nx-Nt+n-1])
        # F_V , F[2*Nt*Nx]->F[2*Nt*Nx+Nt-1] *********** 7
        FF[2*Nt*Nx+n]=w[(2*Nt+1)*Nx+n+1]-w[(2*Nt+1)*Nx+n]+dt*f_star(w[(2*Nt+1)*Nx+n+1]/dx,w[n],w[Nt*Nx+Nx+n])+ep2*(w[(2*Nt+1)*Nx+Nt+n+2]-2*w[(2*Nt+1)*Nx+n+1])
        # F_V , F[3*Nt*Nx-Nt]->F[3*Nt*Nx-1] ********** 9
        FF[3*Nt*Nx-Nt+n]=w[(2*Nt+1)*Nx+(Nx-1)*(Nt+1)+n+1]-w[(2*Nt+1)*Nx+(Nx-1)*(Nt+1)+n]+dt*f_star((w[(2*Nt+1)*Nx+(Nx-1)*(Nt+1)+n+1]-w[(2*Nt+1)*Nx+(Nx-2)*(Nt+1)+n+1])/dx,w[(Nx-1)*(Nt+1)+n],w[2*Nt*Nx+Nx-Nt+n])+ep2*(-2*w[(2*Nt+1)*Nx+(Nx-1)*(Nt+1)+n+1]+w[(2*Nt+1)*Nx+(Nx-2)*(Nt+1)+n+1])
    for j in range(2,Nx):
        for n in range(0,Nt):
            # F_rho , F[Nt]->F[Nt*Nx-Nt-1] ************ 2
            FF[(j-1)*Nt+n]=w[(j-1)*(Nt+1)+n+1]-0.5*(w[(j-2)*(Nt+1)+n]+w[j*(Nt+1)+n])+(0.5*dt/dx)*(w[j*(Nt+1)+n]*w[(Nt+1)*Nx+j*Nt+n]-w[(j-2)*(Nt+1)+n]*w[(Nt+1)*Nx+(j-2)*Nt+n])+ep1*(w[j*(Nt+1)+n]-2*w[(j-1)*(Nt+1)+n]+w[(j-2)*(Nt+1)+n])
            # F_u , F[Nt*Nx+Nt]->F[2*Nt*Nx-Nt-1] *********** 5
            FF[(j-1)*Nt+Nt*Nx+n]=w[(Nt+1)*Nx+(j-1)*Nt+n]-f_star_p((w[(2*Nt+1)*Nx+(j-1)*(Nt+1)+n+1]-w[(2*Nt+1)*Nx+(j-2)*(Nt+1)+n+1])/dx,w[(j-1)*(Nt+1)+n])
            # F_V , F[2*Nt*Nx+Nt]->F[3*Nt*Nx-Nt-1] ********* 8
            FF[(j-1)*Nt+2*Nt*Nx+n]=w[(2*Nt+1)*Nx+(j-1)*(Nt+1)+n+1]-w[(2*Nt+1)*Nx+(j-1)*(Nt+1)+n]+dt*f_star((w[(2*Nt+1)*Nx+(j-1)*(Nt+1)+n+1]-w[(2*Nt+1)*Nx+(j-2)*(Nt+1)+n+1])/dx,w[(j-1)*(Nt+1)+n],w[(Nt+1)*Nx+(j-1)*Nt+n])+ep2*(w[(2*Nt+1)*Nx+j*(Nt+1)+n+1]-2*w[(2*Nt+1)*Nx+(j-1)*(Nt+1)+n+1]+w[(2*Nt+1)*Nx+(j-2)*(Nt+1)+n+1])
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

def Fapp(w): # Ignoring the forward-backward coupling  parts
    # FF=[F_rho,F_u,F_V,F_rho_int,F_V_ter], F_rho:0->Nt*Nx-1, F_u:Nt*Nx->2*Nt*Nx-1, F_V:2*Nt*Nx->3*Nt*Nx-1, F_rho_int:3*Nt*Nx->3*Nt*Nx+Nx-1, F_V_ter:3*Nt*Nx+Nx->3*Nt*Nx+2*Nx-1
    FF=np.zeros(3*Nt*Nx+2*Nx)
    for n in range(0,Nt):
        # F_rho , F[0]->F[Nt-1] ************** 1
        FF[n]=w[n+1]-0.5*w[n+Nt+1]+(0.5*dt/dx)*w[n+Nt+1]*w[n+(Nt+1)*Nx+Nt]+ep1*(w[n+Nt+1]-2*w[n])
        # F_rho , F[Nt*Nx-Nt]->F[Nt*Nx-1] ********** 3
        FF[Nt*(Nx-1)+n]=w[(Nt+1)*(Nx-1)+n+1]-0.5*w[(Nt+1)*(Nx-2)+n]-(0.5*dt/dx)*w[(Nt+1)*(Nx-2)+n]*w[(Nt+1)*Nx+(Nx-2)*Nt+n]+ep1*(-2*w[(Nt+1)*(Nx-1)+n]+w[(Nt+1)*(Nx-2)+n])
        # F_u , F[Nt*Nx]->F[Nt*Nx+Nt-1] *********** 4
        FF[Nt*Nx+n]=w[(Nt+1)*Nx+n]
        # F_u , F[2*Nt*Nx-Nt]->F[2*Nt*Nx-1] ********* 6
        FF[2*Nt*Nx-Nt+n]=w[2*Nt*Nx+Nx-Nt+n]
        # F_V , F[2*Nt*Nx]->F[2*Nt*Nx+Nt-1] *********** 7
        FF[2*Nt*Nx+n]=w[(2*Nt+1)*Nx+n+1]-w[(2*Nt+1)*Nx+n]+ep2*(w[(2*Nt+1)*Nx+Nt+n+2]-2*w[(2*Nt+1)*Nx+n+1])
        # F_V , F[3*Nt*Nx-Nt]->F[3*Nt*Nx-1] ********** 9
        FF[3*Nt*Nx-Nt+n]=w[(2*Nt+1)*Nx+(Nx-1)*(Nt+1)+n+1]-w[(2*Nt+1)*Nx+(Nx-1)*(Nt+1)+n]+ep2*(-2*w[(2*Nt+1)*Nx+(Nx-1)*(Nt+1)+n+1]+w[(2*Nt+1)*Nx+(Nx-2)*(Nt+1)+n+1])
    for j in range(2,Nx):
        for n in range(0,Nt):
            # F_rho , F[Nt]->F[Nt*Nx-Nt-1] ************ 2
            FF[(j-1)*Nt+n]=w[(j-1)*(Nt+1)+n+1]-0.5*(w[(j-2)*(Nt+1)+n]+w[j*(Nt+1)+n])+(0.5*dt/dx)*(w[j*(Nt+1)+n]*w[(Nt+1)*Nx+j*Nt+n]-w[(j-2)*(Nt+1)+n]*w[(Nt+1)*Nx+(j-2)*Nt+n])+ep1*(w[j*(Nt+1)+n]-2*w[(j-1)*(Nt+1)+n]+w[(j-2)*(Nt+1)+n])
            # F_u , F[Nt*Nx+Nt]->F[2*Nt*Nx-Nt-1] *********** 5
            FF[(j-1)*Nt+Nt*Nx+n]=w[(Nt+1)*Nx+(j-1)*Nt+n]
            # F_V , F[2*Nt*Nx+Nt]->F[3*Nt*Nx-Nt-1] ********* 8
            FF[(j-1)*Nt+2*Nt*Nx+n]=w[(2*Nt+1)*Nx+(j-1)*(Nt+1)+n+1]-w[(2*Nt+1)*Nx+(j-1)*(Nt+1)+n]+ep2*(w[(2*Nt+1)*Nx+j*(Nt+1)+n+1]-2*w[(2*Nt+1)*Nx+(j-1)*(Nt+1)+n+1]+w[(2*Nt+1)*Nx+(j-2)*(Nt+1)+n+1])
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

def gauss_legendre(ordergl,tol):
    """
    Returns nodal abscissas {x} and weights {A} of
    Gauss-Legendre m-point quadrature.
    """
    m = ordergl + 1
    from math import cos,pi
    from numpy import zeros

    def legendre(t,m):
        p0 = 1.0; p1 = t
        for k in range(1,m):
            p = ((2.0*k + 1.0)*t*p1 - k*p0)/(1.0 + k )
            p0 = p1; p1 = p
        dp = m*(p0 - t*p1)/(1.0 - t**2)
        return p1,dp

    A = zeros(m)
    x = zeros(m)
    nRoots = (m + 1)// 2          # Number of non-neg. roots
    for i in range(nRoots):
        t = cos(pi*(i + 0.75)/(m + 0.5))  # Approx. root
        for j in range(30):
            p,dp = legendre(t,m)          # Newton-Raphson
            dt = -p/dp; t = t + dt        # method
            if abs(dt) < tol:
                x[i] = t; x[m-i-1] = -t
                A[i] = 2.0/(1.0 - t**2)/(dp**2) # Eq.(6.25)
                A[m-i-1] = A[i]
                break
    
    return x,A

def integral(a, b):
    x, w = gauss_legendre(50,10e-14)
    G = 0
    for i in range(n):
        G = G + w[i]*rho_int(0.5*(b-a)*x[i]+ 0.5*(b+a))
    G = 0.5*(b-a)*G
    return G


def get_preconditioner(a):
    Jac=nd.Jacobian(Fapp)
    J1=Jac(a)
    # the *incomplete LU* decomposition
    J1_ilu = spla.spilu(J1)
    # matrix-vector product -> LinearOperator 
    M_x = lambda r: J1_ilu.solve(r)
    M = spla.LinearOperator(J1.shape, M_x)
    
    return M
    

def interpol(Nt,Nt_mul,Nx,Nx_mul,w): # 1D interpolation
    
    """" Go from a coarse grid Nt*Nx to a finer grid spacing (Nt_mul*Nt)*(Nx_mul*Nx) """""

    # method 2
    n=w.shape[0] # n=3Nt*Nx+2Nx
    i = np.indices(w.shape)[0]/(n-1)  # [0, ..., 1]
    new_n = 3*(Nt_mul*Nt)*(Nx_mul*Nx)+2*(Nx_mul*Nx)
    print('n={n}, new_n={new_n}'.format(n=n,new_n=new_n))
    new_i = np.linspace(0, 1, new_n)
    new_w=griddata(i, w, new_i, method="linear")  # method{‘linear’, ‘nearest’, ‘cubic’}
    
    return Nt_mul*Nt, Nx_mul*Nx, new_w

def solution(sol,rho,u,V,Q):
    for j in range(1,Nx+1):
        for n in range(0,Nt):
            rho[j,n]=sol[(j-1)*(Nt+1)+n]
            u[j,n]=sol[(Nt+1)*Nx+(j-1)*Nt+n]
            V[j,n]=sol[(2*Nt+1)*Nx+(j-1)*(Nt+1)+n]
            Q[j,n]=rho[j,n]*u[j,n]
        rho[j,Nt]=sol[(j-1)*(Nt+1)+Nt]
        V[j,Nt]=sol[(2*Nt+1)*Nx+(j-1)*(Nt+1)+Nt]
    for n in range(0,Nt+1): # periodic boundary conditions
        rho[0,n]=rho[Nx,n]
        V[0,n]=V[Nx,n]
    for n in range(0,Nt):
        u[0,n]=f_star_p(V[1,n]/dx,rho[0,n])
        Q[0,n]=rho[0,n]*u[0,n]
#     print("rho=",rho)
#     print("u=",u)
#     print("V=",V)
    return 0

def plotting(text,t,x,rho,u,V,Q,Nx_list,Error_list):
    tt, xx = np.meshgrid(t, x)
    fig = plt.figure(figsize=(6, 5), dpi=100)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xx, tt, rho, cmap=cm.viridis)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')
    ax.set_zlabel('density')
    ax.invert_xaxis()
    ax.text2D(0.05, 0.95, text, transform=ax.transAxes)

    plt.figure(figsize=(25, 5))
    plt.subplot(1,4,1)
    plt.plot(x,rho[:,0],label='density')
    plt.plot(x,u[:,0],label='speed')
    plt.plot(x,V[:,0],label='Optimal cost')
    plt.legend()
    plt.grid()
    plt.title("t=0, T={T}".format(T=T))
    plt.xlabel('x')
    plt.subplot(1,4,2)
    plt.plot(x,rho[:,Nt-1],label='density')
    plt.plot(x,u[:,Nt-1],label='speed')
    plt.plot(x,V[:,Nt-1],label='Optimal cost')
    plt.grid()
    plt.legend()
    plt.title("t={t}, T={T}".format(t=round(t[Nt-1],3),T=T))
    plt.xlabel('x')
    plt.subplot(1,4,3)
    plt.plot(rho[:,Nt-1],Q[:,Nt-1],label='flow-density')
    plt.xlabel('density')
    plt.ylabel('Flow')
    plt.grid()
    plt.title("Fundamental diagram (T={T})".format(T=T))
    plt.subplot(1,4,4)
    plt.plot(Nx_list,Error_list,label='MFG-LWR')
    plt.xlabel('Spatial grid size')
    plt.ylabel('error')
    plt.grid()
    plt.title("convergence of solution algorithm")
    return 0

def convergence(guess,sol,o):
    rho=np.zeros((Nx+1,Nt+1))
    u=np.zeros((Nx+1,Nt))
    V=np.zeros((Nx+1,Nt+1))
    Q=np.zeros((Nx+1,Nt))
    solution(guess,rho,u,V,Q)
    rho_LWR=np.zeros((Nx+1,Nt+1))
    u_LWR=np.zeros((Nx+1,Nt))
    V_LWR=np.zeros((Nx+1,Nt+1))
    Q_LWR=np.zeros((Nx+1,Nt))
    solution(sol,rho_LWR,u_LWR,V_LWR,Q_LWR)
    error=np.linalg.norm(rho_LWR-rho,ord=o)/np.linalg.norm(rho_LWR,ord=o)+np.linalg.norm(u_LWR-u,ord=o)/np.linalg.norm(u_LWR,ord=o)
    return error
