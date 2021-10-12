#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 12:13:45 2021

@author: amal
"""

import numpy as np


def get_Hessenberg_matrix(H,m):
    h=np.zeros((m+1,m))
    for s in range(m):
        for k in range(s+2):
            h[k,s]=H[s][k]
    return h

def AxB(X,Y):
    XY=np.zeros((len(X),len(Y[0])))
    for i in range(len(X)):
    # iterate through columns of Y
        for j in range(len(Y[0])):
        # iterate through rows of Y
            for k in range(len(Y)):
                XY[i][j] += X[i][k] * Y[k][j]
    return XY


def get_preconditionner(A):
    # A: Jacobian of fct, Ignoring the forward-backward coupling  parts (beta=1)
    n = len(A)
    
    """ Compute LU Factorization """
    L = np.zeros((n,n),dtype=np.float32)
    for i in range(n):
        L[i,i] = 1
    U = np.zeros((n,n),dtype= np.float32)
    U[:] = A
    for i in range(n):
        for j in range(i+1,n):
            L[j,i] = (1/U[i,i])*U[j,i]
            for k in range(n):
                U[j,k] = U[j,k] - L[j,i]*U[i,k]
         
    """ This E1 is used to find the inverse of U """
    E1 = np.eye(n) 
    for j in range(n-1,-1,-1):
        E1[j,:] = E1[j,:]/U[j,j]
        U[j,:] = U[j,:]/U[j,j]
        for i in range(j-1,-1,-1):
            E1[i,:] = E1[i,:]-E1[j,:]*U[i,j]
            U[i,:] = U[i,:]-U[j,:]*U[i,j]
    
    """ This E2 is used to find the inverse of L """
    E2 = np.eye(n) 
    for j in range(n):
        for i in range(j+1,n):
            E2[i,:] = E2[i,:]-E2[j,:]*L[i,j]
            L[i,:] = L[i,:]-L[j,:]*L[i,j]
            
    return AxB(E1,E2)


def matvec(A,v):
    Av = np.zeros(len(A))
    for i in range(len(A)):
        s = 0.0
        for j in range(len(v)):
              s += A[i,j]*v[j]
        Av[i] = s
    return Av

def scalar(u,v):
    s = 0.0
    for i in range(len(u)):
        s += u[i]*v[i]
    return s

def norm_two(v):
    norm=v[0]**2
    for i in range(1,len(v)):
        norm+=v[i]**2
    return np.sqrt(norm)

def matT(A):
    At=np.zeros((len(A[0]),len(A)))
    for i in range(len(A[0])):
        for j in range(len(A)):
            At[i,j]=A[j,i]
    return At

def vec_asarray(v):
    vv=np.zeros((len(v),len(v[0])))
    for i in range(len(v)):
        for j in range(len(v[0])):
            vv[i,j]=v[i][j]
    return vv
    
def least_squares(A,b):
    # print('A=',A)
    At=matT(A)
    # print('At=',At)
    AtA=AxB(At,A)
    Atb=matvec(At,b)
    return GC(AtA,Atb) 

def GC(A,b):
    """ Conjugate Gradient """
    r=b
    p=b
    x=np.zeros(len(A))
    rr = scalar(r,r)
    while True:
        Ap = matvec(A,p)
        alpha = rr / scalar(Ap,p)
        x = x + alpha*p
        r = r - alpha*Ap
        rr_new = scalar(r,r)
        if np.sqrt(rr_new) < 1e-08:
            break
        p = r + (rr_new/rr)*p
        rr = rr_new
    return x


def gmres(w0, fct, sigma, tol):
    while True :
        r=-fct(w0,beta=1)
        v=[]
        v.append(r/norm_two(r))
        M=get_preconditionner(Jacobian(w0))
        H=[]
        j=0
        while True :
            hh=[]
            xj=matvec(M, v[j])
            q=(fct(w0+sigma*xj,beta=1)-fct(w0,beta=1))/sigma
            vj=q
            for i in range(j+1):
                hij=scalar(q,v[i])
                hh.append(hij)
                vj-=hij*v[i]
            hjpj=norm_two(vj)
            hh.append(hjpj)
            H.append(hh)
            m=j
            if hjpj==0.0 :
                break
            v.append(vj/hjpj)
            if norm_two(fct(w0,beta=1)+q)<=tol:
                break
            j+=1
        h=get_Hessenberg_matrix(H,m)
        # calcul of beta*e1
        beta=np.zeros(m)
        beta[0]=norm_two(r)
        # Minimize for y
        y=least_squares(matT(h),beta)
        # y=np.linalg.lstsq(matT(h),beta,rcond=-1)[0]
        # print('y=',y)
        w0_new=w0+matvec(matT(vec_asarray(v)),y)
        if norm_two(fct(w0,beta=1))<=tol or norm_two(w0-w0_new)<=1e-08 :
            break
        # # Update tolerance 
        # tol=max(0.9*(norm_two(fct(w0_new,beta=1))/norm_two(fct(w0,beta=1)))**2,0.9*tol**2)
        # print(tol)
        w0=w0_new
    
    return w0_new




def quad(f,a,b):
    n=1000
    """ gauss_legendre : Returns nodal abscissas {x} and weights {A} of
    Gauss-Legendre m-point quadrature."""
    m = n + 1
    from math import cos,pi

    def legendre(t,m):
        p0 = 1.0; p1 = t
        for k in range(1,m):
            p = ((2.0*k + 1.0)*t*p1 - k*p0)/(1.0 + k )
            p0 = p1; p1 = p
        dp = m*(p0 - t*p1)/(1.0 - t**2)
        return p1,dp

    w = np.zeros(m)
    x = np.zeros(m)
    nRoots = (m + 1)// 2          # Number of non-neg. roots
    for i in range(nRoots):
        t = cos(pi*(i + 0.75)/(m + 0.5))  # Approx. root
        for j in range(30):
            p,dp = legendre(t,m)          # Newton-Raphson
            dt = -p/dp; t = t + dt        # method
            if abs(dt) < 10e-14:
                x[i] = t; x[m-i-1] = -t
                w[i] = 2.0/(1.0 - t**2)/(dp**2) # Eq.(6.25)
                w[m-i-1] = w[i]
                break
            
    G = 0
    for i in range(n):
        G = G + w[i]*f(0.5*(b-a)*x[i]+ 0.5*(b+a))
    G = 0.5*(b-a)*G
    return G



""" ******************************************************************************************* """
''' inputs '''
T=3.0 # horizon length 
N=1 # number of cars 
u_max=1.0 # free flow speed
rho_jam=1.0 # jam density
L=N # road length
CFL=0.75    # CFL<1
# rho_a=0.05; rho_b=0.95; gama=0.1
rho_a=0.2; rho_b=0.8; gama=0.15*L
ep1=0.0  # rho
ep2=0.0  # V
mu=0.1 # viscosity coefficient 
costf="LWR"
Error_list=[]
Nx_list=[]

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


def rho_int(s): # initial density
    return rho_a+(rho_b-rho_a)*np.exp(-0.5*((s-0.5*L)/gama)**2) # 0<=rho<=rho_jam

def VT(a): # Terminal cost
    return 0.0


def func(w,beta=0):
    dx=L/Nx # spatial step size
    dt=min(T/Nt,CFL*dx/abs(u_max)) # temporal step size
    x=np.linspace(0,L,Nx+1)
    # FF=[F_rho,F_u,F_V,F_rho_int,F_V_ter], F_rho:0->Nt*Nx-1, F_u:Nt*Nx->2*Nt*Nx-1, F_V:2*Nt*Nx->3*Nt*Nx-1, F_rho_int:3*Nt*Nx->3*Nt*Nx+Nx-1, F_V_ter:3*Nt*Nx+Nx->3*Nt*Nx+2*Nx-1
    FF=np.zeros(3*Nt*Nx+2*Nx)
    for n in range(0,Nt):
        # F_rho , F[0]->F[Nt-1] ************** 1
        FF[n]=w[n+1]-0.5*w[n+Nt+1]+(0.5*dt/dx)*w[n+Nt+1]*w[n+(Nt+1)*Nx+Nt]+ep1*(w[n+Nt+1]-2*w[n])
        # F_rho , F[Nt*Nx-Nt]->F[Nt*Nx-1] ********** 3
        FF[Nt*(Nx-1)+n]=w[(Nt+1)*(Nx-1)+n+1]-0.5*w[(Nt+1)*(Nx-2)+n]-(0.5*dt/dx)*w[(Nt+1)*(Nx-2)+n]*w[(Nt+1)*Nx+(Nx-2)*Nt+n]+ep1*(-2*w[(Nt+1)*(Nx-1)+n]+w[(Nt+1)*(Nx-2)+n])
        # F_u , F[Nt*Nx]->F[Nt*Nx+Nt-1] *********** 4
        FF[Nt*Nx+n]=w[(Nt+1)*Nx+n]-beta*f_star_p(w[(2*Nt+1)*Nx+n+1]/dx,w[n])
        # F_u , F[2*Nt*Nx-Nt]->F[2*Nt*Nx-1] ********* 6
        FF[2*Nt*Nx-Nt+n]=w[(Nt+1)*Nx+(Nx-1)*Nt+n]-beta*f_star_p((w[(2*Nt+1)*Nx+(Nx-1)*(Nt+1)+n+1]-w[(2*Nt+1)*Nx+(Nx-2)*(Nt+1)+n+1])/dx,w[(Nx-1)*(Nt+1)+n])
        # F_V , F[2*Nt*Nx]->F[2*Nt*Nx+Nt-1] *********** 7
        FF[2*Nt*Nx+n]=w[(2*Nt+1)*Nx+n+1]-w[(2*Nt+1)*Nx+n]+beta*dt*f_star(w[(2*Nt+1)*Nx+n+1]/dx,w[n])+ep2*(w[(2*Nt+1)*Nx+Nt+n+2]-2*w[(2*Nt+1)*Nx+n+1])
        # F_V , F[3*Nt*Nx-Nt]->F[3*Nt*Nx-1] ********** 9
        FF[3*Nt*Nx-Nt+n]=w[(2*Nt+1)*Nx+(Nx-1)*(Nt+1)+n+1]-w[(2*Nt+1)*Nx+(Nx-1)*(Nt+1)+n]+beta*dt*f_star((w[(2*Nt+1)*Nx+(Nx-1)*(Nt+1)+n+1]-w[(2*Nt+1)*Nx+(Nx-2)*(Nt+1)+n+1])/dx,w[(Nx-1)*(Nt+1)+n])+ep2*(-2*w[(2*Nt+1)*Nx+(Nx-1)*(Nt+1)+n+1]+w[(2*Nt+1)*Nx+(Nx-2)*(Nt+1)+n+1])
    for j in range(2,Nx):
        for n in range(0,Nt):
            # F_rho , F[Nt]->F[Nt*Nx-Nt-1] ************ 2
            FF[(j-1)*Nt+n]=w[(j-1)*(Nt+1)+n+1]-0.5*(w[(j-2)*(Nt+1)+n]+w[j*(Nt+1)+n])+(0.5*dt/dx)*(w[j*(Nt+1)+n]*w[(Nt+1)*Nx+j*Nt+n]-w[(j-2)*(Nt+1)+n]*w[(Nt+1)*Nx+(j-2)*Nt+n])+ep1*(w[j*(Nt+1)+n]-2*w[(j-1)*(Nt+1)+n]+w[(j-2)*(Nt+1)+n])
            # F_u , F[Nt*Nx+Nt]->F[2*Nt*Nx-Nt-1] *********** 5
            FF[(j-1)*Nt+Nt*Nx+n]=w[(Nt+1)*Nx+(j-1)*Nt+n]-beta*f_star_p((w[(2*Nt+1)*Nx+(j-1)*(Nt+1)+n+1]-w[(2*Nt+1)*Nx+(j-2)*(Nt+1)+n+1])/dx,w[(j-1)*(Nt+1)+n])
            # F_V , F[2*Nt*Nx+Nt]->F[3*Nt*Nx-Nt-1] ********* 8
            FF[(j-1)*Nt+2*Nt*Nx+n]=w[(2*Nt+1)*Nx+(j-1)*(Nt+1)+n+1]-w[(2*Nt+1)*Nx+(j-1)*(Nt+1)+n]+beta*dt*f_star((w[(2*Nt+1)*Nx+(j-1)*(Nt+1)+n+1]-w[(2*Nt+1)*Nx+(j-2)*(Nt+1)+n+1])/dx,w[(j-1)*(Nt+1)+n])+ep2*(w[(2*Nt+1)*Nx+j*(Nt+1)+n+1]-2*w[(2*Nt+1)*Nx+(j-1)*(Nt+1)+n+1]+w[(2*Nt+1)*Nx+(j-2)*(Nt+1)+n+1])
        # F_rho_int , F[3*Nt*Nx+1]->F[3*Nt*Nx+Nx-2] ********** 11
        FF[3*Nt*Nx+j-1]=w[(j-1)*(Nt+1)]-(1/dx)*quad(rho_int,x[j-1],x[j])
        # F_V_ter , F[3*Nt*Nx+Nx+1]->F[3*Nt*Nx+2*Nx-2] ********* 14
        FF[3*Nt*Nx+Nx+j-1]=w[(2*Nt+1)*Nx+(j-1)*(Nt+1)+Nt]-VT(x[j])
    # F_rho_int , F[3*Nt*Nx] ********* 10
    FF[3*Nt*Nx]=w[0]-(1/dx)*quad(rho_int,x[0],x[1])
    # F_rho_int , F[3*Nt*Nx+Nx-1] ********* 12
    FF[3*Nt*Nx+Nx-1]=w[(Nx-1)*(Nt+1)]-(1/dx)*quad(rho_int,x[Nx-1],x[Nx])
    # F_V_ter , F[3*Nt*Nx+Nx] *********** 13 
    FF[3*Nt*Nx+Nx]=w[(2*Nt+1)*Nx+Nt]-VT(x[1])
    # F_V_ter , F[3*Nt*Nx+2*Nx-1] ************** 15
    FF[3*Nt*Nx+2*Nx-1]=w[(2*Nt+1)*Nx+(Nx-1)*(Nt+1)+Nt]-VT(x[Nx])
    
    return FF


import numdifftools as nd
def Jacobian(w):
    
    Jac=nd.Jacobian(func)
    J1=Jac(w)
    return J1

# print(func(guess0))
# print(Jacobian(guess0))

from scipy.interpolate import griddata
def interpol(Nt,Nt_mul,Nx,Nx_mul,w): # 1D interpolation
    
    """" Go from a coarse grid Nt*Nx to a finer grid spacing (Nt_mul*Nt)*(Nx_mul*Nx) """""

    n=w.shape[0] # n=3Nt*Nx+2Nx
    i = np.indices(w.shape)[0]/(n-1)  # [0, ..., 1]
    new_n = 3*(Nt_mul*Nt)*(Nx_mul*Nx)+2*(Nx_mul*Nx)
    print('n={n}, new_n={new_n}'.format(n=n,new_n=new_n))
    new_i = np.linspace(0, 1, new_n)
    new_w=griddata(i, w, new_i, method="cubic")  # method{‘linear’, ‘nearest’, ‘cubic’}
    
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
        u[0,n]=f_star_p(V[0,n+1]/dx,rho[0,n])
        Q[0,n]=rho[0,n]*u[0,n]
#     print("rho=",rho)
#     print("u=",u)
#     print("V=",V)
    return 0

""" ********************************************************************************************* """
""" solve in coarse grid """
Nx=2; Nt=3 # spatial-temporal grid sizes
print(3*Nt*Nx+2*Nx)
dx=L/Nx # spatial step size
# dt=min(T/Nt,CFL*dx/abs(u_max)) # temporal step size
EPS=0.45
dt=min(T/Nt,CFL*dx/abs(u_max),EPS*(dx**2)/mu) # temporal step size
# ep1=-mu*dt/(dx**2)  # rho
ep2=mu*dt/(dx**2) # V
print('dx={dx}, dt={dt}'.format(dx=round(dx,3),dt=round(dt,3)))
x=np.linspace(0,L,Nx+1)
t=np.linspace(0,T,Nt+1)
guess0 = np.zeros(3*Nt*Nx+2*Nx)

import time
t0 = time.process_time()   ###
sol0=gmres(guess0, func, 0.01, 1e-10)
t1 = time.process_time()   ###
print("Time spent :",t1-t0)

print('sol0=',sol0)
print('f(sol0)=',func(sol0,beta=1))
