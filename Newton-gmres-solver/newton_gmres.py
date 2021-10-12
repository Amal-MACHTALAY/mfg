#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 19:28:40 2021

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
    # A: Jacobian of fct, Ignoring the forward-backward coupling  parts
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
        r=-fct(w0)
        v=[]
        v.append(r/norm_two(r))
        M=get_preconditionner(Jacobian(w0))
        H=[]
        j=0
        while True :
            hh=[]
            xj=matvec(M, v[j])
            q=(fct(w0+sigma*xj)-fct(w0))/sigma
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
            if norm_two(fct(w0)+q)<=tol:
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
        if norm_two(fct(w0))<=tol or norm_two(w0-w0_new)<=1e-08 :
            break
        # # Update tolerance 
        # tol=max(0.9*(norm_two(fct(w0_new))/norm_two(fct(w0)))**2,0.9*tol**2)
        # print(tol)
        w0=w0_new
    
    return w0_new


def func(x):
    return np.array([x[0] * np.cos(x[1]) - 4, x[1] * x[0] - x[1] - 5])


def Jacobian(x):
    return np.array([[np.cos(x[1]),-x[0] * np.sin(x[1])],[x[1],x[0]-1]])




print("\n***************** Using Newton-GMRES ******************** \n")
result=gmres(np.array([0,0]), func, 0.01, 1e-10)
print('x=',result)
print('f(x)=',func(result))

print("\n***************** Exact **********************************\n")
from scipy.optimize import fsolve
root = fsolve(func, [1, 1])
print('x=',root)
# array([6.50409711, 0.90841421])
print('f(x)=',func(root))  # func(root) should be almost 0.0.
