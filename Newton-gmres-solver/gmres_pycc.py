#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 19:22:05 2021

@author: amal
"""

from numpy import sqrt,zeros,float32


def get_Hessenberg_matrix(H:'float32[:,:]',m:'int',h:'float32[:,:]'):
    for s in range(m):
        for k in range(s+2):
            h[k,s]=H[s][k]

def identity(n:'int',I:'float32[:,:]'):
    for i in range(n):        
         I[i,i] = float32(1.0)
         

def AxB(X:'float32[:,:]',Y:'float32[:,:]',XY:'float32[:,:]'):
    for i in range(len(X)):
    # iterate through columns of Y
        for j in range(len(Y[0])):
        # iterate through rows of Y
            for k in range(len(Y)):
                XY[i][j] += X[i][k] * Y[k][j]


def get_preconditionner(A:'float32[:,:]',M:'float32[:,:]'):
    # A: Jacobian of fct, Ignoring the forward-backward coupling  parts
    
    """ Compute LU Factorization """
    n = len(A)
    L = zeros((n,n),dtype=float32)
    U = zeros((n,n),dtype=float32)
    for i in range(n):
        L[i,i] = 1
    U[:] = A
    # n = len(L)
    for i in range(n):
        pivot=U[i,i]
        for j in range(i+1,n):
            L[j,i] = (1/pivot)*U[j,i]
            for k in range(n):
                U[j,k] = U[j,k] - L[j,i]*U[i,k]
    
    """ This E1 is used to find the inverse of U """
    E1=zeros((n,n),dtype= float32)
    identity(n,E1)
    for j in range(n-1,-1,-1):
        E1[j,:] = E1[j,:]/U[j,j]
        U[j,:] = U[j,:]/U[j,j]
        for i in range(j-1,-1,-1):
            E1[i,:] = E1[i,:]-E1[j,:]*U[i,j]
            U[i,:] = U[i,:]-U[j,:]*U[i,j]
    
    """ This E2 is used to find the inverse of L """
    E2=zeros((n,n),dtype= float32)
    identity(n,E2) 
    for j in range(n):
        for i in range(j+1,n):
            E2[i,:] = E2[i,:]-E2[j,:]*L[i,j]
            L[i,:] = L[i,:]-L[j,:]*L[i,j]

    """ Compute inverse of A=LU : M=(U*)(L*) """
    AxB(E1,E2,M)
    

def matvec(A:'float32[:,:]',v:'float32[:]',Av:'float32[:]'):
    n =len(A)
    m = len(v)
    for i in range(n):
        s = 0.0
        for j in range(m):
              s += A[i,j]*v[j]
        Av[i] = s


def scalar(u:'float32[:]',v:'float32[:]'):
    n = len(u)
    s = 0.0
    for i in range(n):
        s += u[i]*v[i]
    return s

def norm_two(v:'float32[:]'):
    n=len(v)
    norm=v[0]**2
    for i in range(1,n):
        norm+=v[i]**2
    return sqrt(norm)

def matT(A:'float32[:,:]',At:'float32[:,:]'):
    n=len(A)
    m=len(A[0])
    for i in range(m):
        for j in range(n):
            At[i,j]=A[j,i]


def vec_asarray(v:'float32[:]',vv:'float32[:,:]'):
    for i in range(len(v)):
        for j in range(len(v[0])):
            vv[i,j]=v[i][j]

    
def least_squares(A:'float32[:,:]',b:'float32[:]',y:'float32[:]'):
    At=zeros((len(A[0]),len(A)),dtype= float32)
    matT(A,At)
    AtA=zeros((len(A[0]),len(A[0])),dtype= float32)
    AxB(At,A,AtA)
    Atb=zeros(len(A[0]),dtype= float32)
    matvec(At,b,Atb)
    GC(AtA,Atb,y) 

def GC(A:'float32[:,:]',b:'float32[:]',x:'float32[:]'):
    """ Conjugate Gradient """
    r=b
    p=b
    rr = scalar(r,r)
    while True:
        Ap=zeros(len(A),dtype=float32)
        matvec(A,p,Ap)
        alpha = rr / scalar(Ap,p)
        x = x + alpha*p
        r = r - alpha*Ap
        rrnew = scalar(r,r)
        if sqrt(rrnew) < 1e-08:
            break
        p = r + (rrnew/rr)*p
        rr = rrnew
    

def gmres(w0:'float32[:]', sigma:'float', tol:'float'):
    while True :
        r=-func(w0)
        v=[]
        v.append(r/norm_two(r))
        Matr=Jacobian(w0)
        M=zeros((len(Matr[0]),len(Matr)),dtype= float32)
        get_preconditionner(Jacobian(w0),M)
        H=[]
        j=0
        while True :
            hh=[]
            xj=zeros(len(M),dtype = float32)
            matvec(M, v[j],xj)
            q=(func(w0+sigma*xj)-func(w0))/sigma
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
            if norm_two(func(w0)+q)<=tol:
                break
            j+=1
        h=zeros((m+1,m),dtype= float32)
        get_Hessenberg_matrix(H,m,h)
        # calcul of beta*e1
        beta=zeros(m,dtype = float32)
        beta[0]=norm_two(r)
        # Minimize for y
        ht=zeros((m,m+1),dtype= float32)
        matT(h,ht)
        y=zeros(m+1,dtype=float32)
        least_squares(ht,beta,y)
        # y=np.linalg.lstsq(ht,beta,rcond=-1)[0]
        # print(matT(h))
        # print('beta=',beta)
        # y=least_squares(matT(h),beta)
        v_arr=zeros((len(v),len(v[0])),dtype= float32)
        vec_asarray(v,v_arr)
        v_arr_t=zeros((len(v[0]),len(v)),dtype= float32)
        matT(v_arr,v_arr_t)
        vy=zeros(len(v_arr_t),dtype = float32)
        matvec(v_arr_t,y,vy)
        w0_new=w0+vy
        if norm_two(func(w0))<=tol or norm_two(w0-w0_new)<=0.00000001 :
            break
        # tol=max(0.9*(norm_two(func(w0_new))/norm_two(func(w0)))**2,0.9*tol**2)
        # print(tol)
        w0=w0_new
        print('w0=',w0)
    w0=w0_new


import numpy as np

def func(x):
    return np.array([x[0] * np.cos(x[1]) - 4, x[1] * x[0] - x[1] - 5])

def Jacobian(x):
    return np.array([[np.cos(x[1]),-x[0] * np.sin(x[1])],[x[1],x[0]-1]])


result=np.array([0,0])
gmres(result, 0.01, 1e-10)
print("\n***************** Using Newton-GMRES ******************** \n")
print('x=',result)
print('f(x)=',func(result))

