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
Nx=5; Nt=30 # spatial-temporal grid sizes
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
def U(rho): # Greenshields desired speed
    return u_max*(1-rho/rho_jam)

def f_mfg_LWR(u,r):
    return 0.5*((U(r)-u)**2) # MFG-LWR

def f_star_p_LWR(p,r): # 0<=u<=u_max
    return U(r)-p # MFG-LWR
    
def f_star_LWR(p,r): # p=Vx
    return -0.5*(p**2)+U(r)*p # MFG-LWR

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

def formFunction(snes, w, F):
    def f_star_p(p,r): # 0<=u<=u_max
        return U(r)-p # MFG-LWR
    
    def f_star(p,r): # p=Vx
        return -0.5*(p**2)+U(r)*p # MFG-LWR
    
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

def jacobian(w): # Ignoring the forward-backward coupling  parts
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

    return row, col, data


def get_preconditioner(a):
    row, col, data =jacobian(a)
    shap=(3*Nt*Nx+2*Nx,3*Nt*Nx+2*Nx)
    Jac1 = csc_matrix((data, (row, col)),shape = shap)
    # the *incomplete LU* decomposition
    J_ilu = spla.spilu(Jac1)
    M_x = lambda r: J_ilu.solve(r)
    M = spla.LinearOperator(shap, M_x)

    return M
       

 
# """************************ solve in grid 1***************************** """
from petsc4py import PETSc

shap=(3*Nt*Nx+2*Nx,3*Nt*Nx+2*Nx)
# guess = np.zeros(3*Nt*Nx+2*Nx) # initial guess
# t0 = time.process_time()   ###
# prec=get_preconditioner(guess) # get-preconditioner
# mat = PETSc.Mat().create()
# mat.setSizes(shap)
# mat.setType("mpiaij")
# mat.setFromOptions()

# mat.setPreallocationNNZ(10)
# mat.setOption(option=19, flag=0)

# row, col, data =jacobian(guess)
# for i in range(len(data)):
#     mat.setValues(row[i], col[i], data[i], addv=False)
    
# mat.assemblyBegin(mat.AssemblyType.FINAL)
# mat.assemblyEnd(mat.AssemblyType.FINAL)

# create linear solver
snes = PETSc.SNES()
snes.create()

F = PETSc.Vec()
F.create()
F.setSizes(shap[0])
F.setFromOptions()

b = None
# snes.setJacobian(mat)
snes.setFunction(formFunction, F)

xx = PETSc.Vec().createSeq(shap[0]) 

t0 = time.process_time()   ###
snes.solve(b, xx)
t1 = time.process_time()   ###
time2=t1-t0
print("Time spent:",time2)


sol = [0.05152234, 0.10776953, 0.26610081, 0.25807329, 0.27168349,
       0.29121674, 0.25282184, 0.2995675 , 0.24858936, 0.29983911,
       0.25203438, 0.29586752, 0.25777863, 0.29043971, 0.26346511,
       0.28530748, 0.26814325, 0.28124413, 0.27156117, 0.27840095,
       0.27381628, 0.27661983, 0.27515064, 0.27563558, 0.27583325,
       0.27518528, 0.27610015, 0.27505531, 0.27613162, 0.27509267,
       0.27605102, 0.22743772, 0.41111901, 0.14771499, 0.33755932,
       0.22598257, 0.30684979, 0.26182199, 0.28072014, 0.27837102,
       0.26771448, 0.2856024 , 0.2635933 , 0.28729391, 0.26397448,
       0.2860293 , 0.26621502, 0.28360343, 0.26887628, 0.2810768 ,
       0.27125816, 0.27895718, 0.27309731, 0.27740162, 0.27436903,
       0.27637882, 0.27516063, 0.27577858, 0.2755955 , 0.27547557,
       0.27579219, 0.27536033, 0.82006195, 0.22743772, 0.43420794,
       0.16806608, 0.35799801, 0.21305325, 0.32361027, 0.24466263,
       0.29805176, 0.2634107 , 0.28217104, 0.2740224 , 0.2739586 ,
       0.27920039, 0.27063521, 0.28095558, 0.27004946, 0.28081563,
       0.27077532, 0.27982878, 0.2719558 , 0.27863055, 0.27313133,
       0.27754969, 0.27409737, 0.27671451, 0.27479859, 0.276139  ,
       0.27525645, 0.27578301, 0.27552346, 0.22743772, 0.46046529,
       0.2077705 , 0.38502003, 0.18944428, 0.34737751, 0.21535661,
       0.32392379, 0.23893165, 0.30449462, 0.25554689, 0.29027581,
       0.26645784, 0.28121695, 0.27301177, 0.27615929, 0.2764555 ,
       0.27380031, 0.27787461, 0.27306514, 0.2781181 , 0.27318532,
       0.27778245, 0.27366709, 0.2772431 , 0.27422596, 0.27670721,
       0.27472072, 0.27626808, 0.27509908, 0.27595067, 0.05152234,
       0.17119054, 0.32218784, 0.22926336, 0.33287372, 0.21948479,
       0.32437136, 0.22910802, 0.31403828, 0.24252316, 0.30262737,
       0.25422304, 0.2924931 , 0.26315054, 0.28484069, 0.26934471,
       0.27973043, 0.27324573, 0.27669418, 0.27542904, 0.27513472,
       0.27644906, 0.27451603, 0.2767607 , 0.27442953, 0.2766957 ,
       0.27459755, 0.27647155, 0.27485035, 0.27621512, 0.2750966 ,
       0.94847766, 0.89223047, 0.73389919, 0.74192671, 0.72831651,
       0.70878326, 0.74717816, 0.7004325 , 0.75141064, 0.70016089,
       0.74796562, 0.70413248, 0.74222137, 0.70956029, 0.73653489,
       0.71469252, 0.73185675, 0.71875587, 0.72843883, 0.72159905,
       0.72618372, 0.72338017, 0.72484936, 0.72436442, 0.72416675,
       0.72481472, 0.72389985, 0.72494469, 0.72386838, 0.72490733,
       0.77256228, 0.58888099, 0.85228501, 0.66244068, 0.77401743,
       0.69315021, 0.73817801, 0.71927986, 0.72162898, 0.73228552,
       0.7143976 , 0.7364067 , 0.71270609, 0.73602552, 0.7139707 ,
       0.73378498, 0.71639657, 0.73112372, 0.7189232 , 0.72874184,
       0.72104282, 0.72690269, 0.72259838, 0.72563097, 0.72362118,
       0.72483937, 0.72422142, 0.7244045 , 0.72452443, 0.72420781,
       0.17993805, 0.77256228, 0.56579206, 0.83193392, 0.64200199,
       0.78694675, 0.67638973, 0.75533737, 0.70194824, 0.7365893 ,
       0.71782896, 0.7259776 , 0.7260414 , 0.72079961, 0.72936479,
       0.71904442, 0.72995054, 0.71918437, 0.72922468, 0.72017122,
       0.7280442 , 0.72136945, 0.72686867, 0.72245031, 0.72590263,
       0.72328549, 0.72520141, 0.723861  , 0.72474355, 0.72421699,
       0.77256228, 0.53953471, 0.7922295 , 0.61497997, 0.81055572,
       0.65262249, 0.78464339, 0.67607621, 0.76106835, 0.69550538,
       0.74445311, 0.70972419, 0.73354216, 0.71878305, 0.72698823,
       0.72384071, 0.7235445 , 0.72619969, 0.72212539, 0.72693486,
       0.7218819 , 0.72681468, 0.72221755, 0.72633291, 0.7227569 ,
       0.72577404, 0.72329279, 0.72527928, 0.72373192, 0.72490092,
       0.94847766, 0.82880946, 0.67781216, 0.77073664, 0.66712628,
       0.78051521, 0.67562864, 0.77089198, 0.68596172, 0.75747684,
       0.69737263, 0.74577696, 0.7075069 , 0.73684946, 0.71515931,
       0.73065529, 0.72026957, 0.72675427, 0.72330582, 0.72457096,
       0.72486528, 0.72355094, 0.72548397, 0.7232393 , 0.72557047,
       0.7233043 , 0.72540245, 0.72352845, 0.72514965, 0.72378488,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ]


its = snes.getIterationNumber()
lits = snes.getLinearSolveIterations()

print ("Number of SNES iterations = :", its)
print ("Number of Linear iterations =" , lits)

litspit = lits/float(its)
print ("Average Linear its / SNES = %e", float(litspit))

print(max(xx.array-sol))
