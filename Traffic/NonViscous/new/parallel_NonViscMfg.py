#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 17:44:36 2021

@author: amal
"""

import numpy as np
from scipy import integrate
from scipy.optimize.nonlin import newton_krylov
# from scipy.sparse import csc_matrix
# import scipy.sparse.linalg as spla
# import time
from mpi4py import MPI

COMM = MPI.COMM_WORLD  # The default communicator
SIZE = COMM.Get_size() # The number of processes
RANK = COMM.Get_rank() # The rank of processes

''' inputs..............................................................................................................''' ## DONE
T=3.0 # horizon length 
N=1 # number of cars 
u_max=1.0 # free flow speed
rho_jam=1.0 # jam density
L=N # road length
CFL=0.75    # CFL<1
rho_a=0.05; rho_b=0.95; gama=0.1
# rho_a=0.2; rho_b=0.8; gama=0.15*L
costf="LWR"
# """ Viscous solution"""
EPS=0.45
mu=0.0 # viscosity coefficient 

# Nx_list=[]
# Nt_list=[]

""" grid discretization.................................................................................................. """  ## DONE
Nx=15; Nt=60 # Final spatial-temporal grid  
dx=L/Nx # spatial step size
if mu==0.0:
    dt=min(T/Nt,(CFL*dx)/u_max) # temporal step size
    eps=0.0
else:
    dt=min(T/Nt,CFL*dx/abs(u_max),EPS*(dx**2)/mu) # temporal step size
    eps=mu*dt/(dx**2) # V
x=np.linspace(0,L,Nx+1)
# t=np.linspace(0,T,Nt+1)
t=np.arange(0,T+dt,dt)
Nt=len(t)-1
if RANK==0:
    print('Nx={Nx}, Nt={Nt}'.format(Nx=Nx,Nt=Nt))
    print('dx={dx}, dt={dt}'.format(dx=round(dx,4),dt=round(dt,4)))
    
guess0 = np.zeros(3*Nt*Nx+2*Nx) ## Done
# guess0 = np.arange(0,3*Nt*Nx+2*Nx,1)
# guess0=np.loadtxt('Sol0_LWR_T3_N1.dat')


""" for MPI : Creates a division of processors in a cartesian grid.................................................... """ ## DONE
nbr_x=Nx+1; nbr_t=Nt+1 # spatial-temporal grid sizes 
nx=nbr_x-1; nt=nbr_t # number of points for MPI
px=int(np.sqrt(SIZE))-1 # number of processes on each line-1
pt=px # number of processes on each column-1
# print("px={px}, pt={pt}".format(px=px, pt=pt))
new_size=(px+1)*(pt+1) # the Number of processes to decompose 
# print('new_size=',new_size)
nbrx=int(nx/(px+1)) #number of points for px (except root)
nbrt=int(nt/(pt+1)) #number of points for pt (except root)
dims=[px+1,pt+1] # The array containing the number of processes to assign to each dimension
# print('dims=',dims)
npoints=[nx,nt]
# print('npoints=',npoints)

""" functions For MPI.................................................................................................. """ ## DONE

def create_2d_cart(): # return communicator (cart2d) with new cartesian topology
                                                                                                                                                                                                                                              
    periods = tuple([True, False]) # True : periodic, False : non-periodic Cartesian topology
    reorder = False # the rank of the processes in the new communicator (COMM_2D) is the same as in the old communicator (COMM). 
    
    # if (RANK == 0):
    #     print("Exécution avec",SIZE," MPI processes\n"
    #             "Taille du domaine : nx=",npoints[0], " nt=",npoints[1],"\n"
    #             "Dimension pour la topologie :",dims[0]," along x", dims[1]," along t\n"
    #             "-----------------------------------------") 

    cart2d = COMM.Create_cart(dims = dims, periods = periods, reorder = reorder)
    
    return cart2d

nb_neighbours = 4
N = 0 # hight
E = 1 # right
S = 2 # low
W = 3 # left

def create_neighbours(cart2d): # Find processor neighbors

    neighbour = np.zeros(nb_neighbours, dtype=np.int8)
    # Outputs : rank of source, destination processes
    neighbour[W],neighbour[E] = cart2d.Shift(direction=0,disp=1) # direction 0: <->
    neighbour[S],neighbour[N] = cart2d.Shift(direction=1,disp=1) # direction 1 : |
    
    # print("I am", RANK," my neighbours are : N", neighbour[N]," E",neighbour[E] ," S ",neighbour[S]," W",neighbour[W])

    return neighbour

def Coords_2D(cart2d):

    coord2d = cart2d.Get_coords(RANK)
    # print("I’m rank :",RANK," my 2d coords are",coord2d)
    
    sy = int((coord2d[1] * npoints[1]) / dims[1]) + 1
    
    sx = int((coord2d[0] * npoints[0]) / dims[0]) + 1

    ex = int(((coord2d[0] + 1) * npoints[0]) / dims[0])
    ey = int(((coord2d[1] + 1) * npoints[1]) / dims[1])

    # print("Rank in the topology :",RANK," Local Grid Index :", sx, " to ",ex," along x, ",
    #       sy, " to", ey," along t")
    
    return coord2d, sx, ex, sy, ey


def create_derived_type(sx, ex, sy, ey): 
    
    '''Creation of the type_line derived datatype to exchange points
     with northern to southern neighbours '''
    ## Row of a matrix is not contiguous in memory
    ## In N x M matrix, each element of a row is separated by N elements
    type_ligne = MPI.DOUBLE.Create_contiguous(ey-sy + 1) # count = N = ey-sy + 1
    type_ligne.Commit() # A new datatype must be committed before using it
    
    '''Creation of the type_column derived datatype to exchange points
     with western to eastern neighbours '''
    ## A vector type describes a series of blocks, all of equal size (blocklen), spaced with a constant stride.
    ## In N x M matrix, N= and M=block_count
    # block_count : The number of blocks to create.
    # blocklen : The number of (same) elements in each block
    # stride : Distance between the start of each block, expressed in number of elements.
    type_column = MPI.DOUBLE.Create_vector(ex-sx + 3, 1, ey-sy + 3) # block_count = ex-sx + 1 ; blocklen = 1 ; stride = ey-sy + 3
    type_column.Commit()

    return type_ligne, type_column

def IDX(i, j): 
        return ( ((i)-(sx-1))*(ey-sy+3) + (j)-(sy-1) )

def communications(u, sx, ex, sy, ey, type_column, type_ligne):

    #[data, count, datatype]
    ''' Envoyer au voisin W et recevoir du voisin E '''#type_ligne
    COMM.Send([u[IDX(sx, sy) : ], 1, type_ligne], dest=neighbour[W]) #IDX(sx, sy)XX
    COMM.Recv([u[IDX(ex+1, sy) : ], 1, type_ligne], source=neighbour[E]) #IDX(ex, sy+1)XX ,,, IDX(ex+1, sy)

    ''' Envoyer au voisin E et recevoir du voisin W '''#type_ligne
    COMM.Send([u[IDX(ex, sy) : ], 1, type_ligne], dest=neighbour[E]) #IDX(ex, sy)XX
    COMM.Recv([u[IDX(sx-1, sy) : ], 1, type_ligne], source=neighbour[W]) #IDX(sx, sy-1)XX

    ''' Envoyer au voisin S et recevoir du voisin N '''#type_column
    COMM.Send([u[IDX(sx-1, sy) : ], 1, type_column], dest=neighbour[S]) #IDX(sx, sy)XX
    COMM.Recv([u[IDX(sx-1, ey+1) : ], 1, type_column], source=neighbour[N]) #IDX(sx+1, ey)XX

    ''' Envoyer au voisin N et recevoir du voisin S ''' #type_column
    COMM.Send([u[IDX(sx-1, ey) : ], 1, type_column], dest=neighbour[N]) #IDX(sx, ey)XX
    COMM.Recv([u[IDX(sx-1, sy-1) : ], 1, type_column], source=neighbour[S]) #IDX(sx-1, sy)XX ,,,, IDX(sx, sy-1)
    
    return 0


cart2D=create_2d_cart() ## Done
neighbour = create_neighbours(cart2D) ## Done
coord2D, sx, ex, sy, ey = Coords_2D(cart2D) ## Done
type_ligne, type_column = create_derived_type(sx, ex, sy, ey) ## Done

""" Initialization................................................................................................ """ ## DONE

def r_id(j,n): ## Done
    return (j-1)*(Nt+1)+n

def u_id(j,n): ## Done
    return (Nt+1)*Nx+(j-1)*Nt+n

def V_id(j,n): ## Done
    return (2*Nt+1)*Nx+(j-1)*(Nt+1)+n

def to_solution(Nx,Nt,sol,rho,u,V): 
    for j in range(1,Nx+1):
        for n in range(0,Nt):
            rho[j,n]=sol[r_id(j,n)]
            u[j,n]=sol[u_id(j,n)]
            V[j,n]=sol[V_id(j,n)]
        rho[j,Nt]=sol[r_id(j,Nt)]
        V[j,Nt]=sol[V_id(j,Nt)]
    return 0

def solution_to(Nx,Nt,sol,rho,u,V): 
    for j in range(1,Nx+1):
        for n in range(0,Nt):
            sol[r_id(j,n)]=rho[j,n]
            sol[u_id(j,n)]=u[j,n]
            sol[V_id(j,n)]=V[j,n]
        sol[r_id(j,Nt)]=rho[j,Nt]
        sol[V_id(j,Nt)]=V[j,Nt]
    return 0

rho0=np.zeros((Nx+1,Nt+1))  ## Done
u0=np.zeros((Nx+1,Nt+1))  ## Done
V0=np.zeros((Nx+1,Nt+1))  ## Done
to_solution(Nx,Nt,guess0,rho0,u0,V0)  ## Done
# if RANK==0:
#     print('guess0=',guess0)
#     print('rho0=',rho0)
#     print('u0=',u0)
#     print('V0=',V0)

f_rho=rho0.copy()
f_u=u0.copy()
f_V=V0.copy()

def initialization(rho0, u0, V0, sx, ex, sy, ey):  
    
    SIZE = (ex-sx+3) * (ey-sy+3)
    
    rho       = np.zeros(SIZE)
    u       = np.zeros(SIZE)
    V   = np.zeros(SIZE)
    
    '''Initialization of rho, u et V '''
    for i in range(sx, ex+1): # x axis
        for j in range(sy, ey+1): # y axis
        
            rho[IDX(i, j)] = rho0[i,j-1]
            u[IDX(i, j)] = u0[i,j-1]
            V[IDX(i, j)] = V0[i,j-1]
                   
    # print("I’m rank :",RANK, "sx",sx,"ex",ex, "sy",sy, "ey",ey," new_rho",rho)
            
    return rho, u, V

rho, u, V = initialization(rho0, u0, V0, sx, ex, sy, ey) ## Done
# print("rank",RANK,"len(rho)",len(rho),"len(u)",len(u),"len(V)",len(V))
# if RANK==3:
#     print('rho=',rho)
#     print('u=',u)
#     print('V=',V)
    
def r_idx(j,n): 
    return (j-(sx-1))*w_nby+(n-(sy-1))

def u_idx(j,n): 
    return w_nby*w_nbx+(j-(sx-1))*w_nby+(n-(sy-1))

def V_idx(j,n): 
    return 2*w_nby*w_nbx+(j-(sx-1))*w_nby+(n-(sy-1))
    
def to_sol(sol,rho,u,V): 
    for j in range(sx-1,ex+2):
        for n in range(sy-1,ey+2):
            sol[r_idx(j,n)]=rho[IDX(j,n)]
            sol[u_idx(j,n)]=u[IDX(j,n)]
            sol[V_idx(j,n)]=V[IDX(j,n)]
    return 0

def sol_to(sol,rho,u,V): 
    for j in range(sx-1,ex+2):
        for n in range(sy-1,ey+2):
            rho[IDX(j,n)]=sol[r_idx(j,n)]
            u[IDX(j,n)]=sol[u_idx(j,n)]
            V[IDX(j,n)]=sol[V_idx(j,n)]
    return 0

def r_idx_loc(j,n): 
        return (j-j0)*(F_nby+1)+(n-n0)

def u_idx_loc(j,n): 
    return (F_nby+1)*F_nbx+(j-j0)*F_nby+(n-n0)

def V_idx_loc(j,n): 
    return 2*F_nby*F_nbx+F_nbx+(j-j0)*(F_nby+1)+(n-n0)

def global_to_local(local_sol,global_sol):  ### Done
    for j in range(j0,F_Nx+1):
        for n in range(n0,F_Nt+1):
            local_sol[r_idx_loc(j,n)]=global_sol[r_idx(j,n)]
            if n!=F_Nt:
                local_sol[u_idx_loc(j,n)]=global_sol[u_idx(j,n)]
            local_sol[V_idx_loc(j,n)]=global_sol[V_idx(j,n)]
    return 0

def local_sol_to(sol,rho,u,V):  ###?
        for j in range(j0,F_Nx+1):
            for n in range(n0,F_Nt+1):
                rho[IDX(j,n)]=sol[r_idx_loc(j,n)]
                if n!=F_Nt:
                    u[IDX(j,n)]=sol[u_idx_loc(j,n)]
                V[IDX(j,n)]=sol[V_idx_loc(j,n)]
        return 0
    
''' MFG functions....................................................................................................... ''' ## DONE
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

def Fr_idx_loc(j,n): 
        return (j-j0)*F_nby+(n-n0)

def Fu_idx_loc(j,n): 
    return F_nby*F_nbx+(j-j0)*F_nby+(n-n0)

def FV_idx_loc(j,n): 
    return 2*F_nby*F_nbx+(j-j0)*F_nby+(n-n0)

# from scipy.special import logsumexp

def get_newton_krylov(w,wloc):
    
    def F(wloc): 
        # FF=[F_rho,F_u,F_V,F_rho_int,F_V_ter]
        FF=np.zeros(3*F_nby*F_nbx+2*F_nbx)
        for n in range(n0,F_Nt):
            for j in range(j0+1,F_Nx):
                # F_rho , F[0]->F[F_nby*F_nbx-1] ************ 2 
                FF[Fr_idx_loc(j,n)]=wloc[r_idx_loc(j,n+1)]-0.5*(wloc[r_idx_loc(j-1,n)]+wloc[r_idx_loc(j+1,n)])
                +(0.5*dt/dx)*(wloc[r_idx_loc(j+1,n)]*wloc[u_idx_loc(j+1,n)]-wloc[r_idx_loc(j-1,n)]*wloc[u_idx_loc(j-1,n)])
                # F_u , F[F_nby*F_nbx]->F[2*F_nby*F_nbx-1] *********** 5
                FF[Fu_idx_loc(j,n)]=wloc[u_idx_loc(j,n)]-f_star_p((wloc[V_idx_loc(j,n+1)]-wloc[V_idx_loc(j-1,n+1)])/dx,wloc[r_idx_loc(j,n)])
                # F_V , F[2*F_nby*F_nbx]->F[3*F_nby*F_nbx-1] ********* 8 
                FF[FV_idx_loc(j,n)]=wloc[V_idx_loc(j,n+1)]-wloc[V_idx_loc(j,n)]
                +dt*f_star((wloc[V_idx_loc(j,n+1)]-wloc[V_idx_loc(j-1,n+1)])/dx,wloc[r_idx_loc(j,n)])
                +eps*(wloc[V_idx_loc(j+1,n+1)]-2*wloc[V_idx_loc(j,n+1)]+wloc[V_idx_loc(j-1,n+1)])
        
            # F_rho , F[0]->F[F_nby*F_nbx-1] ************ 1 
            FF[Fr_idx_loc(j0,n)]=wloc[r_idx_loc(j0,n+1)]-0.5*(w[r_idx(j0-1,n)]+wloc[r_idx_loc(j0+1,n)])
            +(0.5*dt/dx)*(wloc[r_idx_loc(j0+1,n)]*wloc[u_idx_loc(j0+1,n)]-w[r_idx(j0-1,n)]*w[u_idx(j0-1,n)])
            # F_u , F[F_nby*F_nbx]->F[2*F_nby*F_nbx-1] *********** 4 
            FF[Fu_idx_loc(j0,n)]=wloc[u_idx_loc(j0,n)]-f_star_p((wloc[V_idx_loc(j0,n+1)]-w[V_idx(j0-1,n+1)])/dx,wloc[r_idx_loc(j0,n)])
            # F_V , F[2*F_nby*F_nbx]->F[3*F_nby*F_nbx-1] ********* 7 
            FF[FV_idx_loc(j0,n)]=wloc[V_idx_loc(j0,n+1)]-wloc[V_idx_loc(j0,n)]
            +dt*f_star((wloc[V_idx_loc(j0,n+1)]-w[V_idx(j0-1,n+1)])/dx,wloc[r_idx_loc(j0,n)])
            +eps*(wloc[V_idx_loc(j0+1,n+1)]-2*wloc[V_idx_loc(j0,n+1)]+w[V_idx(j0-1,n+1)])
        
            # F_rho , F[0]->F[F_nby*F_nbx-1] ************ 3 
            FF[Fr_idx_loc(F_Nx,n)]=wloc[r_idx_loc(F_Nx,n+1)]-0.5*(wloc[r_idx_loc(F_Nx-1,n)]+w[r_idx(F_Nx+1,n)])
            +(0.5*dt/dx)*(w[r_idx(F_Nx+1,n)]*w[u_idx(F_Nx+1,n)]-wloc[r_idx_loc(F_Nx-1,n)]*wloc[u_idx_loc(F_Nx-1,n)])
            # F_u , F[F_nby*F_nbx]->F[2*F_nby*F_nbx-1] *********** 6 
            FF[Fu_idx_loc(F_Nx,n)]=wloc[u_idx_loc(F_Nx,n)]-f_star_p((wloc[V_idx_loc(F_Nx,n+1)]-wloc[V_idx_loc(F_Nx-1,n+1)])/dx,wloc[r_idx_loc(F_Nx,n)])
            # F_V , F[2*F_nby*F_nbx]->F[3*F_nby*F_nbx-1] ********* 9 
            FF[FV_idx_loc(F_Nx,n)]=wloc[V_idx_loc(F_Nx,n+1)]-wloc[V_idx_loc(F_Nx,n)]
            +dt*f_star((wloc[V_idx_loc(F_Nx,n+1)]-wloc[V_idx_loc(F_Nx-1,n+1)])/dx,wloc[r_idx_loc(F_Nx,n)])
            +eps*(w[V_idx(F_Nx+1,n+1)]-2*wloc[V_idx_loc(F_Nx,n+1)]+wloc[V_idx_loc(F_Nx-1,n+1)])
            
                       
        for j in range(j0,F_Nx+1):    
            # F_rho_int , F[3*F_nby*F_nbx]->F[3*F_nby*F_nbx+F_nbx-1] ********** 10
            FF[3*F_nby*F_nbx+j-j0]=wloc[r_idx_loc(j,n0)]-(1/dx)*integral(x[j-1],x[j])
            # F_V_ter , F[3*F_nby*F_nbx+F_nbx]->F[3*F_nby*F_nbx+2*F_nbx-1] ********* 11
            FF[3*F_nby*F_nbx+F_nbx+j-j0]=wloc[V_idx_loc(j,F_Nt)]-VT(x[j])
            
            
            
        return FF
    
    def jacobian(wloc): # Ignoring the forward-backward coupling  parts  
        J=np.zeros((3*F_nby*F_nbx+2*F_nbx,3*F_nby*F_nbx+2*F_nbx))
        for j in range(j0,F_Nx+1):
            for n in range(n0,F_Nt):
                J[Fr_idx_loc(j,n),r_idx_loc(j,n)+1]=1 # F_rho - rho
                if j!=F_Nx:
                    J[Fr_idx_loc(j,n),r_idx_loc(j,n)+F_nby+1]=(0.5*dt/dx)*wloc[u_idx_loc(j+1,n)]-0.5 # F_rho -rho
                    J[Fr_idx_loc(j,n),u_idx_loc(j,n)+F_nby]=(0.5*dt/dx)*wloc[r_idx_loc(j+1,n)] # F_rho - u
                    J[FV_idx_loc(j,n),V_idx_loc(j,n)+(F_nby+1)+1]=eps # F_V - V 
                if j==F_Nx:
                    J[Fr_idx_loc(j,n),r_idx_loc(j,n)+F_nby+1]=(0.5*dt/dx)*w[u_idx(j+1,n)]-0.5 # F_rho -rho
                    J[Fr_idx_loc(j,n),u_idx_loc(j,n)+F_nby]=(0.5*dt/dx)*w[r_idx(j+1,n)] # F_rho - u
                if j!=j0:
                    J[Fr_idx_loc(j,n),r_idx_loc(j,n)-F_nby-1]=-(0.5*dt/dx)*wloc[u_idx_loc(j-1,n)]-0.5 # F_rho -rho
                    J[Fr_idx_loc(j,n),u_idx_loc(j,n)-F_nby]=-(0.5*dt/dx)*wloc[r_idx_loc(j-1,n)] # F_rho - u
                    J[FV_idx_loc(j,n),V_idx_loc(j,n)+1]=eps # F_V - V
                if j==j0:
                    J[Fr_idx_loc(j,n),r_idx_loc(j,n)-F_nby-1]=-(0.5*dt/dx)*w[u_idx(j-1,n)]-0.5 # F_rho -rho
                    J[Fr_idx_loc(j,n),u_idx_loc(j,n)-F_nby]=-(0.5*dt/dx)*w[r_idx(j-1,n)] # F_rho - u
                    
                J[Fu_idx_loc(j,n),u_idx_loc(j,n)]=1 # F_u - u
                J[FV_idx_loc(j,n),V_idx_loc(j,n)]=-1 # F_V - V
                J[FV_idx_loc(j,n),V_idx_loc(j,n)+1]=1-2*eps # F_V - V 
            J[3*F_nby*F_nbx+(j-j0),r_idx_loc(j,n0)]=1 # F_rho_int - rho
            J[3*F_nby*F_nbx+F_nbx+(j-j0),V_idx_loc(j,n0)+F_nby-1]=1 # F_V_ter - V
        
        return J
    
    # if RANK==0:
    #     print('F(wloc)=',F(wloc))
    # get_preconditioner
    # t0 = time.process_time()   ###
    Jac=jacobian(wloc)
    M=np.linalg.inv(Jac)
    # Jac1 = csc_matrix(Jac)
    # J_ilu = spla.spilu(Jac1)
    # M_x = lambda r: J_ilu.solve(r)
    # M = spla.LinearOperator(Jac.shape, M_x)
    # if RANK==0:
    #     print('prec=',prec)
    # t1 = time.process_time()   ###
    # print("Time spent (anal_precond) :",t1-t0)
        
    # t0 = time.process_time()   ###
    solu= newton_krylov(F, wloc, method='gmres', verbose=0, inner_M=M, x_rtol=2e-12) # verbose=1
    # print('sol=',solu)
    # t1 = time.process_time()   ###
    # print("Time spent (gmres) :",t1-t0)
    
    return solu
    # return 0
    

''' Calcul of global erreur............................................................................................. '''
def global_error(w, w_new): 
   
    local_error = 0
     
    for i in range(sx, ex+1):
        for y in range(sy, ey+1):
            temp = np.fabs( w[IDX(i, y)] - w_new[IDX(i, y)]  )
            if local_error < temp:
                local_error = temp;
    
    return local_error
        

""" Solve in grid (Nx,Nt)................................................................................................ """

rho_new = rho.copy() 
u_new = u.copy() 
V_new = V.copy()

w_nbx=ex-sx+3 ## Done
w_nby=ey-sy+3 ## Done
# guess_glob=np.zeros(3*w_nbx*w_nby) ## Done
j0=sx; F_Nx=ex; F_Nt=ey    
if coord2D[1]==0:
    n0=sy
else:
    n0=sy-1
F_nbx=F_Nx-j0+1
F_nby=F_Nt-n0
# guess_loc=np.zeros(3*F_nbx*F_nby+2*F_nbx)

''' Stepping time.......................................................... '''
it = 0
convergence = False
it_max = 1000
epsilon = 2.e-16

''' spend time................................................... '''
t1 = MPI.Wtime()
while (not(convergence) and (it < it_max)):
    it = it+1;

    rho_temp = rho.copy() 
    u_temp = u.copy() 
    V_temp = V.copy() 
    rho = rho_new.copy()
    u = u_new.copy() 
    V = V_new.copy() 
    rho_new = rho_temp.copy()
    u_new = u_temp.copy()
    V_new = V_temp.copy() 
        
    
    ''' Échange des interfaces à la n itération '''
    communications(rho, sx, ex, sy, ey, type_column, type_ligne) ## Done
    communications(u, sx, ex, sy, ey, type_column, type_ligne) ## Done
    communications(V, sx, ex, sy, ey, type_column, type_ligne) ## Done
    # if RANK==0:
    #     print('rho=',rho)
    # print(RANK,'ru=',u)
     
    '''Calcul de rho_new, u_new et V_new à l’itération n 1 '''
    
    '''  Global guess '''
    guess_glob=np.zeros(3*w_nbx*w_nby) ## Done
    to_sol(guess_glob,rho,u,V) ## Done
    # print(RANK,len(guess_glob))
    # if RANK==0:
        # print(w_nbx,w_nby,len(rho),len(u),len(V),len(guess_glob))
        # print('guess_glob=',guess_glob)
        
    
    '''  Local guess, Local F => Local solution '''
    guess_loc=np.zeros(3*F_nbx*F_nby+2*F_nbx)
    global_to_local(guess_loc,guess_glob) # Done
    # if RANK==3:
    #     print('guess_loc=',guess_loc)
    
    ''' Solve with Newton-GMRES solver'''
    # t0 = time.process_time()   ###
    solu=get_newton_krylov(guess_glob,guess_loc)
    # t1 = time.process_time()   ###
    # print("Time spent (gmres) :",t1-t0)
    # if RANK==0:
    #     print('*****************************')
    #     print('it=',it)
        
        
    local_sol_to(solu,rho_new,u_new,V_new)
    # if RANK==0:
    #     print('rho_new=',rho_new)
    
    ''' Computation of the global error '''
    local_error = global_error(guess_loc, sol_loc);
    diffnorm = COMM.allreduce(np.array(local_error), op=MPI.MAX )  
#     print(RANK,local_error)
       
    ''' Stop if we got the precision of the machine '''
    convergence = (diffnorm < epsilon) 
    
    ''' Print diffnorm for processor 0 '''
    # if ((RANK == 0) and ((it % 100) == 0)):
    if (RANK == 0):
        print("Iteration", it, " global_error = ", diffnorm);
        
''' temps écoulé...................................................................... '''
t2 = MPI.Wtime()

if (RANK == 0):
    ''' Print convergence time for processor 0 '''
    print("convergence après:",it, 'iterations in', t2-t1,'secs')
    
# if RANK==0:
#     print(rho_new.shape,rho_new)
    
# print(RANK,f_rho.shape,f_rho)
r_recvbuf = COMM.gather(rho_new, root=0)
u_recvbuf = COMM.gather(u_new, root=0)
V_recvbuf = COMM.gather(V_new, root=0)
j0_recvbuf=COMM.gather(sx, root=0)
jf_recvbuf=COMM.gather(ex, root=0)
n0_recvbuf=COMM.gather(sy, root=0)
nf_recvbuf=COMM.gather(ey, root=0)



if RANK==0:
    for rk in range(new_size):
        def IDX2(i, j): 
            return ( (i)*(nf_recvbuf[rk]-n0_recvbuf[rk]+3) + (j) )
        ro=r_recvbuf[rk]
        uu=u_recvbuf[rk]
        VV=V_recvbuf[rk]
        for j in range(j0_recvbuf[rk], jf_recvbuf[rk]+1): # x axis
            for n in range(n0_recvbuf[rk], nf_recvbuf[rk]+1): # y axis
                f_rho[j,n-1]=ro[IDX2(j-j0_recvbuf[rk]+1, n-n0_recvbuf[rk]+1)]
                f_u[j,n-1]=uu[IDX2(j-j0_recvbuf[rk]+1, n-n0_recvbuf[rk]+1)]
                f_V[j,n-1]=VV[IDX2(j-j0_recvbuf[rk]+1, n-n0_recvbuf[rk]+1)]
        
    final_solu=np.zeros(3*Nt*Nx+2*Nx)
    solution_to(Nx,Nt,final_solu,f_rho,f_u,f_V)
    print(Nx,Nt,final_solu.shape)
    
    np.savetxt('PL_Sol0_LWR_T3_N1.dat', final_solu)

























