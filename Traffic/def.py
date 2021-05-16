# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 23:41:14 2021

@author: MACHTALAY AMAL
"""

apt=42  # change apt to update random samples

def location_int(pdf, x_mfg):
    """ generate initial locations from a given initial density pdf """
    random_loc=sample(pdf,N)
    loc=np.zeros(len(random_loc))
    for j in range(len(random_loc)):
        loc[j]=closest(x_mfg,random_loc[j])
    return loc

def closest(lst, val):
    """ Find Closest number in a list """
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-val))]

def sample(pdf, nbr_samples, n_max=10**6):
    np.random.seed(apt)
    """ genearte a list of random samples from a given pdf
    suggests random samples between 0 and L 
    and accepts-rejects the suggestion with probability pdf(x) 
    """
    samples=[]
    n=0
    while len(samples)<nbr_samples and n<n_max:
        x=np.random.uniform(low=0,high=L)
        new_sample=pdf(x)
        assert new_sample>=0 and new_sample<=1
        if np.random.uniform(low=0,high=1) <=new_sample:
            samples += [x]
        n+=1
    return sorted(samples)

def random_sampling(pdf,n=N,xmin=0,xmax=L):  
      np.random.seed(apt)
      """  
      genearte a list of random samples from a given pdf
      suggests random samples between 0 and L 
      and accepts-rejects the suggestion with probability pdf(x)
      where  
      P : probability distribution function from which you want to generate random numbers  
      N : desired number of random values  
      xmin,xmax : range of random numbers desired  
 
     - generate x' in the desired range  
     - generate y' between Pmin and Pmax (Pmax is the maximal value of your pdf)  
     - if y'<P(x') accept x', otherwise reject  
     - repeat until desired number is achieved    
      """   
      x=np.linspace(xmin,xmax,1000)  
      y=pdf(x)  
      pmin=0. 
      pmax=y.max()   
      naccept=0     
      samples=[] # output list of random numbers  
      while naccept<n:
        x=np.random.uniform(xmin,xmax) # x'  
        y=np.random.uniform(pmin,pmax) # y'  

        if y<pdf(x):
            samples.append(x)  
            naccept=naccept+1    
            
      return sorted(samples)

def find_index(lst, val):
    """ find the index of val in lst"""
    ind='nan'
    for i in range(len(lst)):
        if lst[i]==val:
            ind=i
            break
    return ind

def control(u_mfg, x_mfg, t_mfg):
    X=np.zeros((N,len(t_mfg)))
    v=np.zeros((N,len(t_mfg)))
    X[:,0]=location_int(rho_int, x_mfg)
    for n in t_mfg:
        for i in range(N):
            ind=find_index(x_mfg,X[i,n])
            v[i,n]=u_mfg[ind,n]
            new_x=X[i,n]+dt*v[i,n]
            if n<N-1:
                X[i,n+1]=closest(x_mfg,new_x)
    return X,v
        
 
def integral(a,b): 
    x2 = lambda x: rho_int(x)
    I=integrate.quad(x2, a, b)
#     result=(1/dx)*max(min(I[0],rho_jam),0) # 0<  <=rho_jam
    return I[0]
