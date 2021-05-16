# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 23:41:14 2021

@author: MACHTALAY AMAL
"""

def location_int(pdf, interval, x_mfg):
    """ generate initial locations from a given initial density pdf """
    random_loc0=sample(pdf,interval,N)
    loc0=np.zeros(len(random_loc0))
    for j in range(len(random_loc0)):
        loc0[j]=closest(x_mfg,random_loc0[j])
    return loc0


def closest(lst, val):
    """ Find Closest number in a list """
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-val))]

def inverse_sample_function(dist, pnts, x_min=-100, x_max=100, n=1e5, 
                            **kwargs):
        
    x = np.linspace(x_min, x_max, int(n))
    cumulative = np.cumsum(dist(x, **kwargs))
    cumulative -= cumulative.min()
    f = interp1d(cumulative/cumulative.max(), x)
        
    return f(np.random.random(pnts))

def sample(pdf,interval,nbr_samples, n_max=10**6):
    np.random.seed(42)
    """ genearte a list of random samples from a given pdf
    suggests random samples between interval[0] and interval[1] 
    and accepts-rejects the suggestion with probability pdf(x) 
    """
    samples=[]
    n=0
    while len(samples)<nbr_samples and n<n_max:
        x=np.random.uniform(low=interval[0],high=interval[1])
        new_sample=pdf(x)
        assert new_sample>=0 and new_sample<=1
        if np.random.uniform(low=0,high=1) <=new_sample:
            samples += [x]
        n+=1
    return sorted(samples)

def find_index(lst, val):
    ind='nan'
    for i in range(len(lst)):
        if lst[i]==val:
            ind=i
            break
    return ind
