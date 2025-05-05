import numpy as np
from math import log,exp
import scipy.optimize as opt
import matplotlib.pyplot as plt
import emcee
import corner
from scipy import stats

def samplepost(m1,c1,d1,m2,c2,d2,mmin,mmax,al,bet,N):
    inv=np.array([stats.norm.ppf(1-al),stats.norm.ppf(1-bet)])
    ci1,ci2=np.array([c1,d1]),np.array([c2,d2])
    sig1=np.average(ci1/inv)
    sig2=np.average(ci2/inv)
    vals=[]
    while (len(vals)<N):
        m1_t=np.random.normal(m1,sig1)
        m2_t=np.random.normal(m2,sig2)
        if (mmin<=m2_t)and(m1_t>0)and(m2_t>0)and(m1_t>=m2_t)and(m1_t<=mmax):
            vals.append([m1_t,m2_t])
    return vals

def model_a(m1,m2,mmin,mmax,al):
    if (mmin<=m2)and(m2<=m1)and(m1<=mmax):
        return (m1**(-al))/(m1-mmin)
    return 0

def model_b(m1,m2,mmin,mmax,al,betq):
    return model_a(m1,m2,mmin,mmax,al)*(m2/m1)**betq
    
def model_c(m1,m2,mmin,mmax,al,lamb,mu,sig):
    s=model_a(m1,m2,mmin,mmax,al)
    if (s==0):
        return 0
    return model_a(m1,m2,mmin,mmax,al)+lamb*exp(-(m1-mu)**2/(2*sig**2))

def model_test(m1,m2,mu1,mu2,sig1,sig2,mmax,mmin):
    if (mmin<=m2)and(m2<=m1)and(m1<=mmax):
        return gaussian(m1,mu1,sig1)*gaussian(m2,mu2,sig2)
    return 0

def gaussian(x1,x0,s):
    return exp(-((x1-x0)**2)/(2*s**2))/(np.sqrt(2*np.pi)*s)

def Gen_gauss(x0,s0,Num):
    xs=[]
    for i in range(Num):
        mem=0
        while(mem==0):
            xr=np.random.normal(x0,s0)
            if(xr>=0)and(xr<=1):
                xs.append(xr)
                mem=1
    if (Num==1):
        return xs[0]
    else:
        return np.array(xs)

def lnprior(th):
    mmax,al=th
    if (-20<al<20) and (30<mmax<100):#and(0<mu<100)and(0<sig<=100):
        return 0.0
    return -np.inf

def lnlike(th,samp):
    prod=1
    for i in range(len(samp)):
        sum=0
        for j in range(len(samp[i])):
            sum=sum+model_a(samp[i][j][0],samp[i][j][1],5,th[0],th[1])
        sum=sum/(1.*len(samp[i]))
        prod=prod*sum
        #print prod
        #print prod
    if (prod==0):
        return -np.inf
    return log(prod)

def lnprob(th,samp):
    lp=lnprior(th)
    if not np.isfinite(lp):
        return -np.inf
    return lp+lnlike(th,samp)
