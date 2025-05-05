import numpy as np
from math import log,exp
import emcee
import corner

def gaussian(x1,x0,s):
    return exp(-((x1-x0)**2)/(2*s**2))/(np.sqrt(2*np.pi)*s)

def Gen_gauss(x0,s0,Num):
    xs=[]
    for i in range(Num):
        while True:
            xr=np.random.normal(x0,s0)
            if xr > 0 and xr < 1:
                xs.append(xr)
                break
    if Num==1:
        return xs[0]
    else:
        return np.array(xs)

def lnprior(th):
    mu,sig=th
    if mu < 1 and mu > 0 and sig > 0 and sig < 0.2:
        return 0.0
    return -np.inf

def lnlike(th,samp):
    s_s=0
    row, col = np.shape(samp)
    for i in range(row):
        sm=0
        for j in range(col):
            sm=sm+gaussian(samp[i][j],th[0],th[1])
        sm=sm/(1.*col)
        if sm==0:
            return -np.inf
        s_s += log(sm)
    return s_s

def lnprob(th,samp):
    lp=lnprior(th)
    if not np.isfinite(lp):
        return -np.inf
    return lp+lnlike(th,samp)

k,N=20,100
mu_true, std_true = 0.4, 0.1
ls,samples,res=[],[],[]
for i in range(k):
    ls.append(Gen_gauss(mu_true, std_true, 1))
    s_tmp=np.random.uniform(0,0.2)
    samples.append(Gen_gauss(ls[i],s_tmp,N))

ndim,nwalkers=2,100
pos=[[np.random.uniform(0.3,0.5), np.random.uniform(0.05,.15)] for i in range(nwalkers)]
sampler=emcee.EnsembleSampler(nwalkers,ndim,lnprob,args=[samples])
sampler.run_mcmc(pos,500)
samps=sampler.chain[:,50:,:].reshape((-1,ndim))

print("Mean acceptance fraction: {0:.3f}"
                .format(np.mean(sampler.acceptance_fraction)))

fig=corner.corner(samps,labels=[r'$\mu$',r'$\sigma$'], quantiles=[0.16, 0.5, 0.84], truths=[mu_true, std_true], show_titles=True, title_kwargs={"fontsize": 12} )
plt.suptitle(r'Recovered 1D PPD for $\mu$,  $\sigma$, and their joint 2D distribution')
fig.tight_layout(rect=[0, 0.03, 1, 0.95]);
fig.savefig('triangle.png')
fig.show()
