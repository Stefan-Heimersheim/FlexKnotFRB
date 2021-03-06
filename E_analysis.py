import astropy.constants as aco
import astropy.units as aun
import cosmowrap as cw
import fgivenx
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pandas as pd
import scipy.integrate as sin
import scipy.interpolate as sip
import scipy.optimize as sop
import scipy.stats as sst
from anesthetic import MCMCSamples, NestedSamples
from anesthetic.samples import merge_samples_weighted
from copy import deepcopy

mpl.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
mpl.rc('text', usetex=True)
mpl.rc('text.latex', preamble=r'\usepackage{amsmath}')
mpl.rc('image', cmap='viridis')
texDict = {"tau": r"$\tau$", "Omega_b_over_h": r"$\Omega_b/h$", "Omega_m": r"$\Omega_m$"}

# Font sizes, thanks to Pedro M Duarte on stackoverflow, shared under Creative Commons BY-SA
# https://stackoverflow.com/a/39566040
SMALL_SIZE = 10
MEDIUM_SIZE = 12
LARGE_SIZE = 14
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
ANNOTATION_SIZE = SMALL_SIZE

"""
Start of important code, adjust settings here:
"""

# Version string to use in file names
version = "final_v2"
# File name for Figure 1 and some other outputs, and Figure 2+3.
# (For historical reasons I denote 100 and 1000 FRBs as n3 and n6, respevtively.)
# Colors used for Figure 2+3 to be grayscale friendly
color1 = plt.cm.tab20b.colors[0]
color2 = plt.cm.tab20b.colors[8]

# Whether or not to generate new samples from the FlexKnot prior, and make new fits for cancelling the prior effect
regeneratePriors = False #new prior samples
regenerateTau = False #recompute tau for prior samples
redoFits = False #redo prior distribution fits

# Number of prior samples
Nprior = int(1e6)
# Option to skip slow and memory-intensive Figure 1
Fig1_skip = False
# Option to run a low-resolution version instead
Fig1_low_res = False
# Load extra plots (seed 2 and 3, and the ones with 5x planck prior)
load_extra = False
# Load very large files (seed 1)
load_large = False



print("========== Setting up ==========")

F = cw.cosmology(modules='newfrb', params=deepcopy(cw.planck18_class_bestfit_ncdm))

def read_reionization_history():
    reio_history = np.genfromtxt("kulkarni_et_al_2019_reionization.txt").T
    zreio =list(reio_history[0])
    xreio = list(reio_history[1])
    logreio_history_interp = sip.interp1d(zreio, np.log(xreio), kind="linear", fill_value="extrapolate")
    linreio_history_interp = sip.interp1d([14, 15], [np.exp(logreio_history_interp(14)), 0], kind="linear", fill_value="extrapolate")
    reio_history_interp = lambda z: np.exp(logreio_history_interp(z))*np.heaviside(14-z, 0)+linreio_history_interp(z)*np.heaviside(z-14, 1)
    zreio.append(15)
    xreio.append(0)
    zreio = np.array(zreio)
    xreio = np.array(xreio)
    def xi_phys(z):
        r = reio_history_interp(z)
        return np.minimum(1, np.maximum(0, r))
    return xi_phys

xi_phys = read_reionization_history()
xe_kulk = F.xefunc_of_xifunc(xi_phys)

triangle = ['Omega_b_over_h','Omega_m','tau']

# Fiducial values
fid = deepcopy(cw.planck18_class_bestfit_ncdm)
fid['H0'] = 100*fid["h"]
fid['Omega_b_over_h'] = fid['omega_b']/fid['h']**3
fid['Omega_b'] = fid['omega_b']/fid['h']**2
fid['Omega_b_H0'] = fid['Omega_b']*fid['H0']
fid['Omega_m'] = (fid["omega_cdm"]+fid["omega_b"])/fid['h']**2
fid['tau'] = F.optical_depth_of_xi(xi_phys, zlow=0, zhigh=30)
fid['tau05'] = F.optical_depth_of_xi(xi_phys, zlow=0, zhigh=5)
fid['tau510'] = F.optical_depth_of_xi(xi_phys, zlow=5, zhigh=10)
fid['tau1015'] = F.optical_depth_of_xi(xi_phys, zlow=10, zhigh=15)
fid['tau1030'] = F.optical_depth_of_xi(xi_phys, zlow=10, zhigh=30)
fid['tau010'] = F.optical_depth_of_xi(xi_phys, zlow=0, zhigh=10)
fid['tau1530'] = F.optical_depth_of_xi(xi_phys, zlow=15, zhigh=30)
fid['logA'] = np.log(1e10*fid['A_s']),

for xi in [0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95]:
    key = "z_at_xi"+str(xi)
    f = lambda z: xi_phys(z)-xi
    fid[key] = sop.bisect(f, 5, 30)


print("======== Reading Planck ========")
# Chains obtained from Planck Legacy Archive website

def readPlanck(root='chains/Planck/base_plikHM_TTTEEE_lowl_lowE_lensing_post_BAO', label='Planck 2018'):
    planck = MCMCSamples(root=root, label=label,
        burn_in=0.3)
    planck['Omega_m'] = planck['omegam']
    planck['Omega_b_over_h'] = planck['omegabh2']/(planck['H0']/100)**3
    planck['Omega_b_H0'] = planck['omegabh2']/(planck['H0']/100)*100
    planck['Omega_c'] = planck['omegach2']/(planck['H0']/100)**2
    planck.tex['Omega_m'] = r'$\Omega_m$'
    planck.tex['Omega_b_over_h'] = r'$\Omega_b/h$'
    planck.tex['Omega_b_H0'] = r'$\Omega_b\cdot H_0$'
    return planck

def printMeanStd(sample, key, label=None):
    label = sample.tex[key] if label is None else label
    print('# {0:} = {1:.5f} +/- {2:.5f} ({3:.2f}%)'.format(label, sample[key].mean(), sample.std()[key], sample[key].std()/sample[key].mean()*100))
    return sample[key].mean(), sample[key].std()

planck = readPlanck()
planckmnu = readPlanck(root='chains/Planck/base_mnu_plikHM_TTTEEE_lowl_lowE_lensing_BAO', label="Planck 2018 + mnu, incl. lensing + BAO")
print('Planck priors for other runs:')
printMeanStd(planck, "Omega_m", label="Omega_m")
printMeanStd(planck, "Omega_b_H0", label="Omega_b H0")
printMeanStd(planck, "omegabh2", label="Omega_b h^2")
printMeanStd(planck, "Omega_b_over_h", label="Omega_b/h")


print("=== Getting FlexKnot Priors  ===")

# Note on notation: There are two ways I have been using FlexKnot in the past. Either with
# fixed endpoints at z=5 and 30, or with the endpoints having a variable redshift.
# I only use the latter now ("varyzend=True") but most of the code is compatible with both.

# This also means "number of knots" is sometimes a confusing unit, so that I decided to use
# the less ambigous number of "half degrees of freedom" (hdof) = 2 * number of parameters.
# Here, n_knots = hdof+1, because we have the movable end-points with one degree of freedom each.

def propertiesFromKeys(keys, debug=False):
    # Extract number of knots and mode used (fixed or varying endpoints) from key list
    if debug:
        print("Adjusting to find hdof")
    if "x1" in keys:
        varyzend = False
        for f in range(20):
            if "z"+str(f) in keys:
                hdof = f
    elif "z1" in keys:
        varyzend = True
        hdof = 1
        for f in range(20):
            if "x"+str(f+1) in keys:
                hdof = f+1
    else:
        assert False, ("No hdof keys found", keys)
    return hdof, varyzend

# Here are 2 different implementations of xi(z). This simple one in a
# form that fgivenx can use, with the parameters passed as an array:
def xifunc(z, p, keys, usePCHIP=False, return_func=False, debug=False):
    # Check how many knots and which mode based on keys
    hdof, varyzend = propertiesFromKeys(keys, debug=debug)
    if debug:
        print("Using hdof =",hdof,"and varyzend =", varyzend)
    # Interpolate FlexKnot
    xfull = [1,1]
    if not varyzend:
        zfull = [0,5]
        for i in range(hdof):
            zfull.append(*p[np.where(keys=='z'+str(i+1))])
            xfull.append(*p[np.where(keys=='x'+str(i+1))])
        zfull.append(30)
    else:
        zfull = [0, *p[np.where(keys=='z1')]]
        for i in range(hdof-1):
            zfull.append(*p[np.where(keys=="z"+str(i+2))])
            xfull.append(*p[np.where(keys=="x"+str(i+2))])
        zfull.append(*p[np.where(keys=="z"+str(hdof+1))])
    xfull.append(0)
    # Add 30 - 50 for integrations
    zfull.append(50)
    xfull.append(0)
    zfull = np.array(zfull)
    xfull = np.array(xfull)
    assert np.all(np.diff(xfull)<=0), xfull
    assert np.all(np.diff(zfull)>=0), zfull
    if usePCHIP:
        custom_xifunc = sip.pchip(zfull, xfull)
    else:
        custom_xifunc = sip.interp1d(zfull, xfull)
    if return_func:
        return custom_xifunc
    else:
        return custom_xifunc(z)

# This one provides a vectorized implementation, a nice equation that resembles
# this parallelized form can be found in equaton (14) from arXiv:1908.00906
def vectorized_x_of_z(zarr, samples, invert=False):
    df = deepcopy(samples)
    hdof, varyzend = propertiesFromKeys(df.keys())
    if varyzend:
        df['z0'] = 0
        df['z'+str(hdof+2)] = 50
        df['x0'] = 1
        df['x1'] = 1
        df['x'+str(hdof+1)] = 0
        df['x'+str(hdof+2)] = 0
        nmax = hdof+2
    else:
        df['z0'] = 0
        df['z1'] = 5
        df['z'+str(hdof+2)] = 30
        df['z'+str(hdof+3)] = 50
        df['x0'] = 1
        df['x1'] = 1
        df['x'+str(hdof+2)] = 0
        df['x'+str(hdof+3)] = 0
        nmax = hdof+3
    if not invert:
        yi = np.array([df["x"+str(i)] for i in range(nmax)])
        yip1 = np.array([df["x"+str(i+1)] for i in range(nmax)])
        xi = np.array([df["z"+str(i)] for i in range(nmax)])
        xip1 = np.array([df["z"+str(i+1)] for i in range(nmax)])
    else:
        yi = np.array([df["z"+str(nmax-i)] for i in range(nmax)])
        yip1 = np.array([df["z"+str(nmax-i-1)] for i in range(nmax)])
        xi = np.array([df["x"+str(nmax-i)] for i in range(nmax)])
        xip1 = np.array([df["x"+str(nmax-i-1)] for i in range(nmax)])
    shape = np.shape(xi)
    x = np.multiply.outer(zarr, np.ones(shape))
    dxp1 = xip1 - x
    dx = x - xi
    frac = (yi*dxp1+yip1*dx)/(xip1 - xi)
    mask = np.heaviside(dxp1,0)*np.heaviside(dx,1)
    frac[np.logical_not(mask)] = 0
    assert np.all(np.sum(mask, axis=1)==1)
    f_of_z = np.sum(frac*mask, axis=1)
    return f_of_z

# Analogous functions for the tanh reionization history
def tanh_vectorized_x_of_z(zarr, samples, invert=False):
    z = np.multiply.outer(zarr, np.ones(len(samples['deltaz'])))
    xi = lambda zreio, deltaz: F.xifunc_Planck(zreio1=zreio, dz=deltaz)(z)
    return xi(samples["zreio"], samples["deltaz"])

# Functions to add derived variables to samples
def addExtraTauColumnsPosterior(samples):
    assert 'tau05' in samples, "No redshift dependent tau available"
    samples['tau010'] = np.sum([samples['tau05'],samples['tau510']], axis=0)
    samples.tex['tau010'] = r"$\tau_{0,10}$"
    samples['tau1030'] = np.sum([samples['tau1015'],samples['tau1530']], axis=0)
    samples.tex['tau1030'] = r"$\tau_{10,30}$"

def addIonizedFractionColumns(samples):
    zarr = np.array([int(z) for z in np.arange(5,30.1,1)])
    xarr = vectorized_x_of_z(zarr, samples)
    for i in range(len(zarr)):
        key = "xi_at_z_"+str(zarr[i])
        samples[key] = xarr[i]
        samples.tex[key] = r"$x_i(z="+str(zarr[i])+r")$"
        samples.limits[key] = [0,1]

def addMidpointRedshiftColumns(samples):
    xarr = [0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95]
    np.arange(0.05,0.99,0.05)
    zarr = vectorized_x_of_z(xarr, samples, invert=True)
    for i in range(len(zarr)):
        key = "z_at_xi"+str(xarr[i])
        samples[key] = zarr[i]
        samples.tex[key] = r"$z(x="+str(xarr[i])+r")$"
        samples.limits[key] = [5,30]

# Computation of optical depth using Legendre-Gauss quadrature

def legendreSamples(zmin=0, zmax=30, size=999):
    x_tmp, fast_tau_weights = np.polynomial.legendre.leggauss(size)
    fast_tau_z_arr = zmin+(zmax-zmin)/2*(x_tmp+1)
    fast_tau_weights *= (zmax-zmin)/2
    return fast_tau_z_arr, fast_tau_weights

# Compute tau for FlexKnot prior investigation (fixed Omega and H)
def tauOfArraysPriors(zint, xiint):
    Omega_m = fid['Omega_m']
    Omega_b_over_h = fid['Omega_b_over_h']
    prefactor_int = F.toVal(F.n_H(0) * aco.c * aco.sigma_T / F.H(0), aun.one) \
                    * fid["h"] / fid["Omega_b"] * Omega_b_over_h \
                    * (1+zint)**2/np.sqrt(Omega_m*(1+zint)**3+(1-Omega_m))
    baseline_int = prefactor_int*F.xeHeII_Planck(zint)
    reio_int = (1+F.He_to_H_number_ratio) * np.multiply(prefactor_int, xiint)
    return reio_int+baseline_int

# Compute tau for FlexKnot prior investigation (fixed Omega and H)
# Only used for tanh in the end due to high memory usage.
def vectorizedTauPriors(samples, zmin=0, zmax=30, size=999, method='leggauss', function="FlexKnot"):
    if function=="FlexKnot":
        f = vectorized_x_of_z
    elif function=="DaiXia":
        f = dai_vectorized_x_of_z
    elif function=="Tanh":
        f = tanh_vectorized_x_of_z
    # Memory usage ~ 4GB * size/10,000 * (2+hdof)/3 * len(sample)/2500
    if method=="leggauss":
        zleg, wleg = legendreSamples(zmin=zmin, zmax=zmax, size=size)
        xileg = f(zleg, samples).T
        tauarr = tauOfArraysPriors(zleg, xileg)
        return np.sum(tauarr*wleg, axis=-1), None
    elif method=="minmax":
        zint = np.linspace(zmin, zmax, size)
        xiint = f(zint, samples).T
        tauarr = tauOfArraysPriors(zint, xiint)
        wint = (zmax-zmin)/len(zint)
        wint2 = (zmax-zmin)/(len(zint)-1)
        return np.sum(tauarr*wint, axis=-1), [np.sum(tauarr[:,1:]*wint2, axis=-1), np.sum(tauarr[:,:-1]*wint2, axis=-1)]

# Different implementation of the tau computation to save memory
fast_tau_prefactor = F.toVal(F.n_H(0)*aco.c*aco.sigma_T/F.H(0), aun.one)*fid["h"]/fid["Omega_b"]
x_tmp, fast_tau_weights = np.polynomial.legendre.leggauss(999)
fast_tau_z_arr = 0+(50-0)/2*(x_tmp+1)
fast_tau_weights *= (50-0)/2
def fast_tau(Omega_b_over_h, Omega_m, *args, varyzend=True, usePCHIP=False):
    interp = sip.pchip if usePCHIP else sip.interp1d
    zargs = [0, *args[:int(len(args)/2)+1], 50] if varyzend else [0, 5, *args[:int(len(args)/2)], 30, 50]
    xargs = [1,1,*args[int(len(args)/2)+1:],0,0] if varyzend else [1,1,*args[int(len(args)/2):],0,0]
    custom_xifunc = interp(zargs, xargs)
    custom_xefunc = lambda z: F.xe1_of_xi(custom_xifunc(z)) + F.xeHeII_Planck(z)
    f = lambda z: custom_xefunc(z)*(1+z)**2/np.sqrt(Omega_m*(1+z)**3+(1-Omega_m))
    tau = fast_tau_prefactor*Omega_b_over_h*np.sum(f(fast_tau_z_arr)*fast_tau_weights)
    return tau
vfast_tau = np.vectorize(fast_tau)

# Add those derived tau columns to posterior runs
def addBaseTauColumnsPriors(samples):
    if 'tau56' in samples:
        print("tau56 already present, skipping.")
        return None
    samples['tau05'], _ = vectorizedTauPriors(samples, zmin=0, zmax=5)
    samples['tau510'], _ = vectorizedTauPriors(samples, zmin=5, zmax=10)
    samples['tau1015'], _ = vectorizedTauPriors(samples, zmin=10, zmax=15)
    samples['tau1530'], _ = vectorizedTauPriors(samples, zmin=15, zmax=30)
    samples['tau030'] = samples['tau05']+samples['tau510']+samples['tau1015']+samples['tau1530']
    samples['tau010'] = samples['tau05']+samples['tau510']
    samples['tau015'] = samples['tau05']+samples['tau510']+samples['tau1015']
    samples.tex['tau05'] = r"$\tau_{0,5}$"
    samples.tex['tau510'] = r"$\tau_{5,10}$"
    samples.tex['tau1015'] = r"$\tau_{10,15}$"
    samples.tex['tau1530'] = r"$\tau_{15,30}$"
    samples.tex['tau010'] = r"$\tau_{0,10}$"
    samples.tex['tau015'] = r"$\tau_{0,15}$"

# Generate prior samples to be able to correct for the priors
def generatePriorFlexKnotSamples(hdof, varyzend, size=10000):
    # Note: These conditional priors have Omega fixed, can also generate
    # marginalized samples by sampling Omegas from random.normal instead.
    # The conditional priors though are those one would expect to be flat.
    np.random.seed(20200201+2*hdof+int(varyzend))
    tmp = {}
    if not varyzend:
        # These keys are only required for the formula below and removed afterwards
        tmp['x0'] = 1
        tmp['z0'] = 5.001
        for i in range(hdof):
            j = str(i+1)
            k = str(i)
            tmp['u'+str(j)] = np.random.uniform(low=0, high=1, size=size)
            tmp['v'+str(j)] = np.random.uniform(low=0, high=1, size=size)
            tmp['x'+str(j)] = eval("tmp['x"+k+"']*tmp['u"+j+"']**(1/("+str(hdof)+"-"+j+"+1))")
            tmp['z'+str(j)] = eval("tmp['z"+k+"']+(30-tmp['z"+k+"'])*(1-tmp['v"+j+"']**(1/("+str(hdof)+"-"+j+"+1)))")
        tmp.pop('x0')
        tmp.pop('z0')
    else:
        tmp['z0'] = 5.001
        tmp['x1'] = 1
        for i in range(hdof+1):
            j = str(i+1)
            k = str(i)
            tmp['v'+j] = np.random.uniform(low=0, high=1, size=size)
            tmp['z'+j] = eval("tmp['z"+k+"']+(30-tmp['z"+k+"'])*(1-tmp['v"+j+"']**(1/("+str(hdof+1)+"-"+j+"+1)))")
            tmp.pop('v'+j)
        for i in range(hdof-1):
            j = str(i+1)
            k = str(i)
            k2 = str(i+1)
            tmp['u'+j] = np.random.uniform(low=0, high=1, size=size)
            tmp['x'+str(i+2)] = eval("tmp['x"+k2+"']*tmp['u"+j+"']**(1/("+str(hdof-1)+"-"+j+"+1))")
            tmp.pop('u'+j)
        tmp.pop('x1')
        tmp.pop('z0')
    tmp["Omega_b_over_h"] = fid["Omega_b_over_h"]*np.ones(size)
    tmp["Omega_m"] = fid["Omega_m"]*np.ones(size)
    return MCMCSamples(tmp)

# Generate prior samples for tanh
def generatePriorTanhSamples(size=10000):
    np.random.seed(20200201+size)
    tmp = {}
    tmp["zreio"] = np.random.uniform(low=5, high=15, size=size)
    tmp["deltaz"] = np.random.uniform(low=0.1, high=3, size=size)
    tmp["Omega_b_over_h"] = fid["Omega_b_over_h"]*np.ones(size)
    tmp["Omega_b_H0"] = fid["Omega_b_H0"]*np.ones(size)
    tmp["Omega_m"] = fid["Omega_m"]*np.ones(size)
    tmp["tau"], _ = vectorizedTauPriors(tmp, function="Tanh")
    data = MCMCSamples(tmp)
    data.tex = texDict
    return data

# Here we actually generate the prior samples.
# As we need large amounts of memory we use the other
# function to compute tau. Here's a check that the return
# equal results:
def checkEqualTauFormulae():
    hdof = 3
    varyzend = True
    tmp = generatePriorFlexKnotSamples(hdof, varyzend, size=77)
    with multiprocessing.Pool() as pool:
        zxargs = [*[tmp['z'+str(i+1)] for i in range(hdof+1)], *[tmp['x'+str(i+2)] for i in range(hdof-1)]] if varyzend else [*[tmp['z'+str(i+1)] for i in range(hdof)], *[tmp['x'+str(i+1)] for i in range(hdof)]]
        res = pool.starmap(vfast_tau, np.array([tmp["Omega_b_over_h"], tmp["Omega_m"], *zxargs]).T)
        tmp['tau'] = np.array(res)
    tmp2 = generatePriorFlexKnotSamples(hdof, varyzend, size=77)
    addBaseTauColumnsPriors(tmp2)
    assert np.allclose(tmp["tau"],tmp2["tau030"], atol=0.0001)

checkEqualTauFormulae()

print("Getting FlexKnot prior samples")

priorSamples = []
for hdof in np.arange(1,11,1):
    print("hdof =", hdof)
    filename = "cache/priors_"+version+"hdof"+str(hdof)+"Nprior"+str(Nprior)+".hdf"
    if regeneratePriors:
        print("Generating prior data for hdof =", hdof)
        tmp = generatePriorFlexKnotSamples(hdof, True, size=Nprior)
        print("Adding xi")
        addIonizedFractionColumns(tmp)
        print("Adding z")
        addMidpointRedshiftColumns(tmp)
        s = MCMCSamples(tmp)
        s.weights = np.ones(len(s))
        s["logL"] = np.zeros(len(s))
        s.to_hdf(filename, key='priors')
    else:
        print("  Loading", filename)
        s = MCMCSamples(pd.read_hdf(filename))
    priorSamples.append(s)

# Add tau afterwards
if regenerateTau:
    print("Adding tau to FlexKnot prior samples")
    priorSamples = []
    for hdof in np.arange(1,11,1):
        filename = "cache/priors_"+version+"hdof"+str(hdof)+"Nprior"+str(Nprior)+".hdf"
        tmp = MCMCSamples(pd.read_hdf(filename))
        if "tau" in tmp.keys():
            print("Skipping hdof =", hdof)
            continue
        print("Adding tau for hdof =", hdof)
        #addBaseTauColumnsPriors(tmp)
        varyzend = True
        zxargs = [*[tmp['z'+str(i+1)] for i in range(hdof+1)], *[tmp['x'+str(i+2)] for i in range(hdof-1)]] if varyzend else [*[tmp['z'+str(i+1)] for i in range(hdof)], *[tmp['x'+str(i+1)] for i in range(hdof)]]
        with multiprocessing.Pool() as pool:
            res = pool.starmap(vfast_tau, np.array([tmp["Omega_b_over_h"], tmp["Omega_m"], *zxargs]).T)
            tmp['tau'] = np.array(res)
        tmp.to_hdf(filename, key='priors')
        priorSamples.append(tmp)

# Fit the prior PDFs using a kde
def fitPrior(calib, plot=False, key='tau', bw_method=None, label=""):
    # Evaluating the kde is the slowest part of this function
    if key=="tau":
        xinterp = np.concatenate([np.arange(0,0.1,0.001), np.linspace(0.1,0.3,100)])
        cutoff = 1e-3
    elif "z_at" in key:
        cutoff = 1e-5
        xinterp = np.concatenate([np.arange(5,10,0.05), np.linspace(10,30,100)])
    else:
        assert False, key
    lo, up = np.min(xinterp), np.max(xinterp)
    k = sst.gaussian_kde(calib[key], weights=calib.weights, bw_method=bw_method)
    f = k(xinterp)
    fit = sip.interp1d(xinterp,np.maximum(f, cutoff), fill_value=np.inf, bounds_error=False)
    norm = sin.quad(fit, lo, up)[0]
    fit = sip.interp1d(xinterp,np.maximum(f, cutoff)/norm, fill_value=np.inf, bounds_error=False)
    if plot:
        fig, ax = plt.subplots()
        fig.suptitle(label+" "+key.replace("_", ""))
        xplot = np.linspace(np.min(xinterp),np.max(xinterp),1000)
        ax.plot(xplot, fit(xplot), label='scipy kde with bw method = '+str(bw_method))
        ax.hist(calib[key], weights=calib.weights, density=True, bins=1000, range=(lo, up), label='samples', alpha=0.5)
        ax.set_xlabel(key.replace("_", ""))
        ax.semilogy()
        if key=="tau":
            ax.set_ylim(1e-4,1e3)
        else:
            ax.set_ylim(1e-6,1e0)
        ax.axvline(fid[key], ls='dashed', color='k',label='True value (i.e. important around here)')
        fig.legend()
        plt.show()
    return fit

print("Fitting FlexKnot priors")
prior_fits = {"tau": [], "z_at_xi0.1": [], "z_at_xi0.5": [], "z_at_xi0.9": []}
if redoFits=="useOld":
    print("Loading old fits")
    tauinterp = np.load("prior_runs/tau_interp.npy")
    zinterp = np.load("prior_runs/z_interp.npy")
    file_names = {"tau": "fit", "z_at_xi0.1": "fit_Xi010", "z_at_xi0.5": "fit_Xi050", "z_at_xi0.9": "fit_Xi090"}
    for hdof in np.arange(1,11,1):
        for key in prior_fits.keys():
            x = tauinterp if "tau" in key else zinterp
            filename = "prior_runs/"+file_names[key]+"_hdof"+str(hdof)+"_delta_varyzend_linear_N1000000.npy"
            print("Loading", filename)
            prior_fits[key].append(sip.interp1d(x, np.load(filename)))
elif redoFits:
    for i in range(len(priorSamples)):
        hdof = i+1
        print("Fitting hist #"+str(i+1))
        for key in prior_fits.keys():
            bw_method = 0.02 if key=="tau" else 0.05
            if key=="z_at_xi0.1" and hdof>5:
                bw_method = 0.1
            if key=="tau" and hdof>5:
                bw_method = 0.04
            fit = fitPrior(priorSamples[i], label="hdof="+str(i+1), key=key, bw_method=bw_method, plot=False)
            prior_fits[key].append(fit)
            x = np.linspace(0,1,10000) if key=="tau" else np.linspace(5,30,10000)
            filename = "cache/fit_"+key+version+"hdof"+str(hdof)+".npy"
            print("Saving to", filename)
            np.save(filename, fit(x))
else:
    print("Loading prior fits")
    for i in range(len(priorSamples)):
        hdof = i+1
        for key in prior_fits.keys():
            x = np.linspace(0,1,10000) if key=="tau" else np.linspace(5,30,10000)
            filename = "cache/fit_"+key+version+"hdof"+str(hdof)+".npy"
            print("Loading from", filename)
            y = np.load(filename)
            fit = sip.interp1d(x,y)
            prior_fits[key].append(fit)

print("Generating tanh priors")
priorSamplesTanh = generatePriorTanhSamples(size=100000)
priorFitTanh = fitPrior(priorSamplesTanh, plot=False, key='tau', bw_method=0.05, label='tanh')


print("====== Reading posteriors ======")

def reweighNested(samples, key=None, fit=None, plot=False):
    samples = deepcopy(samples)
    tex = samples.tex
    assert key in samples.keys(), "Missing "+key
    samples = MCMCSamples(samples)
    taulogpdf_f = lambda x: np.log(fit(x))
    taulogpdf = (taulogpdf_f(samples[key]))
    newsamples, deltaLogZ = importance_sample_withLogZ(samples, -taulogpdf)
    newsamples.tex = tex
    if plot:
        print(np.shape(newsamples), np.shape(newsamples.weights))
        plt.hist(newsamples[key], weights=newsamples.weights, density=True, bins=200, alpha=0.5, label='Ratio from "fit" reweigh', range=(0.03, 0.2))
        assert np.all(np.array(samples[key])==np.array(prior[key]))
        prtau,xtau,_ = plt.hist(samples[key], weights=prior.weights, density=True, bins=200, alpha=0.5, label='Prior', range=(0.03, 0.2))
        potai,xtau_,_ = plt.hist(samples[key], weights=samples.weights, density=True, bins=200, alpha=0.5, label='Post', range=(0.03, 0.2))
        assert np.all(xtau_==xtau), (xtau,xtau_)
        xtau = np.array(xtau)
        mids = (xtau[1:]+xtau[:-1])/2
        plt.scatter(mids, potai/prtau, color='darkgreen', label='Ratio of bins')
        xplot = np.linspace(0.03+np.pi/1e5,0.2+np.pi/1e5,10000)
        plt.plot(xplot, np.exp(taulogpdf_f(xplot)), color='brown', label='Fit')
        plt.xlim(0.04, 0.1)
        plt.semilogy()
        plt.ylim(1e-7, 1e3)
        plt.legend()
        plt.show()
    return newsamples, deltaLogZ

def importance_sample_withLogZ(samples, deltaLogL, action='add', inplace=False):
    indices = np.array(samples.index.to_list())[:,0]
    drop_mask = samples.weights == 0
    drop_samples = indices[drop_mask]
    if len(drop_samples)>0:
        print("Warning: Dropping", len(drop_samples), "out of", len(samples), "samples due to weights=0")
        samples = samples.drop(index=drop_samples, inplace=False)
        if isinstance(deltaLogL, pd.DataFrame):
            deltaLogL = deltaLogL.drop(index=drop_samples, inplace=False)
        else:
            deltaLogL = np.array(deltaLogL)
            deltaLogL = np.delete(deltaLogL, drop_mask)
    assert np.all(samples.weights != 0)
    deltaLogZ = np.log(np.average(np.exp(deltaLogL), weights=samples.weights))
    samples = MCMCSamples(samples) #to avoid bug in importance sample
    return samples.importance_sample(deltaLogL, action=action, inplace=inplace), deltaLogZ

def readNested(filename, chains="chains_x1/hostHPCx1v2/", columns=None, saveMem=True, fixPriors=True, fixPriorstimesx=1):
    # columns is an optional argument to specify the variable names
    # saveMem drops information from memory (RAM) where the loglikelihood for
    #   every single FRB data point has been saved (does not affect file, just
    #   to save memory)
    # fixPriors corrects our priors from Omega_b / H_0 to Omega_b * H_0
    # fixPriorstimesx = 5 adjusts this for the "5xplanck" prior.
    root = chains+"/"+filename+"_polychord_raw/"+filename
    print(filename, end='')
    samples = NestedSamples(root=root) if columns is None else NestedSamples(root=root, columns=columns)
    if saveMem:
        for key in list(samples.keys()):
            if "logL_z" in key:
                del samples[key]
    print("(", len(samples), ")")
    samples['Omega_b_H0'] = samples['Omega_b_over_h']*fid['h']**2*100
    samples.tex['Omega_m'] = r'$\Omega_m$'
    samples.tex['Omega_b_over_h'] = r'$\Omega_b/h$'
    samples.tex['Omega_b_H0'] = r'$\Omega_b\cdot H_0$'
    if fixPriors:
        def oldLogPrior(Omega_b_over_h, Omega_m):
            return sst.norm.logpdf(Omega_b_over_h, loc=0.0724, scale=0.0011*fixPriorstimesx)+sst.norm.logpdf(Omega_m, loc=0.31108, scale=0.00555*fixPriorstimesx)
        def newLogPrior(Omega_b_H0, Omega_m):
            return sst.norm.logpdf(Omega_b_H0, loc=3.31, scale=0.0167*fixPriorstimesx)+sst.norm.logpdf(Omega_m, loc=0.31108, scale=0.00555*fixPriorstimesx)
        offset = (samples["logprior__0"]-oldLogPrior(samples["Omega_b_over_h"], samples["Omega_m"])).mean()
        assert np.allclose(samples["logprior__0"], offset+oldLogPrior(samples["Omega_b_over_h"], samples["Omega_m"]), atol=1e-3)
        samples.drop("Omega_b_over_h", axis=1, inplace=True)
        samples["logprior_old"] = samples["logprior__0"]
        samples.drop("logprior__0", axis=1, inplace=True)
        samples["logprior_new"] = offset+newLogPrior(samples["Omega_b_H0"], samples['Omega_m'])
        samples.weights *= np.exp(samples["logprior_new"]-samples["logprior_old"])

    planckstring='_pprior'
    if "z1" in samples.keys():
        addIonizedFractionColumns(samples)
        addMidpointRedshiftColumns(samples)
    if "tau05" in samples.keys():
        addExtraTauColumnsPosterior(samples)
    return samples

def remove_etl_withLogZ(chain):
    tmp = MCMCSamples(deepcopy(chain))
    assert "logL_tau1530" in tmp.keys()
    add = np.array(-tmp.logL_tau1530+np.max(tmp.logL_tau1530))
    add[np.where(tmp.weights==0)]=0
    tmp, deltaLogZ = importance_sample_withLogZ(tmp, np.maximum(add, -10), action='add')
    tmp.tex = chain.tex
    tmp.limits = chain.limits
    return tmp, deltaLogZ

def readAllFlexNest(filenames, chains=None, reweigh=False, reweighFits=None, plot=False):
    runs = []
    logZs = []
    for i in range(len(filenames)):
        f = filenames[i]
        s = readNested(f, chains=chains)
        logZs.append(s.logZ())
        runs.append(s)
    logZs = np.array(logZs)-np.max(logZs)
    merge = merge_samples_weighted(runs, weights=np.exp(logZs))
    data = {"chains":pd.Series(runs), "mergedChain":merge, "logZs":pd.DataFrame(logZs, columns=['logZ'])}

    # Generate reweighed chain without Planck tau1530 data
    if "logL_tau1530" in runs[0].keys():
        noETLruns = []
        noETLlogZs = []
        for i in range(len(filenames)):
            noETL, dLogZ = remove_etl_withLogZ(deepcopy(runs[i]))
            noETLruns.append(noETL)
            noETLlogZs.append(logZs[i]+dLogZ)
        data["rwChains_tauNoETL"] = pd.Series(noETLruns)
        data["rwLogZs_tauNoETL"] = pd.DataFrame(noETLlogZs, columns=['logZ'])
        noETLmerge = merge_samples_weighted(noETLruns, weights=np.exp(noETLlogZs))
        data["rwMergedChain_tauNoETL"] = noETLmerge

    # Reweigh for flat priors
    if reweigh:
        for key in reweighFits.keys():
            rwRuns = []
            rwLogZs = []
            for i in range(len(filenames)):
                rwRun, dLogZ = reweighNested(runs[i], key=key, fit=reweighFits[key][i], plot=plot)
                rwRuns.append(rwRun)
                rwLogZs.append(logZs[i]+dLogZ)
            data["rwChains_"+key] = pd.Series(rwRuns)
            data["rwLogZs_"+key] = pd.DataFrame(rwLogZs, columns=['logZ'])
            rwMerge = merge_samples_weighted(rwRuns, weights=np.exp(rwLogZs))
            data["rwMergedChain_"+key] = rwMerge
    return data

# --- Load all the NestedSampling chains ---

chains_tanh = "chains/FRBs/tanh_chains"
x1_pp_n3_vzend_eprior_etl_tanh_dz = readNested("run_hostcalx_try2_polyba_Mockdata_phys_sfr_z10%_norm100_obs1_vardz_pprior_etl_eprior", chains=chains_tanh)
x1_pp_n6_vzend_eprior_etl_tanh_dz = readNested("run_hostcalx_try2_polyba_Mockdata_phys_sfr_z10%_norm1000_obs1_vardz_pprior_etl_eprior", chains=chains_tanh)
x1_pp_n9_vzend_eprior_etl_tanh_dz = readNested("run_hostcalx_try2_polyba_Mockdata_phys_sfr_z10%_norm10000_obs1_vardz_pprior_etl_eprior", chains=chains_tanh)
x1_pp_n3_vzend_eprior_etl_tanh_dz_rw, _ = reweighNested(x1_pp_n3_vzend_eprior_etl_tanh_dz, key="tau", fit=priorFitTanh, plot=False)
x1_pp_n6_vzend_eprior_etl_tanh_dz_rw, _ = reweighNested(x1_pp_n6_vzend_eprior_etl_tanh_dz, key="tau", fit=priorFitTanh, plot=False)
x1_pp_n9_vzend_eprior_etl_tanh_dz_rw, _ = reweighNested(x1_pp_n9_vzend_eprior_etl_tanh_dz, key="tau", fit=priorFitTanh, plot=False)

chains_seed0 = "chains/FRBs/flexknot_chains"
n=10; x1_pp_n3_vzend_eprior_etl = readAllFlexNest(["run_hostHPCx1v2_polyba_Mockdata_phys_sfr_z10%_norm100_obs1_monotonousflexknot"+str(f+1)+"_pprior_etl_eprior_vzend"for f in range(n)], chains=chains_seed0, reweigh=True, reweighFits=prior_fits)
n=10; x1_pp_n6_vzend_eprior_etl = readAllFlexNest(["run_hostHPCx1v2_polyba_Mockdata_phys_sfr_z10%_norm1000_obs1_monotonousflexknot"+str(f+1)+"_pprior_etl_eprior_vzend"for f in range(n)], chains=chains_seed0, reweigh=True, reweighFits=prior_fits)

# --- Extra chains to check effects of random seed & priors ---

# New DM samples, same redshifts:
# Note that these two sets (*_seed1) are moved to a separate zenodo data set
# due to their large size (27GB, 71GB uncompressed).
if load_large:
    chains_seed1 = "chains/FRBs/update_seed1"
    n=10; x1_pp_n3_vzend_eprior_etl_seed1 = readAllFlexNest(["run_hostHPCx1v2_s2_polyba_Mockdata_s1_phys_sfr_z10%_norm100_obs1_monotonousflexknot"+str(f+1)+"_pprior_etl_eprior_likez_vzend"for f in range(n)], chains=chains_seed1, reweigh=True, reweighFits=prior_fits)
    n=10; x1_pp_n6_vzend_eprior_etl_seed1 = readAllFlexNest(["run_hostHPCx1v2_s2_polyba_Mockdata_s1_phys_sfr_z10%_norm1000_obs1_monotonousflexknot"+str(f+1)+"_pprior_etl_eprior_likez_vzend"for f in range(n)], chains=chains_seed1, reweigh=True, reweighFits=prior_fits)

# New z *and* DM samples, twice:
if load_extra:
    chains_seed23 = "chains/FRBs/update_seed23"
    n=10; x1_pp_n3_vzend_eprior_etl_seed2 = readAllFlexNest(["run_hostHPC22_polyba_Mockdata_phys_seed2_sfr_z10%_norm100_obs1_monotonousflexknot"+str(f+1)+"_pprior_etl_eprior_vzend"for f in range(n)], chains=chains_seed23, reweigh=True, reweighFits=prior_fits)
    n=10; x1_pp_n6_vzend_eprior_etl_seed2 = readAllFlexNest(["run_hostHPC22_polyba_Mockdata_phys_seed2_sfr_z10%_norm1000_obs1_monotonousflexknot"+str(f+1)+"_pprior_etl_eprior_vzend"for f in range(n)], chains=chains_seed23, reweigh=True, reweighFits=prior_fits)
    n=10; x1_pp_n3_vzend_eprior_etl_seed3 = readAllFlexNest(["run_hostHPC22_polyba_Mockdata_phys_seed3_sfr_z10%_norm100_obs1_monotonousflexknot"+str(f+1)+"_pprior_etl_eprior_vzend"for f in range(n)], chains=chains_seed23, reweigh=True, reweighFits=prior_fits)
    n=10; x1_pp_n6_vzend_eprior_etl_seed3 = readAllFlexNest(["run_hostHPC22_polyba_Mockdata_phys_seed3_sfr_z10%_norm1000_obs1_monotonousflexknot"+str(f+1)+"_pprior_etl_eprior_vzend"for f in range(n)], chains=chains_seed23, reweigh=True, reweighFits=prior_fits)

# Extra Planck prior runs with 1x and 5x priors to check prior effects
if load_extra:
    chains_extra_planck = "chains/FRBs/update_planck"
    x1_pp_f4 = readNested("run_hostHPC22_polyba_Mockdata_phys_sfr_z10%_norm1000_obs1_monotonousflexknot4_pprior_etl_eprior_vzend", chains=chains_extra_planck)
    x1_5xpp_f4 = readNested("run_hostHPC22_polyba_Mockdata_phys_sfr_z10%_norm1000_obs1_monotonousflexknot4_5xpprior_etl_eprior_vzend", chains=chains_extra_planck, fixPriorstimesx=5)


# Data selection
#  Fig 3
chainsFig3 = x1_pp_n6_vzend_eprior_etl; stringFig3 = "seed0_"
#  Fig 4 and 5
chain100 = x1_pp_n3_vzend_eprior_etl['rwMergedChain_tau']
chain1k = x1_pp_n6_vzend_eprior_etl['rwMergedChain_tau']

# --- Analyze data ---

print("======= Running analysis =======")

def fastCL(data, key="tau", level=0.68):
    assert level<1, "Level >= 1!"
    samples = data[key]
    weights = data.weights
    # Sort and normalize
    order = np.argsort(samples)
    samples = deepcopy(np.array(samples))[order]
    weights = deepcopy(np.array(weights))[order]/np.sum(weights)
    # Compute inverse cumulative distribution function
    cumsum = np.cumsum(weights)
    #cdf = sip.interp1d(samples, cumsum, fill_value=(0,1), bounds_error=False)
    S = np.array([np.min(samples), *samples, np.max(samples)])
    CDF = np.append(np.insert(np.cumsum(weights), 0, 0), 1)
    invcdf = sip.interp1d(CDF, S)
    # Find smallest interval
    distance = lambda a, level=level: invcdf(a+level)-invcdf(a)
    res = sop.minimize(distance, (1-level)/2, bounds=[(0,1-level)], method="Nelder-Mead")
    return np.array([invcdf(res.x[0]), invcdf(res.x[0]+level)])

def bestFit(data, key, ncompress=10000, plot=False, verbose=True):
    # KDE from anesthetic, courtesy of Lukas Hergt:
    # https://github.com/williamjameshandley/anesthetic/issues/178
    fig, axes = data.plot_1d([key], q=1, density=True, label='anesthetic', ncompress=ncompress)
    x = axes.iloc[0].lines[0].get_xdata()
    y = axes.iloc[0].lines[0].get_ydata()
    mask = [*(np.diff(x)==0), False]
    if np.sum(mask) > 0:
        if verbose:
            print("bestFit: Removing", np.sum(mask), "duplicates from kde interpolation function")
        indices = np.where([*(np.diff(x)==0), False])[0]
        assert np.all(y[indices] == y[indices+1])
        x = x[np.logical_not(mask)]
        y = y[np.logical_not(mask)]
    pdf_kde = sip.interp1d(x,y,kind='cubic', bounds_error=False, fill_value=0)
    if plot:
        axes[key].hist(data[key], weights=data.weights, density=True, label='anesthetic', range=data.limits[key], bins=50)
    else:
        fig.clear()
        plt.close(fig)
    return sop.minimize(lambda x: -pdf_kde(x), fid[key], method="Nelder-Mead").x[0]

print("======= Numbers in paper =======")
print("All numbers are FlexKnot prior-corrected, and we are using the Planck Omegab*H0 priors.")
print("======= Numbers Abstract =======")

bestfit_tau_100FRB = bestFit(x1_pp_n3_vzend_eprior_etl["rwMergedChain_tau"], key='tau', plot=False)
bestfit_tau_1000FRB = bestFit(x1_pp_n6_vzend_eprior_etl["rwMergedChain_tau"], key='tau', plot=False)
CL_tau_100FRB = fastCL(x1_pp_n3_vzend_eprior_etl["rwMergedChain_tau"], key="tau")
CL_tau_1000FRB = fastCL(x1_pp_n6_vzend_eprior_etl["rwMergedChain_tau"], key="tau")
print("    tau accuracy 100 FRBs: {0:.0f}% (in more detail: +{2:.0f}% {1:.0f}%)".format(100/2*np.diff(CL_tau_100FRB)[0]/bestfit_tau_100FRB, *(100*(CL_tau_100FRB-bestfit_tau_100FRB)/bestfit_tau_100FRB)))
print("    tau accuracy 1000 FRBs: {0:.0f}% (in more detail: +{2:.0f}% {1:.0f}%)".format(100/2*np.diff(CL_tau_1000FRB)[0]/bestfit_tau_1000FRB, *(100*(CL_tau_1000FRB-bestfit_tau_1000FRB)/bestfit_tau_1000FRB)))

bestfit_midpoint_100FRB = bestFit(x1_pp_n3_vzend_eprior_etl["rwMergedChain_z_at_xi0.5"], key='z_at_xi0.5', plot=False)
CL_midpoint_100FRB = fastCL(x1_pp_n3_vzend_eprior_etl["rwMergedChain_z_at_xi0.5"], key="z_at_xi0.5")
print("    midpoint accuracy 100 FRBs: {0:.0f}% (in more detail: +{1:.0f}% -{2:.0f}%)".format(100/2*np.diff(CL_midpoint_100FRB)[0]/bestfit_midpoint_100FRB, *(100*(CL_midpoint_100FRB-bestfit_midpoint_100FRB)/bestfit_midpoint_100FRB)))

print("======= Numbers Results  =======")
print("  FlexKnot numbers. Critical points of reionization accuracies (still all prior corrected):")
for i in range(2):
    size = [100, 1000][i]
    data = [x1_pp_n3_vzend_eprior_etl, x1_pp_n6_vzend_eprior_etl][i]
    print(size, "FRBs:")
    for key in ["z_at_xi0.9", "z_at_xi0.5", "z_at_xi0.1"]:
        bf = bestFit(data['rwMergedChain_'+key], key=key)
        diff = np.diff(fastCL(data['rwMergedChain_'+key], key=key, level=0.68))
        print("    Accuracy of "+key+": {0:.0f}%".format(diff[0]/2/bf*100), "(for bestfit)")#. For mean: {0:.0f}%)".format(diff[0]/2/mean*100))

print("  Optical depth (0,30) accuracies (all prior corrected):")
for i in range(2):
    size = [100, 1000][i]
    data = [x1_pp_n3_vzend_eprior_etl, x1_pp_n6_vzend_eprior_etl][i]
    print(size, "FRBs:")
    for key in ["tau", "tauNoETL"]:
        bf = bestFit(data['rwMergedChain_'+key], key="tau")
        diff = np.diff(fastCL(data['rwMergedChain_'+key], key="tau", level=0.68))
        print("    Accuracy of "+key+": {0:.0f}%".format(diff[0]/2/bf*100), "(for bestfit)")#. For mean: {0:.0f}%)".format(diff[0]/2/mean*100))

print("  Optical depth (0,30) accuracies for **TANH** (all prior corrected):")
for i in range(3):
    size = [100, 1000, 10000][i]
    data = [x1_pp_n3_vzend_eprior_etl_tanh_dz_rw, x1_pp_n6_vzend_eprior_etl_tanh_dz_rw, x1_pp_n9_vzend_eprior_etl_tanh_dz_rw][i]
    print(size, "FRBs:")
    key = "tau"
    mean = data[key].mean()
    bf = bestFit(data, key=key)
    std = data[key].std()
    CL = (fastCL(data, key=key, level=0.68))
    CL95 = (fastCL(data, key=key, level=0.95))
    print("    (bestfit) tau = {0:.4f}".format(bf)+"+{0:.4f}".format(CL[1]-bf)+"-{0:.4f}".format(bf-CL[0])+" (std =", std, ")")
    print("    CL 95%:", CL95)

print("  tau(z<10) and tau(z>10) values:")
for i in range(2):
    size = [100, 1000][i]
    data = [x1_pp_n3_vzend_eprior_etl, x1_pp_n6_vzend_eprior_etl][i]["rwMergedChain_tau"]
    print(size, "FRBs:")
    for key in ["tau010", "tau1030"]:
        bf = bestFit(data, key=key)
        std = data[key].std()
        CL = (fastCL(data, key=key, level=0.68))
        print("    (bestfit)", key, "= {0:.4f}".format(bf)+"+{0:.5f}".format(CL[1]-bf)+"-{0:.5f}".format(bf-CL[0])+" (std =", std, ")", "This is {0:.1f}.".format(np.diff(CL)[0]/2*100/bf),"% acc.")


print("  Relative Evidence FlexKnot:")
for i in range(3):
    print("    100 FRBs,", i+2, "knots:","{0:.2f}+/-{1:.2f}".format(np.exp(x1_pp_n3_vzend_eprior_etl['logZs']['logZ'].iloc[i]), np.exp(x1_pp_n3_vzend_eprior_etl['logZs']['logZ'].iloc[i])*np.std(x1_pp_n3_vzend_eprior_etl['chains'][0].logZ(1000))))
for i in range(5):
    print("    1000 FRBs,", i+2, "knots:","{0:.2f}+/-{1:.2f}".format(np.exp(x1_pp_n6_vzend_eprior_etl['logZs']['logZ'].iloc[i]), np.exp(x1_pp_n6_vzend_eprior_etl['logZs']['logZ'].iloc[i])*np.std(x1_pp_n6_vzend_eprior_etl['chains'][0].logZ(1000))))


print("  A_s constraints: See Figure 3 section.")

print("=========== Table 1  ===========")
print("Note: Bestfit results are based on KDE peak, and numerical performance can",
      "vary, leading to negligible changes like tau bestfit = 0.0509 --> 0.0510.")

def newLine(j):
    if j==0:
        print(r" & ", end='')
    elif j==1:
        print(r" \\")
    else:
        assert False

def substack(up, low, d=1):
    if d==1:
        return r"$\substack{+"+"{0:.1f}".format(up)+r"\\-"+"{0:.1f}".format(low)+r"}$"
    else:
        return r"$\substack{+"+"{0:.4f}".format(up)+r"\\-"+"{0:.4f}".format(low)+r"}$"

def printTable():
    for i in range(5):
        key = ["z_at_xi0.1", "z_at_xi0.5", "z_at_xi0.9", "tau", "tauNoETL"][i]
        var_key = ["z_at_xi0.1", "z_at_xi0.5", "z_at_xi0.9", "tau", "tau"][i]
        name = [r"Start ($x_i=0.1$)", r"Midpoint ($x_i=0.5$)", r"End ($x_i=0.9$)", r"Optical depth $\tau$", r"$\tau$ without CMB"][i]
        print(name, end=' & ')
        for j in range(2):
            data = [x1_pp_n3_vzend_eprior_etl, x1_pp_n6_vzend_eprior_etl][j]['rwMergedChain_'+key]
            bf = bestFit(data, key=var_key)
            int1 = fastCL(data, key=var_key, level=0.68)
            int2 = fastCL(data, key=var_key, level=0.95)
            if "xi" in key:
                print("$z="+"{0:.1f}".format(bf)+"$ & "+substack(int1[1]-bf, bf-int1[0])+" & "+substack(int2[1]-bf, bf-int2[0]), end="")
                newLine(j)
            elif "tau" in key:
                print("$"+"{0:.4f}".format(bf)+"$ & "+substack(int1[1]-bf, bf-int1[0], d=4)+" & "+substack(int2[1]-bf, bf-int2[0], d=4), end="")
                newLine(j)
            else:
                assert False

printTable()


print("=========== Figure 3 (was 1) ===========")

# We use always linear interpolation here, not PCHIP.
usePCHIP = False

def fgivenx_plot(chains, logEvidences, ax_fgivenx, cache=None,
                 prior=False, lines=False, **kwargs):
    ny = 100 if prior else 1000
    if Fig1_low_res:
        nx = 10 if prior else 1000
    else:
        nx = 100 if prior else 1000
    zplot = np.linspace(5.01, 29.99, nx)
    if logEvidences is None:
        logEvidences = [c.logZ() for c in chains]
    elif logEvidences=="1":
        logEvidences = [1 for c in chains]
    xifuncs_specific = [lambda z,p, keys=chains[i].keys(): xifunc(z, p, keys=keys, usePCHIP=usePCHIP) for i in range(len(chains))]
    weights = [c.weights for c in chains]
    if not lines:
        cbar = fgivenx.plot_contours(xifuncs_specific, zplot, chains, ax=ax_fgivenx,
            weights=weights, logZ=logEvidences, cache=cache+"{0:.2f}".format(len(zplot)+np.sum(zplot)), ny=ny, **kwargs)
    else:
        cbar = fgivenx.plot_lines(xifuncs_specific, zplot, chains, ax=ax_fgivenx, weights=weights, logZ=logEvidences, cache=cache)
    return cbar


def fgivenx_wrapper(posteriorSims, posteriorSims_logZs,
    priorSamples=None, midpointSimsDict=None,
    figsize=(13, 4), lines=False, **kwargs):
    # posteriorSims, posteriorSims_logZs -- mandatory
    # priorSamples -- optional to plot priors
    # midpointSimsDict -- optional to plot errorbars

    fig, ax_fgivenx = plt.subplots(figsize=figsize)
    ax_fgivenx.set_xlabel("Redshift $z$")
    ax_fgivenx.set_ylabel("Ionized fraction $x_i$")
    ax_fgivenx.set_xlim(5,30)
    plt.grid(ls='dotted')

    # Plot prior
    if priorSamples is not None:
        lenstr = str(np.sum([len(priorSamples[i]) for i in range(len(priorSamples))]))
        cbar_prior = fgivenx_plot(priorSamples, "1", ax_fgivenx,
            "cache/fgivenx_prior_"+lenstr+"/", prior=True,
            alpha=1, colors=plt.cm.Purples_r, lines=lines, **kwargs)

    # Plot posterior
    lenstr = str(np.sum([len(posteriorSims[i]) for i in range(len(posteriorSims))]))
    cbar = fgivenx_plot(posteriorSims, posteriorSims_logZs, ax_fgivenx,
        "cache/fgivenx_post_"+lenstr+"/",
        alpha=0.7, colors=plt.cm.Oranges_r, lines=lines, **kwargs)

    if lines:
        return fig, ax_fgivenx, cbar

    # Plot Kulkarni et al. and see if we do recover it
    ax_fgivenx.plot(np.linspace(5,15,1000), xi_phys(np.linspace(5,15,1000)), label='Fiducial', color="darkturquoise", linestyle='dashed', lw=2, alpha=1)

    # Prepare legend
    colorbar = fig.colorbar(cbar, ticks=[0,1,2], label='Confidence level', pad=0)
    colorbar.set_label("Confidence level", labelpad=0.01)
    colorbar.set_ticklabels(["0", r"68\%", r"95\%"])
    orange_patch = mpatches.Patch(color=colorbar.cmap(0.5), label='Posterior')
    purple_patch = mpatches.Patch(color=plt.cm.Purples_r(0.5), label='Prior')

    # Optionally plot error bars
    if midpointSimsDict is not None:
        xi = [0.1, 0.5, 0.9]
        means = []
        xerr1 = []
        xerr2 = []
        for key in ["z_at_xi0.1", "z_at_xi0.5", "z_at_xi0.9"]:
            midpointSims = midpointSimsDict["rwMergedChain_"+key]
            means.append(midpointSims[key].mean())
            xerr1.append(np.abs(fastCL(midpointSims, key=key, level=0.68)-means[-1]))
            xerr2.append(np.abs(fastCL(midpointSims, key=key, level=0.95)-means[-1]))
        xerr1 = np.array(xerr1)
        xerr2 = np.array(xerr2)
        planckstyle = {"fmt":'o', "markersize":0, "ecolor":'lightblue', "markerfacecolor":'orange', "markeredgecolor":'black', "lw":3}
        ax_fgivenx.errorbar(means, xi, xerr=xerr1.T, **planckstyle, label=r"68\% C.L.", zorder=10)
        planckstyle.update({"ecolor":'blue'})
        ax_fgivenx.errorbar(means, xi, xerr=xerr2.T, **planckstyle, label=r"95\% C.L.")

    # Plot legend
    handles, labels = ax_fgivenx.get_legend_handles_labels()
    axbox = ax_fgivenx.get_position()
    x_value=0.515; y_value=0.5
    loc = (0.57,0.65)
    if priorSamples is not None:
        plt.legend(handles=[orange_patch, purple_patch, *handles], labels=["Posterior", "Prior", *labels])
        colorbar_prior = fig.colorbar(cbar_prior, ticks=[])
    else:
        plt.legend(handles=[orange_patch, *handles], labels=["Posterior", *labels])

    plt.tight_layout()

    # Manual adjusting of colorbar position
    if priorSamples is not None:
        loc = deepcopy(colorbar.ax.get_position())
        loc.x0 -= 0.1
        loc.x1 -= 0.1
        colorbar.ax.set_position(deepcopy(loc))
        loc.x0 += 0.022
        loc.x1 += 0.022
        colorbar_prior.ax.set_position(loc)
        pos=colorbar.ax.get_position()
        pos.x0 = 0.7245576780301003
        pos.x1 = 0.7404006154281132
        colorbar.ax.set_position(deepcopy(pos))
        pos.x1 -= 0.02
        pos.x0 -= 0.02
        colorbar_prior.ax.set_position(pos)

    return fig, colorbar, colorbar_prior


def run_xi_plots(c, s):
    fgivenx_wrapper(c['chains'], list(c['logZs']['logZ']),
        priorSamples=priorSamples, midpointSimsDict=c,
        figsize=(13, 4), fineness=0.25, rasterize_contours=True,
        contour_line_levels=[1,2], linewidths=1)
    if Fig1_low_res:
        s += "_low_res"
    plt.savefig("paper_plots/Fig3_xiplot_"+s+version+".pdf", bbox_inches='tight')
    plt.savefig("paper_plots/Fig3_xiplot_"+s+version+".png", dpi=600, bbox_inches='tight')


if not Fig1_skip:
    run_xi_plots(chainsFig3, stringFig3)
    # For other seeds:
    #run_xi_plots(x1_pp_n6_vzend_eprior_etl_seed1, "seed1_")
    #run_xi_plots(x1_pp_n6_vzend_eprior_etl_seed2, "seed2_")
    #run_xi_plots(x1_pp_n6_vzend_eprior_etl_seed3, "seed3_")


print("=========== Figure 4 (was 2) ===========")

#Planck result
def asym_norm_logpdf(x, mu=0.0504, sigmaleft=0.0079, sigmaright=0.0050):
    heavi = np.heaviside(x-mu, 0.5)
    return np.log((sigmaleft+sigmaright)/np.sqrt(2*np.pi)/sigmaleft/sigmaright)\
    -0.5*(x-mu)**2/sigmaleft**2*(1-heavi)\
    -0.5*(x-mu)**2/sigmaright**2*(heavi)

max_asym_norm_logpdf = np.max(np.exp(asym_norm_logpdf(np.linspace(0.04, 0.06, 1000))))

def run_tau_histogram_plots(c3, c6, bw=True, outfilename=None, line=False):
    c3 = deepcopy(c3); c3.limits['tau'] = [0.03,0.09]
    c6 = deepcopy(c6); c6.limits['tau'] = [0.03,0.09]
    filename = "paper_plots/Fig4_tauplot_"+outfilename
    xplot = np.linspace(0.03, 0.09, 10000)
    if bw:
        filename += "_bw"
        gs = lambda r,g,b: [0.2989 * r + 0.5870 * g + 0.1140 * b]*3
    else:
        gs = lambda r,g,b: (r,g,b)

    fig, axes = c3.plot_1d("tau", label=r"$100$ FRBs", color=gs(*color1), alpha=1, plot_type='hist', bins=50, histtype='step', lw=1.5)
    c6.plot_1d(label=r"$1,000$ FRBs", axes=axes, alpha=0.6, color=gs(*color2), plot_type='hist', bins=50)
    if line:
        axes["tau"].axvline(fid["tau"], color="black", ls="dotted", label="Truth (input)")

    fig.set_size_inches(5,2.5)
    axes["tau"].plot(xplot, np.exp(asym_norm_logpdf(xplot, mu=fid['tau']))/max_asym_norm_logpdf, ls="--", color="black", label='Planck CMB')
    axes['tau'].set_xticks([0.04, 0.05,0.06,0.07, 0.08])
    axes['tau'].get_yaxis().set_visible(False)
    axes['tau'].set_xlabel(r"Optical depth $\tau$")
    axes['tau'].set_xlim(0.03, 0.09)
    handles, labels = axes['tau'].get_legend_handles_labels()
    axbox = axes['tau'].get_position()
    x_value=0.515; y_value=0.4
    fig.legend(handles, labels, loc=(axbox.x0 + x_value, axbox.y0 + y_value))
    plt.tight_layout()
    plt.savefig(filename+".pdf")
    plt.savefig(filename+".png", dpi=600)
    return fig

# Make color and grayscale versions
run_tau_histogram_plots(chain100, chain1k, bw=True, outfilename="line_seed0_"+version, line=True)
run_tau_histogram_plots(chain100, chain1k, bw=False, outfilename="line_seed0_"+version, line=True)

# Versions with other seeds:
#run_tau_histogram_plots(x1_pp_n3_vzend_eprior_etl_seed1['rwMergedChain_tau'],
#                  x1_pp_n6_vzend_eprior_etl_seed1['rwMergedChain_tau'],
#                  bw=False, outfilename="seed1_"+version)
#
#run_tau_histogram_plots(x1_pp_n3_vzend_eprior_etl_seed2['rwMergedChain_tau'],
#                  x1_pp_n6_vzend_eprior_etl_seed2['rwMergedChain_tau'],
#                  bw=False, outfilename="seed2_"+version)
#
#run_tau_histogram_plots(x1_pp_n3_vzend_eprior_etl_seed3['rwMergedChain_tau'],
#                  x1_pp_n6_vzend_eprior_etl_seed3['rwMergedChain_tau'],
#                  bw=False, outfilename="seed3_"+version)


print("=========== Figure 5 (now 3) ===========")

def print_rel_error(samples, label="Planck+?", key="logA"):
    bf = bestFit(samples, key=key)
    CL = fastCL(samples, key=key, level=0.68)
    diff = np.diff(CL)[0]
    print(label+": {0:.2f}%".format(diff/2*100/bf))
    print("  (bestfit) logA = {0:.3f}".format(bf)+"+{0:.3f}".format(CL[1]-bf)+"-{0:.3f}".format(bf-CL[0])+" (std =", std, ")")

def run_As_plots(planckmnu, planckmnunew_frb3, planckmnunew_frb6, bw=True, outfilename=None):
    cosmotriangle = ['tau', 'logA']
    filename = "paper_plots/Fig5_tauAsplot_"+outfilename
    if bw:
        gs = lambda r,g,b: [0.2989 * r + 0.5870 * g + 0.1140 * b]*3
        filename += "_bw"
    else:
        gs = lambda r,g,b: (r,g,b)
    fig, axes = planckmnunew_frb6.plot_2d(cosmotriangle, color=gs(*color2), types={'lower':'kde'}, label='+1,000 FRBs', ls='dotted')
    planckmnu.plot_2d(axes=axes, edgecolor='black', ls='dashed', label="Planck CMB", types={'lower':'kde'}, facecolor=None, lw=1)
    planckmnunew_frb3.plot_2d(axes=axes, edgecolor=gs(*color1), label='+100 FRBs', types={'lower':'kde'}, facecolor=None, lw=1.5)

    axes['tau']['logA'].set_ylabel(r"$\ln(10^{10}A_s)$")
    fig.set_size_inches(5,2.5)
    handles, labels = axes['tau']['logA'].get_legend_handles_labels()
    handles[1] = mpatches.Patch(edgecolor="black", ls="dashed", label='Planck CMB', facecolor="white")

    axbox = axes['tau']['logA'].get_position()
    x_value=0.525; y_value=0.15
    leg = fig.legend(np.array(handles)[[1,2,0]], np.array(labels)[[1,2,0]], loc=(axbox.x0 + x_value, axbox.y0 + y_value))
    plt.tight_layout()
    axes['tau']['logA'].set_xticks([0.04, 0.05,0.06,0.07, 0.08])
    axes['tau']['logA'].set_xlabel(r"Optical depth $\tau$")
    axes['tau']['logA'].set_xlim(0.035, 0.0815)
    plt.savefig(filename+".pdf")
    plt.savefig(filename+".png", dpi=600)
    return fig

def run_As_analysis(chain100, chain1k, outfilename=None):
    kde3 = sst.gaussian_kde(chain100.tau, weights=chain100.weights)
    kde6 = sst.gaussian_kde(chain1k.tau, weights=chain1k.weights)
    kde_interp = np.linspace(planckmnu.tau.min(), planckmnu.tau.max(), 1000)
    fit3 = sip.interp1d(kde_interp, kde3(kde_interp))
    fit6 = sip.interp1d(kde_interp, kde6(kde_interp))

    addll3 = np.log(fit3(planckmnu.tau))
    addll6 = np.log(fit6(planckmnu.tau))
    planckmnunew_frb3 = MCMCSamples(planckmnu).importance_sample(addll3, action='add')
    planckmnunew_frb6 = MCMCSamples(planckmnu).importance_sample(addll6, action='add')

    print("Relative error on primordial power spectrum amplitude ln(1e10 A_s):")

    print_rel_error(planckmnu, label="  Planck alone")
    print_rel_error(planckmnunew_frb3, label="  Planck + 100 FRBs")
    print_rel_error(planckmnunew_frb6, label="  Planck + 1000 FRBs")

    run_As_plots(planckmnu, planckmnunew_frb3, planckmnunew_frb6, bw=True, outfilename=outfilename)
    run_As_plots(planckmnu, planckmnunew_frb3, planckmnunew_frb6, bw=False, outfilename=outfilename)



run_As_analysis(chain100, chain1k, outfilename="seed0_"+version)

# For chains with other seeds:
#run_As_analysis(x1_pp_n3_vzend_eprior_etl_seed1['rwMergedChain_tau'],
#           x1_pp_n6_vzend_eprior_etl_seed1['rwMergedChain_tau'],
#           outfilename="seed1_"+version)
#run_As_analysis(x1_pp_n3_vzend_eprior_etl_seed2['rwMergedChain_tau'],
#           x1_pp_n6_vzend_eprior_etl_seed2['rwMergedChain_tau'],
#           outfilename="seed2_"+version)
#run_As_analysis(x1_pp_n3_vzend_eprior_etl_seed3['rwMergedChain_tau'],
#           x1_pp_n6_vzend_eprior_etl_seed3['rwMergedChain_tau'],
#           outfilename="seed3_"+version)


print("============= New Figure 2 =============")
def renameaxes(ax, newname='2'):
    newaxes={}
    for k1 in ax.keys():
        kn1 = k1[:-1]+newname
        newaxes[kn1] = {}
        for k2 in ax[k1].keys():
            kn2 = k2[:-1]+newname
            print(k1,k2,"->",kn1,kn2)
            newaxes[kn1][kn2] = ax[k1][k2]
    return pd.DataFrame(newaxes)

def paramplot(chains, n=3, axes=None, method="kde"):
    chain = deepcopy(chains[n])
    for i in range(n+1):
        chain.limits["z"+str(n+1)] = [5, 30]
    for i in range(n-1):
        chain.limits["x"+str(n+2)] = [0, 1]
    print("Keys:", chain.keys())
    zorder=10
    cd = [plt.cm.tab20b.colors[8], plt.cm.tab20b.colors[4], plt.cm.tab20b.colors[16]]
    if axes is None:
        fig, axes = chain.plot_2d(['z2','x2'], types={"lower": method}, alpha=0.7, label='Knot 2', color=cd[1])
        for i in range(n-1):
            axes = renameaxes(axes, newname=str(i+3))
            chain.plot_2d(axes, alpha=0.7, types={"lower": method}, label='Knot '+str(i+3), color=cd[(i+2)%len(cd)])
    else:
        fig = None
        for i in range(n):
            axes = renameaxes(axes, newname=str(i+2))
            chain.plot_2d(axes, alpha=0.7, types={"lower": method}, color=cd[(i+1)%len(cd)], facecolor=None, ls="--")
    print(chain.limits)
    k = axes.keys()[0]
    j = axes[k].keys()[0]
    planckstyle = {"fmt":'o', "markersize":4, "lw": 2, "capsize": 4}
    handles, labels = axes[k][j].get_legend_handles_labels()
    return fig, axes

illustration_n = 3
fig,axes = paramplot(x1_pp_n6_vzend_eprior_etl['chains'], n=illustration_n)
plt.ylabel("$x_i$ knot coordinates")
plt.xlabel("$z$ knot coordinates")
assert illustration_n == 3
illustration_keys = np.array(["z1","z2","z3","z4","z5","x2","x3","x4"])
illustration_points = []
illustration_widths = []
for k in illustration_keys:
    illustration_points.append(x1_pp_n6_vzend_eprior_etl['chains'][illustration_n][k].mean())
    illustration_widths.append(x1_pp_n6_vzend_eprior_etl['chains'][illustration_n][k].std())

illustration_dict = dict(zip(illustration_keys, illustration_points))
illustration_dict["x1"] = 1
illustration_dict["x5"] = 0
d = illustration_dict
illustration_points = np.array(illustration_points)
illustration_widths = np.array(illustration_widths)
zplot=np.linspace(4,31,1000)
colorline="black"
plt.plot(zplot, xifunc(zplot, illustration_points, illustration_keys), color=colorline, zorder=1, lw=2)
scatter_z = [illustration_dict["z"+str(i)] for i in [1,2,3,4,5]]
scatter_x = [1,*[illustration_dict["x"+str(i)] for i in [2,3,4]],0]
plt.scatter(scatter_z, scatter_x, color=colorline, zorder=2)
figsize = [5,3]
fig.set_size_inches(*figsize)
plt.xlim(4.5,30)
plt.xticks([5,10,15,20,25,30])
plt.yticks([0,0.5,1])
plt.ylim(-0.05,1.02)
xsize = figsize[0]/25
ysize = figsize[1]/1.1
colorarrow = [plt.cm.tab20b.colors[0], plt.cm.tab20b.colors[4], plt.cm.tab20b.colors[16], plt.cm.tab20b.colors[8], plt.cm.tab20b.colors[12]]
for i in [1,2,3,4,5]:
    arrowkwargs = {"color": colorarrow[i-1], "zorder":2}
    gap = 0.07
    wi = 0.015
    hl = 0.05
    le = 0.15
    plt.arrow(gap/xsize+illustration_dict["z"+str(i)], illustration_dict["x"+str(i)],    le/xsize, 0, width=wi/ysize, head_length=hl/xsize, **arrowkwargs)
    plt.arrow(-gap/xsize+illustration_dict["z"+str(i)], illustration_dict["x"+str(i)],   -le/xsize, 0, width=wi/ysize, head_length=hl/xsize, **arrowkwargs)
    if i==1:
        plt.text(gap/xsize+illustration_dict["z"+str(i)]+le/xsize, illustration_dict["x"+str(i)]-13*wi/ysize, r"$\mathbf{z_"+str(i)+"}$", **arrowkwargs)
    elif i==5:
        plt.text(gap/xsize+illustration_dict["z"+str(i)]+le/xsize, illustration_dict["x"+str(i)]+7*wi/ysize, r"$\mathbf{z_"+str(i)+"}$", **arrowkwargs)
    else:
        plt.text(gap/xsize+illustration_dict["z"+str(i)]+3*wi/xsize, illustration_dict["x"+str(i)]+9*wi/ysize, r"$\mathbf{[z_"+str(i)+", x_"+str(i)+"]}$", **arrowkwargs)
    if i > 1 and i < illustration_n+2:
        plt.arrow(illustration_dict["z"+str(i)], gap/ysize+illustration_dict["x"+str(i)],   0, le/ysize, width=wi/xsize, head_length=hl/ysize, **arrowkwargs)
        plt.arrow(illustration_dict["z"+str(i)], -gap/ysize+illustration_dict["x"+str(i)], 0, -le/ysize, width=wi/xsize, head_length=hl/ysize, **arrowkwargs)

fig.legend(bbox_to_anchor=[0.95, 0.95], loc="upper right")
plt.tight_layout()
plt.savefig("paper_plots/Fig2b_flexknot_illustration_posterior.pdf")
plt.savefig("paper_plots/Fig2b_flexknot_illustration_posterior.png", dpi=600)

illustration_n = 3
fig,axes = paramplot(priorSamples, n=illustration_n)
plt.ylabel("$x_i$ knot coordinates")
plt.xlabel("$z$ knot coordinates")
assert illustration_n == 3
illustration_keys = np.array(["z1","z2","z3","z4","z5","x2","x3","x4"])
illustration_points = []
illustration_widths = []
for k in illustration_keys:
    illustration_points.append(priorSamples[illustration_n][k].mean())
    illustration_widths.append(priorSamples[illustration_n][k].std())

illustration_dict = dict(zip(illustration_keys, illustration_points))
illustration_dict["x1"] = 1
illustration_dict["x5"] = 0
d = illustration_dict
illustration_points = np.array(illustration_points)
illustration_widths = np.array(illustration_widths)
zplot=np.linspace(4,31,1000)
colorline="black"
plt.plot(zplot, xifunc(zplot, illustration_points, illustration_keys), color=colorline, zorder=1, lw=2)
scatter_z = [illustration_dict["z"+str(i)] for i in [1,2,3,4,5]]
scatter_x = [1,*[illustration_dict["x"+str(i)] for i in [2,3,4]],0]
plt.scatter(scatter_z, scatter_x, color=colorline, zorder=2)

figsize = [5,3]
fig.set_size_inches(*figsize)
plt.xlim(4.5,30)
plt.xticks([5,10,15,20,25,30])
plt.yticks([0,0.5,1])
plt.ylim(-0.05,1.02)
xsize = figsize[0]/25
ysize = figsize[1]/1.1
colorarrow = [plt.cm.tab20b.colors[0], plt.cm.tab20b.colors[4], plt.cm.tab20b.colors[16], plt.cm.tab20b.colors[8], plt.cm.tab20b.colors[12]]
for i in [1,2,3,4,5]:
    arrowkwargs = {"color": colorarrow[i-1], "zorder":2}
    gap = 0.07
    wi = 0.015
    hl = 0.05
    le = 0.15
    plt.arrow(gap/xsize+illustration_dict["z"+str(i)], illustration_dict["x"+str(i)],    le/xsize, 0, width=wi/ysize, head_length=hl/xsize, **arrowkwargs)
    plt.arrow(-gap/xsize+illustration_dict["z"+str(i)], illustration_dict["x"+str(i)],   -le/xsize, 0, width=wi/ysize, head_length=hl/xsize, **arrowkwargs)
    if i==1:
        plt.text(gap/xsize+illustration_dict["z"+str(i)]+1.1*le/xsize, illustration_dict["x"+str(i)]-11*wi/ysize, r"$\mathbf{z_"+str(i)+"}$", **arrowkwargs)
    elif i==5:
        plt.text(gap/xsize+illustration_dict["z"+str(i)]+le/xsize, illustration_dict["x"+str(i)]+7*wi/ysize, r"$\mathbf{z_"+str(i)+"}$", **arrowkwargs)
    else:
        plt.text(gap/xsize+illustration_dict["z"+str(i)]+3*wi/xsize, illustration_dict["x"+str(i)]+9*wi/ysize, r"$\mathbf{[z_"+str(i)+", x_"+str(i)+"]}$", **arrowkwargs)
    if i > 1 and i < illustration_n+2:
        plt.arrow(illustration_dict["z"+str(i)], gap/ysize+illustration_dict["x"+str(i)],   0, le/ysize, width=wi/xsize, head_length=hl/ysize, **arrowkwargs)
        plt.arrow(illustration_dict["z"+str(i)], -gap/ysize+illustration_dict["x"+str(i)], 0, -le/ysize, width=wi/xsize, head_length=hl/ysize, **arrowkwargs)

fig.legend(bbox_to_anchor=[0.95, 0.95], loc="upper right")
plt.tight_layout()
plt.savefig("paper_plots/Fig2a_flexknot_illustration_prior.pdf")
plt.savefig("paper_plots/Fig2a_flexknot_illustration_prior.png", dpi=600)



print("=========== Additional checks ===========")
print("  A) Main triangle plot, for 100 and 1,000 FRBs")
def plot_main_triangle(planckmnu, planckmnunew_frb3, planckmnunew_frb6, outfilename=None):
    for c in [planckmnu, planckmnunew_frb3, planckmnunew_frb6]:
        c.limits["tau"] = [0.04, 0.09]
        c.limits["Omega_b_H0"] = [3.2, 3.4]
        c.limits["Omega_m"] = [0.29, 0.33]
    cosmotriangle = ['tau', 'Omega_b_H0', "Omega_m"]
    filename = "extra_plots/E_triangle"+outfilename
    fig, axes = planckmnunew_frb6.plot_2d(cosmotriangle, color=color2, types={'lower':'kde', 'diagonal': 'hist'}, label='1000 FRB', ls='dotted',  bins=50)
    fig.set_size_inches(8,6)
    planckmnunew_frb3.plot_2d(axes=axes, color=color1, label='100 FRB', types={'lower':'kde', 'diagonal': 'hist'}, facecolor=None, lw=1.5, bins=50, alpha=1, histtype="step")
    handles, labels = axes['tau']['Omega_m'].get_legend_handles_labels()
    handles[1] = mpatches.Patch(edgecolor="black", ls="dashed", label='Planck CMB', facecolor="white")
    plt.tight_layout()
    plt.savefig(filename+".pdf")
    plt.savefig(filename+".png", dpi=600)
    return fig

plot_main_triangle(planckmnu, chain100, chain1k, outfilename="seed0_"+version)


print('  B) Triangle plot, for Planck Omega priors, and larger "5xPlanck" priors to check their influence')
def plot_planck_check_triangles(planckmnu, x1_pp_f4, x1_5xpp_f4, outfilename=None):
    for c in [planckmnu, x1_pp_f4, x1_5xpp_f4]:
        c.limits["tau"] = [0.04, 0.14]
        c.limits["Omega_b_H0"] = [3, 3.6]
        c.limits["Omega_m"] = [0.26, 0.37]
    cosmotriangle = ['tau', 'Omega_b_H0', "Omega_m"]
    filename = "extra_plots/E_planckcheck_"+outfilename
    fig, axes = x1_5xpp_f4.plot_2d(cosmotriangle, types={'lower':'kde', 'diagonal': 'hist'}, label='5 times larger Planck Omega prior',  diagonal_kwargs={"bins": 50}, lower_kwargs={"level":[0.68, 0.95]})
    fig.set_size_inches(8,6)
    x1_pp_f4.plot_2d(axes=axes, types={'lower':'kde', 'diagonal': 'hist'}, label='Normal Planck Omega prior', diagonal_kwargs={"bins": 50, "histtype":"step"}, lower_kwargs={"level":[0.68, 0.95]})
    handles, labels = axes["tau"]["Omega_m"].get_legend_handles_labels()
    fig.legend(handles, labels)
    plt.savefig(filename+".pdf")
    plt.savefig(filename+".png", dpi=600)
    return fig, axes

if load_extra:
    plot_planck_check_triangles(planckmnu, x1_pp_f4, x1_5xpp_f4, outfilename="seed0_"+version)


print("  C) Check how many measurements exceed 68\% confidence")
if load_extra:
    for s in [x1_pp_n3_vzend_eprior_etl, x1_pp_n6_vzend_eprior_etl, x1_pp_n3_vzend_eprior_etl_seed2, x1_pp_n6_vzend_eprior_etl_seed2, x1_pp_n3_vzend_eprior_etl_seed3, x1_pp_n6_vzend_eprior_etl_seed3]:
        for key in ["tau", "z_at_xi0.1", "z_at_xi0.5", "z_at_xi0.9"]:
            CL3 = fastCL(s['rwMergedChain_'+key], key=key, level=0.68)
            if CL3[0] < fid[key] and CL3[1] > fid[key]:
                print("OK")
            else:
                print(">68:", CL3, fid[key])

print("  D) Evidence Z as function of number of knots")
if True:
    evidences = np.exp(x1_pp_n3_vzend_eprior_etl['logZs']['logZ'])#[0:8]
    evidences2 = np.exp(x1_pp_n6_vzend_eprior_etl['logZs']['logZ'])#[0:8]
    norm = np.sum(evidences); evidences = np.array(evidences)/norm
    norm2 = np.sum(evidences2); evidences2 = np.array(evidences2)/norm2
    plt.figure(figsize=(5,3))
    plt.scatter(np.arange(len(evidences))+2, evidences, label='100 FRBs', color="#7963ac")
    plt.scatter(np.arange(len(evidences))+2, evidences2, label='1,000 FRBs', color="#ff8000")
    plt.plot(np.arange(len(evidences))+2, evidences, alpha=1, color="#7963ac")
    plt.plot(np.arange(len(evidences))+2, evidences2, alpha=1, color="#ff8000")
    plt.ylabel("Relative evidence")
    plt.xlabel("Model complexity (number of knots)")
    plt.ylim(0, 0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("extra_plots/E_flexknot_evidences.pdf")
    plt.savefig("extra_plots/E_flexknot_evidences.png", dpi=600)

print("  E) Histogram of tau for tanh chains vs. true value")
if True:
    x1_pp_n3_vzend_eprior_etl_tanh_dz.hist("tau", range=[0.045,0.06], bins=20, label="tanh posterior")
    plt.xlabel(r"$\tau$")
    plt.axvline(fid['tau'], label="True (input) value", color="orange")
    plt.legend()
    plt.savefig("extra_plots/E_tanh_tau_histogram_100FRBs.png")
    x1_pp_n6_vzend_eprior_etl_tanh_dz.hist("tau", range=[0.045,0.06], bins=40, label="tanh posterior")
    plt.xlabel(r"$\tau$")
    plt.axvline(fid['tau'], label="True (input) value", color="orange")
    plt.legend()
    plt.savefig("extra_plots/E_tanh_tau_histogram_1kFRBs.png")

