import numpy as np
import os
from matplotlib.ticker import FuncFormatter
import math
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import awkward as ak
from tabulate import tabulate
from scipy import constants
from iminuit import Minuit
from iminuit import minimize
from matplotlib.patches import Rectangle
from cut_flow_functions import cut_flow,lxcosTheta,LxcosTheta,signal_eff_pur,events
from pathlib import Path
from sample_norms import N_expect
from argparse import Namespace
'''
This code analyses the (x,cos(Theta)) distribution of muons originating from semileptonic decay of the t quark.
Event selection cuts are applied to the data try to minimise background events e.g.from the semileptonic decay of B-mesons producing "fake leptons" to the semileptonic signal
'''
###
#Import data
###

#specify BSM modification
BSM_mod = "vr_ttZup_" # ""for no modification,i.e. SM coupling

#import ntuples and specify jet algo
path_tlepThad = "/ceph/skeilbach/FCCee_topEWK/wzp6_ee_SM_tt_tlepThad_noCKMmix_keepPolInfo_{}ecm365.pkl".format(BSM_mod)
path_thadTlep = "/ceph/skeilbach/FCCee_topEWK/wzp6_ee_SM_tt_thadTlep_noCKMmix_keepPolInfo_{}ecm365.pkl".format(BSM_mod)
path_thadThad = "/ceph/skeilbach/FCCee_topEWK/wzp6_ee_SM_tt_thadThad_noCKMmix_keepPolInfo_{}ecm365.pkl".format(BSM_mod)

df_lephad = pd.read_pickle(path_tlepThad)[:100000]
df_hadlep = pd.read_pickle(path_thadTlep)[:100000]
df_hadhad = pd.read_pickle(path_thadThad)[:100000]

#jet_algo = "ee_genkt04" #inclusive anti k_t jet algo with R=0.4
jet_algo = "kt_exactly6"

###
#Projected amount of tT events at FCC-ee, amount of produced MC events and branching ratios for each decay channel
###

N_exp_lephad = N_expect["wzp6_ee_SM_tt_tlepThad_noCKMmix_keepPolInfo_{}ecm365".format(BSM_mod)]
N_exp_hadlep = N_expect["wzp6_ee_SM_tt_thadTlep_noCKMmix_keepPolInfo_{}ecm365".format(BSM_mod)]
N_exp_hadhad = N_expect["wzp6_ee_SM_tt_thadThad_noCKMmix_keepPolInfo_{}ecm365".format(BSM_mod)]


N_lephad = events(df_lephad,1)
N_hadlep = events(df_hadlep,1)
N_hadhad = events(df_hadhad,0)

R_lephad = N_exp_lephad/N_lephad
R_hadlep = N_exp_hadlep/N_hadlep
R_hadhad = N_exp_hadhad/N_hadhad


###
#Apply cut flow
###

df_lephad,table_lephad = cut_flow(df_lephad,jet_algo,"semileptonic",R_lephad,23,13) #leptons originate from a W+ -> lepton_charge=1.0
df_hadlep,table_hadlep = cut_flow(df_hadlep,jet_algo,"semileptonic",R_hadlep,23,13)
df_hadhad,table_hadhad = cut_flow(df_hadhad,jet_algo,"allhadronic",R_hadhad,23,13) #lepton_charge irrelevant for allhadronic sample

cut_names = ["cut2","cut3","cut4_>"]
table_semileptonic,table_allhadronic = signal_eff_pur(cut_names,table_lephad,table_hadlep,table_hadhad,jet_algo)

#load xcosTheta values for leptons for semileptonic (=SL) and allhadronic (AH) ntuples 
x_lplus_lephad,x_lminus_lephad,Theta_lplus_lephad,Theta_lminus_lephad = lxcosTheta(df_lephad)
x_lplus_hadlep,x_lminus_hadlep,Theta_lplus_hadlep,Theta_lminus_hadlep = lxcosTheta(df_hadlep)
x_lplus_hadhad,x_lminus_hadhad,Theta_lplus_hadhad,Theta_lminus_hadhad = lxcosTheta(df_hadhad)


###
#Rescale SL and AH histograms to match statistics we would expect for the projected 10⁶ top events at FCC-ee
###

#for positively charged leptons and genLeptons 
lplus_hist_lephad, lplus_xedges, lplus_yedges = np.histogram2d(x_lplus_lephad,np.cos(Theta_lplus_lephad),weights=np.full_like(x_lplus_lephad, R_lephad), bins=(25,25))
lplus_hist_hadlep,_,_ = np.histogram2d(x_lplus_hadlep,np.cos(Theta_lplus_hadlep),weights=np.full_like(x_lplus_hadlep, R_hadlep), bins=(lplus_xedges,lplus_yedges))
lplus_hist_hadhad,_,_ = np.histogram2d(x_lplus_hadhad,np.cos(Theta_lplus_hadhad),weights=np.full_like(x_lplus_hadhad, R_hadhad), bins=(lplus_xedges,lplus_yedges))
lplus_hist = lplus_hist_lephad+lplus_hist_lephad+lplus_hist_hadhad

'''
Lplus_hist_lephad, Lplus_xedges, Lplus_yedges = np.histogram2d(x_Lplus_lephad,np.cos(Theta_Lplus_lephad), bins=(25,25))
Lplus_hist_hadlep,_,_ = np.histogram2d(x_Lplus_hadlep,np.cos(Theta_Lplus_hadlep), bins=(25,25))
Lplus_hist_hadhad,_,_ = np.histogram2d(x_Lplus_hadhad,np.cos(Theta_Lplus_hadhad), bins=(25,25))
Lplus_hist = R_SL*Lplus_hist_lephad+R_SL*Lplus_hist_hadlep+R_AH* Lplus_hist_hadhad
np.save(path/"Lplus_hist",Lplus_hist)
np.save(path/"Lplus_xedges",Lplus_xedges)
np.save(path/"Lplus_yedges",Lplus_yedges)
'''

#for negatively charged leptons and genLeptons 
lminus_hist_lephad, lminus_xedges, lminus_yedges = np.histogram2d(x_lminus_lephad,np.cos(Theta_lminus_lephad),weights=np.full_like(x_lminus_lephad, R_lephad), bins=(25,25))
lminus_hist_hadlep,_,_ = np.histogram2d(x_lminus_hadlep,np.cos(Theta_lminus_hadlep),weights=np.full_like(x_lminus_hadlep, R_hadlep), bins=(lminus_xedges,lminus_yedges))
lminus_hist_hadhad,_,_ = np.histogram2d(x_lminus_hadhad,np.cos(Theta_lminus_hadhad),weights=np.full_like(x_lminus_hadhad, R_hadhad), bins=(lminus_xedges,lminus_yedges))
lminus_hist = lminus_hist_lephad+lminus_hist_lephad+lminus_hist_hadhad

#save arrays
path = Path('/home/skeilbach/FCCee_topEWK/arrays/SM_{}'.format(BSM_mod))
path.mkdir(parents=True, exist_ok=True)

with open(path/'table_semileptonic.pkl', 'wb') as f:
    pickle.dump(table_semileptonic, f)
with open(path/'table_allhadronic.pkl', 'wb') as f:
    pickle.dump(table_allhadronic, f)

np.save(path/"lminus_hist_lephad",lminus_hist_lephad)
np.save(path/"lminus_hist_hadlep",lminus_hist_hadlep)
np.save(path/"lminus_hist_hadhad",lminus_hist_hadhad)

np.save(path/"lplus_hist_lephad",lplus_hist_lephad)
np.save(path/"lplus_hist_hadlep",lplus_hist_hadlep)
np.save(path/"lplus_hist_hadhad",lplus_hist_hadhad)


###
#Plotting the genLepton and lepton distributions side by side
###

'''
lplus_xpos, lplus_ypos = np.meshgrid(lplus_yedges[:-1]+lplus_yedges[1:], lplus_xedges[:-1]+lplus_xedges[1:]) #use this specific slicing to ensure the x and y position of the bars in the plot is almost in the middle of the chosen bins
lplus_xpos = lplus_xpos.flatten()/2.
lplus_ypos = lplus_ypos.flatten()/2.
lplus_zpos = np.zeros_like (lplus_xpos)

lplus_dx = lplus_xedges [1] - lplus_xedges [0]
lplus_dy = lplus_yedges [1] - lplus_yedges [0]
lplus_dz = lplus_hist.flatten()

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
#ax1.plot_wireframe(xpos,ypos,hist, rstride=3, cstride=3)
ax1.bar3d(lplus_xpos, lplus_ypos, lplus_zpos, lplus_dx, lplus_dy, lplus_dz, zsort='average')
#ax1.plot_surface(xpos, ypos,hist, rstride=1, cstride=1, color= "blue")
ax1.set_title(r"$(x,cos(\Theta))$ for $l^+$")
ax1.set_xlabel(r"$cos(\Theta)$")
ax1.set_ylabel(r"$x$")
ax1.set_xticks([-1,0,1])
ax1.set_yticks([0.2,0.4,0.6,0.8,1.])
ax1.set_zlabel("frequency")
ax1.view_init(25,-35)


#now plotting the distribution for genLeptons
Lplus_xpos, Lplus_ypos = np.meshgrid(Lplus_yedges[:-1]+Lplus_yedges[1:], Lplus_xedges[:-1]+Lplus_xedges[1:]) #use this specific slicing to ensure the x and y position of the bars in the plot is almost in the middle of the chosen bins
Lplus_xpos = Lplus_xpos.flatten()/2.
Lplus_ypos = Lplus_ypos.flatten()/2.
Lplus_zpos = np.zeros_like(Lplus_xpos)

Lplus_dx = Lplus_xedges [1] - Lplus_xedges [0]
Lplus_dy = Lplus_yedges [1] - Lplus_yedges [0]
Lplus_dz = Lplus_hist.flatten()

ax2 = fig.add_subplot(122, projection='3d')
ax2.bar3d(Lplus_xpos, Lplus_ypos, Lplus_zpos, Lplus_dx, Lplus_dy, Lplus_dz, zsort='average')
ax2.set_title(r"$(x,cos(\Theta)$ for $L^+$")
ax2.set_xlabel(r"$cos(\Theta)$")
ax2.set_ylabel(r"$x$")
ax2.set_xticks([-1,0,1])
ax2.set_yticks([0.2,0.4,0.6,0.8,1.])
ax2.set_zlabel("frequency")
ax2.view_init(25,-35)
plt.suptitle(r"$(x,\mathrm{cos}(\Theta))$ for positive reconstructed leptons l and genLeptons L")
plt.savefig("/home/skeilbach/FCCee_topEWK/figures/FCCee_xcosTheta_lplus_SM_{}.png".format(BSM_mod),dpi=300)
plt.close()
'''

#plot x and cosTheta projection for SL and AH for l⁻={e⁻,mu⁻}
x_lminus_counts_SL, x_bin_edges = lminus_hist_hadlep[:, 0]+lminus_hist_lephad[:,0],lminus_xedges
x_lminus_counts_AH = lminus_hist_hadhad[:,0]

cosTheta_lminus_counts_SL, cosTheta_bin_edges = lminus_hist_hadlep[0,:]+lminus_hist_lephad[0,:],lminus_yedges
cosTheta_lminus_counts_AH = lminus_hist_hadhad[0,:]

kit_green100=(0,.59,.51)
kit_green15 =(.85,.93,.93)

plt.stairs(x_lminus_counts_SL,x_bin_edges,hatch="///",color=kit_green100,fill=True,label="semileptonic signal")
plt.stairs(x_lminus_counts_AH,x_bin_edges,hatch="||",color=kit_green15,fill=True,label="allhadronic signal")
plt.xticks(np.arange(0,1.1,0.1))
plt.yscale("log")
plt.yticks([1,10,10**2,10**3,10**4],label=["1","10",r"$10^2$",r"$10^3$",r"$10^4$"])
plt.xlabel(r"$x$")
plt.ylabel(r"Number of events")
plt.legend()
plt.title(r"Reduced energy for $l\in\{e^-,\mu^-\}$ after cuts")
plt.savefig("/home/skeilbach/FCCee_topEWK/figures/ee_tt_SM_{}x_lminus_nod0.png".format(BSM_mod),dpi=300)
plt.close()

plt.stairs(cosTheta_lminus_counts_SL,cosTheta_bin_edges,hatch="///",color=kit_green100,fill=True,label="semileptonic signal")
plt.stairs(cosTheta_lminus_counts_AH,cosTheta_bin_edges,hatch="||",color=kit_green15,fill=True,label="allhadronic signal")
plt.xticks=(np.arange(-1,1.2,0.2))
plt.yscale("log")
plt.yticks([1,10,10**2,10**3,10**4],label=["1","10",r"$10^2$",r"$10^3$",r"$10^4$"])
plt.xlabel(r"$\cos\theta$")
plt.ylabel(r"Number of events")
plt.legend()
plt.title(r"Angular distribution for $l\in\{e^-,\mu^-\}$ after cuts")
plt.savefig("/home/skeilbach/FCCee_topEWK/figures/ee_tt_SM_{}Theta_lminus_nod0.png".format(BSM_mod),dpi=300)
plt.close()


###
#Chi square fit
###


#load histogrammed data from MC run with SM couplings
path_SM = Path('/home/skeilbach/FCCee_topEWK/arrays/SM_')
d_i = np.load(path_SM/"lminus_hist_hadlep.npy")+np.load(path_SM/"lminus_hist_lephad.npy")+np.load(path_SM/"lminus_hist_hadhad.npy")

#Define data
sigma_d_i = np.sqrt(d_i) #gaussian error on bins (central limit theorem applied to poisson dist)
gaussian_noise = np.random.normal(0,sigma_d_i.reshape(-1),sigma_d_i.size).reshape(sigma_d_i.shape)
n_SM = d_i #in this testing case d_i only contains SM value and is thus equal to n_SM
n_i = d_i + gaussian_noise #produce "experimental data" by adding gaussian noise to the binned data. Use the std of each bin, i.e. sqrt(n_bin) to produce take as std for the normal distributed sample that the noise is being calculated -> bins with high uncertainty are more likely to have higher noise
n_mod = np.load(path/"lminus_hist_hadlep.npy")+np.load(path/"lminus_hist_lephad.npy")+np.load(path/"lminus_hist_hadhad.npy")
n_i,n_SM,n_mod = n_i.flatten(),n_SM.flatten(),n_mod.flatten()
data = Namespace(n_i=n_i,n_SM=n_SM,n_mod=n_mod)

#Define chi2 function with the fit parameter k so that for k=0 the experimental data represents the SM couplings 
def chi2(k):
    return np.sum((n_i-(n_SM+k*(n_mod-n_SM)))**2/n_i)


#minimise chi2 using the iminuit package
m = Minuit(chi2,k=0)
m.migrad()

k_min = m.values["k"]
k_std = m.errors["k"]
power_min = math.floor(math.log10(abs(k_min)))
power_std = math.floor(math.log10(abs(k_std)))
#draw Delta chi2 profile
Delta_chi2,k = m.profile("k",subtract_min=True)
fig, ax = plt.subplots()
ax.plot(Delta_chi2,k,color="blue")
ax.grid()
ax.vlines(k_min,0,100,colors="red")
ax.add_patch(Rectangle((-k_std+k_min,0), 2*k_std, 100,facecolor='mediumseagreen',fill=True))
ax.set_ylim(0,3)
ax.set_xlim(-2*k_std+k_min,2*k_std+k_min)
ax.set_xlabel(r"$\delta$")
ax.set_ylabel(r"$\Delta \chi^2$")
plt.title(r"$\delta_{{\mathrm{{{}}}}}={:.2f}\cdot 10^{{{:+d}}}\pm{:.4f}\cdot 10^{{{:+d}}}$".format(BSM_mod[:-1].replace("_", r"\_"), k_min*math.pow(10,-power_min),power_min,k_std*math.pow(10,-power_std),power_std))
plt.savefig("/home/skeilbach/FCCee_topEWK/figures/Delta_chi2_SM_{}_nod0.png".format(BSM_mod),dpi=300)
plt.close()



