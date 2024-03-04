import numpy as np
import os
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
from argparse import Namespace
'''
This code analyses the (x,cos(Theta)) distribution of muons originating from semileptonic decay of the t quark.
Event selection cuts are applied to the data try to minimise background events e.g.from the semileptonic decay of B-mesons producing "fake leptons" to the semileptonic signal
'''
###
#Import data
###

#import ntuples and specify jet algo
path_tlepThad = "/home/skeilbach/FCCee_topEWK/ee_SM_tt_tlepThad.pkl"
path_thadTlep = "/home/skeilbach/FCCee_topEWK/ee_SM_tt_thadTlep.pkl"
path_thadThad = "/home/skeilbach/FCCee_topEWK/ee_SM_tt_thadThad.pkl"

df_lephad = pd.read_pickle(path_tlepThad)
df_hadlep = pd.read_pickle(path_thadTlep)
df_hadhad = pd.read_pickle(path_thadThad)


jet_algo = "ee_genkt04" #inclusive anti k_t jet algo with R=0.4

###
#Projected amount of tT events at FCC-ee, amount of produced MC events and branching ratios for each decay channel
###

N_exp = 2*10**6
N_SL = events(df_lephad,1)+events(df_hadlep,1)
N_AH = events(df_hadhad,0)
BR_SL = 0.438
BR_AH = 0.457
R_SL = N_exp*BR_SL/N_SL*60 #scale by 60 because only 1/60 (100k <> 6 million) of all created MC was taken (for testing purposes)
R_AH = N_exp*BR_AH/N_AH*60


###
#Apply cut flow
###

df_lephad,table_lephad = cut_flow(df_lephad,jet_algo,"semileptonic")
df_hadlep,table_hadlep = cut_flow(df_hadlep,jet_algo,"semileptonic")
df_hadhad,table_hadhad = cut_flow(df_hadhad,jet_algo,"allhadronic")

cut_names = ["cut2","cut3","cut4","cut5"]
table_semileptonic,table_allhadronic = signal_eff_pur(cut_names,table_lephad,table_hadlep,table_hadhad,jet_algo,R_SL,R_AH)

#load xcosTheta values for leptons for semileptonic (=SL) and allhadronic (AH) ntuples 
x_lplus_lephad,x_lminus_lephad,Theta_lplus_lephad,Theta_lminus_lephad = lxcosTheta(df_lephad)
x_lplus_hadlep,x_lminus_hadlep,Theta_lplus_hadlep,Theta_lminus_hadlep = lxcosTheta(df_hadlep)
x_lplus_hadhad,x_lminus_hadhad,Theta_lplus_hadhad,Theta_lminus_hadhad = lxcosTheta(df_hadhad)

#save arrays
path = Path('/home/skeilbach/FCCee_topEWK/arrays/')
path.mkdir(parents=True, exist_ok=True)

np.save(path/"x_lplus_lephad",x_lplus_lephad)
np.save(path/"x_lplushadlep",x_lplus_hadlep)
np.save(path/"x_lplus_hadhad",x_lplus_hadhad)
np.save(path/"Theta_lplus_lephad",Theta_lplus_lephad)
np.save(path/"Theta_lplushadlep",Theta_lplus_hadlep)
np.save(path/"Theta_lplus_hadhad",Theta_lplus_hadhad)

np.save(path/"x_lminus_lephad",x_lminus_lephad)
np.save(path/"x_lminushadlep",x_lminus_hadlep)
np.save(path/"x_lminus_hadhad",x_lminus_hadhad)
np.save(path/"Theta_lminus_lephad",Theta_lminus_lephad)
np.save(path/"Theta_lminushadlep",Theta_lminus_hadlep)
np.save(path/"Theta_lminus_hadhad",Theta_lminus_hadhad)


###
#Rescale SL and AH histograms to match statistics we would expect for the projected 10⁶ top events at FCC-ee
###

#for positively charged leptons and genLeptons (plus scale by 60 because original ntuples contain 6 million samples whereas only 100k are used for testing purposes -> factor 60)
x_lplus_SL = np.concatenate((x_lplus_lephad,x_lplus_hadlep))
Theta_lplus_SL = np.concatenate((Theta_lplus_lephad,Theta_lplus_hadlep))
lplus_hist_SL, lplus_xedges, lplus_yedges = np.histogram2d(x_lplus_SL,np.cos(Theta_lplus_SL), bins=(25,25))
lplus_hist_AH,_,_ = np.histogram2d(x_lplus_hadhad,np.cos(Theta_lplus_hadhad), bins=(lplus_xedges,lplus_yedges))
lplus_hist = R_SL*lplus_hist_SL +R_AH* lplus_hist_AH

'''
Lplus_hist_lephad, Lplus_xedges, Lplus_yedges = np.histogram2d(x_Lplus_lephad,np.cos(Theta_Lplus_lephad), bins=(25,25))
Lplus_hist_hadlep,_,_ = np.histogram2d(x_Lplus_hadlep,np.cos(Theta_Lplus_hadlep), bins=(25,25))
Lplus_hist_hadhad,_,_ = np.histogram2d(x_Lplus_hadhad,np.cos(Theta_Lplus_hadhad), bins=(25,25))
Lplus_hist = R_SL*Lplus_hist_lephad+R_SL*Lplus_hist_hadlep+R_AH* Lplus_hist_hadhad
np.save(path/"Lplus_hist",Lplus_hist)
np.save(path/"Lplus_xedges",Lplus_xedges)
np.save(path/"Lplus_yedges",Lplus_yedges)
'''

#for negatively charged leptons and genLepton
x_lminus_SL = np.concatenate((x_lminus_lephad,x_lminus_hadlep))
Theta_lminus_SL = np.concatenate((Theta_lminus_lephad,Theta_lminus_hadlep))
lminus_hist_SL, lminus_xedges, lminus_yedges = np.histogram2d(x_lminus_SL,np.cos(Theta_lminus_SL), bins=(25,25))
lminus_hist_AH,_,_ = np.histogram2d(x_lminus_hadhad,np.cos(Theta_lminus_hadhad), bins=(lminus_xedges,lminus_yedges))
lminus_hist = R_SL*lminus_hist_SL +R_AH* lminus_hist_AH

###
#Plotting the genLepton and lepton distributions side by side
###

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

'''
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
'''
plt.savefig("/home/skeilbach/FCCee_topEWK/figures/FCCee_xcosTheta_plus.png",dpi=300)
plt.close()

#plot x and cosTheta projection for SL and AH for l⁻={e⁻,mu⁻}
x_lminus_counts_SL, x_bin_edges = np.histogram(x_lminus_SL,bins=25)
x_lminus_counts_AH,_ = np.histogram(x_lminus_hadhad,bins=x_bin_edges)

cosTheta_lminus_counts_SL, cosTheta_bin_edges = np.histogram(np.cos(Theta_lminus_SL), bins=25)
cosTheta_lminus_counts_AH,_ = np.histogram(np.cos(Theta_lminus_hadhad), bins = cosTheta_bin_edges)

kit_green100=(0,.59,.51)
kit_green15 =(.85,.93,.93)

plt.stairs(R_SL*x_lminus_counts_SL,x_bin_edges,hatch="///",color=kit_green100,fill=True,label="semileptonic signal")
plt.stairs(R_AH*x_lminus_counts_AH,x_bin_edges,hatch="||",color=kit_green15,fill=True,label="allhadronic signal")
plt.xticks(np.arange(0,1.1,0.1))
plt.yscale("log")
plt.yticks([1,10,10**2,10**3,10**4,10**5,10**6],label=["1","10",r"$10^2$",r"$10^3$",r"$10^4$",r"$10^5$",r"$10^6$"])
plt.xlabel(r"$x$")
plt.ylabel(r"Number of events")
plt.legend()
plt.title(r"Reduced energy for $l\in\{e^-,\mu^-\}$ after cuts")
plt.savefig("/home/skeilbach/FCCee_topEWK/figures/FCCee_x_lminus.png",dpi=300)
plt.close()

plt.stairs(R_SL*cosTheta_lminus_counts_SL,cosTheta_bin_edges,hatch="///",color=kit_green100,fill=True,label="semileptonic signal")
plt.stairs(R_AH*cosTheta_lminus_counts_AH,cosTheta_bin_edges,hatch="||",color=kit_green15,fill=True,label="allhadronic signal")
plt.xticks=(np.arange(-1,1.2,0.2))
plt.yscale("log")
plt.yticks([1,10,10**2,10**3,10**4,10**5,10**6],label=["1","10",r"$10^2$",r"$10^3$",r"$10^4$",r"$10^5$",r"$10^6$"])
plt.xlabel(r"$\cos\theta$")
plt.ylabel(r"Number of events")
plt.legend()
plt.title(r"Angular distribution for $l\in\{e^-,\mu^-\}$ after cuts")
plt.savefig("/home/skeilbach/FCCee_topEWK/figures/FCCee_Theta_lminus.png",dpi=300)

###
#Chi square fit
###

#load histogrammed data from MC run with SM couplings
d_i = lminus_hist

#Define data
sigma_d_i = np.sqrt(d_i) #gaussian error on bins (central limit theorem applied to poisson dist)
gaussian_noise = np.random.normal(0,sigma_d_i.reshape(-1),sigma_d_i.size).reshape(sigma_d_i.shape)
n_SM = d_i #in this testing case d_i only contains SM value and is thus equal to n_SM
n_i = d_i + gaussian_noise #produce "experimental data" by adding gaussian noise to the binned data. Use the std of each bin, i.e. sqrt(n_bin) to produce take as std for the normal distributed sample that the noise is being calculated -> bins with high uncertainty are more likely to have higher noise
n_MC = d_i #for now MC data is only generated with SM couplings. Later n_MC will include also BSM couplings
n_i,n_SM,n_MC = n_i.flatten(),n_SM.flatten(),n_MC.flatten()
data = Namespace(n_i=n_i,n_SM=n_SM,n_MC=n_MC)

#Define chi2 function with the fit parameter k so that for k=0 the experimental data represents the SM couplings 
def chi2(k):
    return np.sum((n_i-(n_SM+k*(n_MC-n_SM)))**2/n_i)


#minimise chi2 using the iminuit package
m = Minuit(chi2,k=0)
m.migrad()

k_min = m.values[0]
k_std = m.errors[0]

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
plt.title(r"$\delta$={:.4f}$\pm${:.4f}".format(k_min,k_std))
plt.savefig("/home/skeilbach/FCCee_topEWK/figures/Delta_chi2.png",dpi=300)
plt.close()



