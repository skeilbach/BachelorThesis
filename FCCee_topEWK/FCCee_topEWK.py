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
from cut_flow_functions import cut_flow,lxcosTheta,signal_eff_pur,events,df_load,cut1,cut2,cut3,cut4,cut5
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
BSM_mod = ["","ta_ttAdown_"] # ""for no modification,i.e. SM coupling

#Define jet algo
jet_algo = "kt_exactly6"

#Define cuts to be used in cut flow
cut_dic = {"cut1": cut1,
	"cut2": cut2,
	"cut3": cut3,
	"cut4": cut4,
	"cut5": cut5
       }

#Define cut limits
cut_limits = {"cut1": {"jet_algo":jet_algo},
	      "cut2": {"":""}, #cut2 doesnt require any additional cut limits
              "cut3": {"ME_cut":23},
              "cut4": {"p_cut":13,"comparison": ">"},
              "cut5": {"d0":0.1,"d0_signif": 50, "p_cut":13}
             }

#Define which ntuples are to be included
ntuples = ["tlepThad","thadTlep","thadThad"]

def data_loader(ntuples,filepath,projection):
    tmp = []
    for channel in ntuples:
        tmp.append(np.load(filepath/"counts_lminus_{}.npy".format(channel)))
    if projection=="x": 
        return tmp[0][:, 0]+tmp[1][:,0],tmp[2][:,0]
    if projection=="cosTheta":
        return tmp[0][0,:]+tmp[1][0,:],tmp[2][0,:]
        

for BSM_coupling in BSM_mod:
    #Define dictionaries to save results for each coupling, i.e. respective cut-flow eff/pur (in table_dic) and the x,cosTheta results (in results_dic)
    table_dic = {}
    results_dic = {}
    path = Path('/home/skeilbach/FCCee_topEWK/arrays/SM_{}'.format(BSM_coupling))
    path.mkdir(parents=True, exist_ok=True)
    for channel in ntuples:
        df_channel,R_channel = df_load(channel,BSM_coupling)
        #Apply cuts to the df
        df_channel,table_df = cut_flow(df_channel,cut_dic,cut_limits,channel,R_channel)
        #Calculate reduced energy x and polar angle Theta 
        x_lplus,x_lminus,Theta_lplus,Theta_lminus = lxcosTheta(df_channel)
        table_dic[channel] = table_df
        if channel=="tlepThad":
            counts_lplus,lplus_xedges,lplus_yedges = np.histogram2d(x_lplus,np.cos(Theta_lplus),weights=np.full_like(x_lplus, R_channel), bins=(25,25))
            counts_lminus,lminus_xedges,lminus_yedges = np.histogram2d(x_lminus,np.cos(Theta_lminus),weights=np.full_like(x_lminus, R_channel), bins=(25,25))
        else:
            counts_lplus,_,_ = np.histogram2d(x_lplus,np.cos(Theta_lplus),weights=np.full_like(x_lplus, R_channel), bins=(lplus_xedges,lplus_yedges))
            counts_lminus,_,_ = np.histogram2d(x_lminus,np.cos(Theta_lminus),weights=np.full_like(x_lminus, R_channel), bins=(lminus_xedges,lminus_yedges))
        results_dic[channel] = {"counts_lplus": counts_lplus,"counts_lminus":counts_lminus}
        #save arrays
        np.save(path/"counts_lplus_{}".format(channel),counts_lplus)
        np.save(path/"counts_lminus_{}".format(channel),counts_lminus) 
    table_semileptonic,table_allhadronic = signal_eff_pur(cut_dic,jet_algo,**table_dic)
    #save these arrays as well
    with open(path/'table_semileptonic.pkl', 'wb') as f:
        pickle.dump(table_semileptonic, f)
    with open(path/'table_allhadronic.pkl', 'wb') as f:
        pickle.dump(table_allhadronic, f)
    #plot x and cosTheta projection and (BSM-SM) diff plots for negativ leptons
    #Define custom colours
    kit_green100=(0,.59,.51)
    kit_green15=(.85,.93,.93)
    #load SM data 
    path_SM = Path("/home/skeilbach/FCCee_topEWK/arrays/SM_")
    x_lminus_counts_SL_SM,x_lminus_counts_AH_SM = data_loader(ntuples,path_SM,"x")
    cosTheta_lminus_counts_SL_SM,cosTheta_lminus_counts_AH_SM = data_loader(ntuples,path_SM,"cosTheta")
    #x projection + SM-BSM diff plots
    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
    x_lminus_counts_SL, x_lminus_counts_AH = data_loader(ntuples,path,"x")
    x_bin_edges = lminus_xedges
    ax1.stairs(x_lminus_counts_SL,x_bin_edges,hatch="///",color=kit_green100,fill=True,label="semileptonic signal")
    ax1.stairs(x_lminus_counts_AH,x_bin_edges,hatch="||",color=kit_green15,fill=True,label="full hadronic signal")
    ax1.set_xticks(np.arange(0,1.1,0.1))
    ax1.set_yscale("log")
    ax1.set_yticks([1,10,10**2,10**3,10**4])
    ax1.set_yticklabels(["1","10",r"$10^2$",r"$10^3$",r"$10^4$"])
    ax1.set_ylabel(r"Number of events")
    ax1.legend()
    if BSM_coupling=="":
        ax1.set_title(r"Reduced energy for SM coupling")
    if BSM_coupling!="":
        ax1.set_title(r"Reduced energy for modified ({}) coupling".format(BSM_coupling[:-1])) 
    ax2.stairs(x_lminus_counts_SL-x_lminus_counts_SL_SM,x_bin_edges,color="red",label="semileptonic signal")
    ax2.stairs(x_lminus_counts_AH-x_lminus_counts_AH_SM,x_bin_edges,color="blue",label="full hadronic signal")
    ax2.set_xticks(np.arange(0,1.1,0.1))
    ax2.set_xlabel(r"$x$")
    ax2.set_ylabel(r"Number of events")
    ax2.legend()
    if BSM_coupling=="":
        ax2.set_title(r"$\Delta(\mathrm{{SM-SM}})$")
    if BSM_coupling!="":
        ax2.set_title(r"$\Delta(\mathrm{{BSM-SM}})$ for {} modification".format(BSM_coupling[:-1]))
    plt.suptitle(r"Reduced energy for $l\in\{e^-,\mu^-\}$ after cuts",fontweight='bold')
    plt.savefig("/home/skeilbach/FCCee_topEWK/figures/ee_tt_SM_{}x_lminus.png".format(BSM_coupling),dpi=300)
    plt.close()
    #cosTheta projection
    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
    cosTheta_lminus_counts_SL, cosTheta_lminus_counts_AH = data_loader(ntuples,path,"cosTheta")
    cosTheta_bin_edges = lminus_yedges
    ax1.stairs(cosTheta_lminus_counts_SL,cosTheta_bin_edges,hatch="///",color=kit_green100,fill=True,label="semileptonic signal")
    ax1.stairs(cosTheta_lminus_counts_AH,cosTheta_bin_edges,hatch="||",color=kit_green15,fill=True,label="full hadronic signal")
    ax1.set_xticks=(np.arange(-1,1.2,0.2))
    ax1.set_yscale("log")
    ax1.set_yticks([1,10,10**2,10**3,10**4])
    ax1.set_yticklabels(["1","10",r"$10^2$",r"$10^3$",r"$10^4$"])
    ax1.set_ylabel(r"Number of events")
    ax1.legend()
    if BSM_coupling=="":
        ax1.set_title(r"Angular distribution for SM coupling")
    if BSM_coupling!="":
        ax1.set_title(r"Angular distribution for modified ({}) coupling".format(BSM_coupling[:-1]))
    ax2.stairs(cosTheta_lminus_counts_SL-cosTheta_lminus_counts_SL_SM,cosTheta_bin_edges,color="red",label="semileptonic signal")
    ax2.stairs(cosTheta_lminus_counts_AH-cosTheta_lminus_counts_AH_SM,cosTheta_bin_edges,color="blue",label="full hadronic signal")
    ax2.set_xticks(np.arange(-1,1.2,0.2))
    ax2.set_xlabel(r"$\cos\theta$")
    ax2.set_ylabel(r"Number of events")
    ax2.legend()
    if BSM_coupling=="":
        ax2.set_title(r"$\Delta(\mathrm{{SM-SM}})$")
    if BSM_coupling!="":
        ax2.set_title(r"$\Delta(\mathrm{{BSM-SM}})$ for {} modification".format(BSM_coupling[:-1]))
    plt.suptitle(r"Angular distribution for $l\in\{e^-,\mu^-\}$ after cuts",fontweight="bold")
    plt.savefig("/home/skeilbach/FCCee_topEWK/figures/ee_tt_SM_{}cosTheta_lminus.png".format(BSM_coupling),dpi=300)
    plt.close()



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


###
#Chi square fit
###

'''
#load histogrammed data from MC run with SM couplings
path_SM = Path('/home/skeilbach/FCCee_topEWK/arrays/SM_')
n_SM = np.load(path_SM/"lminus_hist_hadlep.npy")+np.load(path_SM/"lminus_hist_lephad.npy")+np.load(path_SM/"lminus_hist_hadhad.npy")

#Define data
n_i = n_SM #assume in this case that the "experimental" data does not include BSM physics
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
power_min = int("{:.2e}".format(k_min).split('e')[1])
power_std = int("{:.2e}".format(k_std).split('e')[1])
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
#specify title format -> print 0 if k_min==0, otherwise k_min = x*10^y
plt.title(r"$\delta_{{\mathrm{{{}}}}}={:.2f}\pm{:.4f}\cdot 10^{{{:+d}}}$".format(BSM_mod[:-1].replace("_", r"\_"), k_min*math.pow(10,-power_min),k_std*math.pow(10,-power_std),power_std)) if k_min==0 else plt.title(r"$\delta_{{\mathrm{{{}}}}}={:.2f}\cdot 10^{{{:+d}}}\pm{:.4f}\cdot 10^{{{:+d}}}$".format(BSM_mod[:-1].replace("_", r"\_"), k_min*math.pow(10,-power_min),power_min,k_std*math.pow(10,-power_std),power_std))
plt.savefig("/home/skeilbach/FCCee_topEWK/figures/Delta_chi2_SM_{}.png".format(BSM_mod),dpi=300)
plt.close()
'''


