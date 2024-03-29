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
BSM_mod = [""] # ""for no modification,i.e. SM coupling

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
              "cut4": {"p_cut":13,"comparison":">"},
              "cut5": {"d0":0.05,"d0_signif": 50, "p_cut":13}
             }

#Define which ntuples are to be included
ntuples = ["tlepThad","thadTlep","thadThad"]
chunks=["chunk0","chunk1","chunk2","chunk3","chunk4"]

       
###
#Apply cut-flow to each decay channel and save the x and Theta values for later use
###

for BSM_coupling in BSM_mod:
    #Define dictionaries to save results for each coupling, i.e. respective cut-flow eff/pur (in table_dic) and the x,cosTheta results (in results_dic)
    table_dic = {}
    xThetaR_dic = {}
    path = Path('/home/skeilbach/FCCee_topEWK/arrays/SM_{}'.format(BSM_coupling))
    path.mkdir(parents=True, exist_ok=True)
    for channel in ntuples:
        print("======== {} ========".format(channel))
        if BSM_coupling=="":
            df_channel,N_exp,N_channel = df_load(channel,BSM_coupling)
            R_channel = N_exp/N_channel #scaling factor for channel
            #Apply cuts to the df
            df_channel,table_channel = cut_flow(df_channel,cut_dic,cut_limits,channel)
            #Calculate reduced energy x and polar angle Theta 
            x_lplus,x_lminus,Theta_lplus,Theta_lminus = lxcosTheta(df_channel)
        if BSM_coupling!="":
            table_channel = 0
            N_channel = 0 #this is the scaling factor for the whole ntuple - which is divided into 5 chunks -> R_channel = R_chunk0 + R_chunk1 +... = N_exp_channel * (N_channel_MC/N_tot_channel_MC) with N_tot_channel_MC = 5M
            x_lplus_channel,x_lminus_channel,Theta_lplus_channel,Theta_lminus_channel = [],[],[],[]
            for chunk in chunks:
                print("######## {} ########".format(chunk))
                df_chunk,N_exp,N_chunk = df_load(channel,BSM_coupling,chunk)
                N_channel += N_chunk
                #Apply cuts to the df
                df_chunk,table_chunk = cut_flow(df_chunk,cut_dic,cut_limits,channel)
                #Calculate reduced energy x and polar angle Theta 
                x_lplus_chunk,x_lminus_chunk,Theta_lplus_chunk,Theta_lminus_chunk = lxcosTheta(df_chunk)
                #save # events before and after each cut and x/Theta values for each chunk
                x_lplus_channel.append(x_lplus_chunk);x_lminus_channel.append(x_lminus_chunk)
                Theta_lplus_channel.append(Theta_lplus_chunk);Theta_lminus_channel.append(Theta_lminus_chunk)
                table_channel += table_chunk
            #Calculate total scaling factor: R_channel = N_exp*(1/(N_chunk1+...+N_chunk2))=N_exp/N_channel
            R_channel = N_exp/N_channel
            x_lplus,x_lminus,Theta_lplus,Theta_lminus = np.concatenate(x_lplus_channel),np.concatenate(x_lminus_channel),np.concatenate(Theta_lplus_channel),np.concatenate(Theta_lminus_channel)
        table_dic[channel] = R_channel*table_channel #scale number of events to match the expected amount of events for the 1.9M ttbar events at the FCC-ee
        xThetaR_dic[channel] = {"x_lplus":x_lplus,"x_lminus":x_lminus,"Theta_lplus":Theta_lplus,"Theta_lminus":Theta_lminus,"R":R_channel} 
    table_semileptonic,table_allhadronic = signal_eff_pur(cut_dic,jet_algo,**table_dic)
    #save the arrays
    with open(path/"xThetaR_dic.pkl","wb") as f:
        pickle.dump(xThetaR_dic,f)
    with open(path/'table_dic.pkl', 'wb') as f:
        pickle.dump(table_dic,f)

#Define function that calculates optimal bin edges to bin x and cosTheta for either positive (=plus) or negative (=minus) leptons based upon the maximum values for x and cosTheta for all couplings, all decay channels that are considered and the number of bins that are fed into the function as lists or integers
def calc_bin_edges(BSM_mod,decay_channel,lepton_charge,n_bins):
    x_max,x_min = [],[]
    cosTheta_max,cosTheta_min = [],[]
    for coupling in BSM_mod:
        x_max_tmp,x_min_tmp = [],[]
        cosTheta_max_tmp,cosTheta_min_tmp= [],[]
        path = Path('/home/skeilbach/FCCee_topEWK/arrays/SM_{}'.format(coupling))
        with open(path/"xThetaR_dic.pkl","rb") as f:
            xThetaR_dic = pickle.load(f)
        for channel in decay_channel:
            x_max_tmp.append(np.max(xThetaR_dic[channel]["x_l{}".format(lepton_charge)]))
            x_min_tmp.append(np.min(xThetaR_dic[channel]["x_l{}".format(lepton_charge)]))
            cosTheta_max_tmp.append(np.max(np.cos(xThetaR_dic[channel]["Theta_l{}".format(lepton_charge)])))
            cosTheta_min_tmp.append(np.min(np.cos(xThetaR_dic[channel]["Theta_l{}".format(lepton_charge)])))
        x_max.append(max(x_max_tmp))
        x_min.append(min(x_min_tmp))
        cosTheta_max.append(max(cosTheta_max_tmp))
        cosTheta_min.append(min(cosTheta_min_tmp))
    x_bin_edges = np.linspace(min(x_min),max(x_max),n_bins+1)
    cosTheta_bin_edges = np.linspace(min(cosTheta_min),max(cosTheta_max),n_bins+1)
    return x_bin_edges,cosTheta_bin_edges

#Define function that returns either binned x or cosTheta data for a specific coupling with coupling, lepton charge and bin edges as input while saving the bin counts for later use
def xcosTheta_hist(BSM_coupling,lepton_charge,decay_channels,x_bin_edges,cosTheta_bin_edges,projection):
    path = Path('/home/skeilbach/FCCee_topEWK/arrays/SM_{}'.format(coupling))
    tmp = {}
    with open(path/"xThetaR_dic.pkl","rb") as f:
        xThetaR_dic = pickle.load(f)
    for channel in decay_channels:
        x_channel = xThetaR_dic[channel]["x_l{}".format(lepton_charge)]
        Theta_channel = xThetaR_dic[channel]["x_l{}".format(lepton_charge)]
        R_channel = xThetaR_dic[channel]["R"]
        counts,_,_ = np.histogram2d(x_channel,np.cos(Theta_channel),weights=np.full_like(x_channel, R_channel,dtype=np.double), bins=[x_bin_edges,cosTheta_bin_edges])
        #save counts (important: as R_channel is dtype=float, counts contains non int values when scaled with R. Therefore a general,i.e. for all counts for all different couplings, conversion to int will be done when saving counts 
        np.save(path/"counts_l{}_{}".format(lepton_charge,channel),counts.astype(int))
        tmp[channel] = counts.astype(int)
    if projection=="x":
        return tmp["tlepThad"][:, 0]+tmp["thadTlep"][:,0],tmp["thadThad"][:,0]
    if projection=="cosTheta":
        return tmp["tlepThad"][0,:]+tmp["thadTlep"][0,:],tmp["thadThad"][0,:]


###
#Plot the x and cosTheta projections and Delta(BSM-SM) plots for negative and positive leptons
###


'''
lepton_charge = ["minus","plus"]

#Define custom colours
kit_green100=(0,.59,.51)
kit_green15=(.85,.93,.93)

for BSM_coupling in BSM_mod:
    for charge in lepton_charge:
        #Define proper bin edges
        x_bin_edges,cosTheta_bin_edges = calc_bin_edges(BSM_mod,ntuples,charge,25)
        #load SM data       
        x_counts_SL_SM,x_counts_AH_SM = xcosTheta_hist("",charge,ntuples,x_bin_edges,cosTheta_bin_edges,"x")
        cosTheta_counts_SL_SM,cosTheta_counts_AH_SM = xcosTheta_hist("",charge,ntuples,x_bin_edges,cosTheta_bin_edges,"cosTheta")
        #x projection + SM-BSM diff plots
        fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
        x_counts_SL, x_counts_AH = xcosTheta_hist(BSM_coupling,charge,ntuples,x_bin_edges,cosTheta_bin_edges,"x")
        ax1.stairs(x_counts_SL,x_bin_edges,hatch="///",color=kit_green100,fill=True,label="semileptonic signal")
        ax1.stairs(x_counts_AH,x_bin_edges,hatch="||",color=kit_green15,fill=True,label="full hadronic signal")
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
        ax2.stairs(x_counts_SL-x_counts_SL_SM,x_bin_edges,color="red",label="semileptonic signal")
        ax2.stairs(x_counts_AH-x_counts_AH_SM,x_bin_edges,color="blue",label="full hadronic signal")
        ax2.axhline(y=0,xmin=0,xmax=1,linestyle="--",color="grey")
        #plot errorbars
        x_bin_centers = (x_bin_edges[:-1] + x_bin_edges[1:]) / 2
        x_sigma_Delta_SL = np.sqrt(x_counts_SL+x_counts_SL_SM)
        x_sigma_Delta_AH = np.sqrt(x_counts_AH+x_counts_AH_SM)
        ax2.errorbar(x_bin_centers,x_counts_SL-x_counts_SL_SM,yerr=x_sigma_Delta_SL,fmt="none",color="black")
        ax2.scatter(x_bin_centers,x_counts_SL-x_counts_SL_SM,marker=".",color="red")
        ax2.errorbar(x_bin_centers,x_counts_AH-x_counts_AH_SM,yerr=x_sigma_Delta_AH,fmt="none",color="black")
        ax2.scatter(x_bin_centers,x_counts_AH-x_counts_AH_SM,marker=".",color="blue")
        ax2.set_xticks(np.arange(0,1.1,0.1))
        ax2.set_xlabel(r"$x$")
        ax2.set_ylabel(r"Number of events")
        ax2.legend()
        if BSM_coupling=="":
            ax2.set_title(r"$\Delta(\mathrm{{SM-SM}})$")
        if BSM_coupling!="":
            ax2.set_title(r"$\Delta(\mathrm{{BSM-SM}})$ for {} modification".format(BSM_coupling[:-1]))
        sign="+" if charge=="plus" else "-"
        plt.suptitle(r"Reduced energy for $l\in\{{e^{{{}}},\mu^{{{}}}\}}$ after cuts".format(sign,sign),fontweight='bold')
        plt.savefig("/home/skeilbach/FCCee_topEWK/figures/ee_tt_SM_{}x_l{}.png".format(BSM_coupling,charge),dpi=300)
        plt.close()
        #cosTheta projection
        fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
        cosTheta_counts_SL, cosTheta_counts_AH = xcosTheta_hist(BSM_coupling,charge,ntuples,x_bin_edges,cosTheta_bin_edges,"cosTheta")
        ax1.stairs(cosTheta_counts_SL,cosTheta_bin_edges,hatch="///",color=kit_green100,fill=True,label="semileptonic signal")
        ax1.stairs(cosTheta_counts_AH,cosTheta_bin_edges,hatch="||",color=kit_green15,fill=True,label="full hadronic signal")
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
        ax2.stairs(cosTheta_counts_SL-cosTheta_counts_SL_SM,cosTheta_bin_edges,color="red",label="semileptonic signal")
        ax2.stairs(cosTheta_counts_AH-cosTheta_counts_AH_SM,cosTheta_bin_edges,color="blue",label="full hadronic signal")
        #plot errorbars
        ax2.axhline(y=0,xmin=-1,xmax=1,linestyle="--",color="grey")
        cosTheta_bin_centers = (cosTheta_bin_edges[:-1] + cosTheta_bin_edges[1:]) / 2
        cosTheta_sigma_Delta_SL = np.sqrt(cosTheta_counts_SL+cosTheta_counts_SL_SM)
        cosTheta_sigma_Delta_AH = np.sqrt(cosTheta_counts_AH+cosTheta_counts_AH_SM)
        ax2.errorbar(cosTheta_bin_centers,cosTheta_counts_SL-cosTheta_counts_SL_SM,yerr=cosTheta_sigma_Delta_SL,fmt="none",color="black")
        ax2.scatter(cosTheta_bin_centers,cosTheta_counts_SL-cosTheta_counts_SL_SM,marker=".",color="red")
        ax2.errorbar(cosTheta_bin_centers,cosTheta_counts_AH-cosTheta_counts_AH_SM,yerr=cosTheta_sigma_Delta_AH,fmt="none",color="black")
        ax2.scatter(cosTheta_bin_centers,cosTheta_counts_AH-cosTheta_counts_AH_SM,marker=".",color="blue")
        ax2.set_xticks(np.arange(-1,1.2,0.2))
        ax2.set_xlabel(r"$\cos\theta$")
        ax2.set_ylabel(r"Number of events")
        ax2.legend()
        if BSM_coupling=="":
            ax2.set_title(r"$\Delta(\mathrm{{SM-SM}})$")
        if BSM_coupling!="":
            ax2.set_title(r"$\Delta(\mathrm{{BSM-SM}})$ for {} modification".format(BSM_coupling[:-1]))
        plt.suptitle(r"Angular distribution for $l\in\{{e^{{{}}},\mu^{{{}}}\}}$ after cuts".format(sign,sign),fontweight="bold")
        plt.savefig("/home/skeilbach/FCCee_topEWK/figures/ee_tt_SM_{}cosTheta_l{}.png".format(BSM_coupling,charge),dpi=300)
        plt.close()

###
#Chi square fit (only for BSM_coupling!="")
###
BSM_mod.remove("") #remove SM

for BSM_coupling in BSM_mod:
    for charge in lepton_charge:
        path = Path('/home/skeilbach/FCCee_topEWK/arrays/SM_{}'.format(BSM_coupling))
        #load histogrammed data from MC run with SM couplings
        path_SM = Path('/home/skeilbach/FCCee_topEWK/arrays/SM_')
        n_SM = np.load(path_SM/"counts_l{}_thadTlep.npy".format(charge))+np.load(path_SM/"counts_l{}_tlepThad.npy".format(charge))+np.load(path_SM/"counts_l{}_thadThad.npy".format(charge))
        #Define data
        n_i = n_SM #assume in this case that the "experimental" data does not include BSM physics
        n_mod = np.load(path/"counts_l{}_thadTlep.npy".format(charge))+np.load(path/"counts_l{}_tlepThad.npy".format(charge))+np.load(path/"counts_l{}_thadThad.npy".format(charge))
        n_i,n_SM,n_mod = n_i.flatten(),n_SM.flatten(),n_mod.flatten()
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
        #specify scientific notation for the x axis
        ax.ticklabel_format(axis="x",style="sci",scilimits=(0,0))
        #specify title format -> print 0 if k_min==0, otherwise k_min = x*10^y
        sign="+" if charge=="plus" else "-"
        plt.title(r"$\delta_{{\mathrm{{l^{{{}}},{}}}}}={:.2f}\pm{:.4f}\cdot 10^{{{:+d}}}$".format(sign,BSM_coupling[:-1].replace("_", r"\_"), k_min*math.pow(10,-power_min),k_std*math.pow(10,-power_std),power_std)) if k_min==0 else plt.title(r"$\delta_{{\mathrm{{{}}}}}={:.2f}\cdot 10^{{{:+d}}}\pm{:.4f}\cdot 10^{{{:+d}}}$".format(BSM_coupling[:-1].replace("_", r"\_"), k_min*math.pow(10,-power_min),power_min,k_std*math.pow(10,-power_std),power_std))
        plt.savefig("/home/skeilbach/FCCee_topEWK/figures/Delta_chi2_SM_{}_l{}.png".format(BSM_coupling[:-1],sign),dpi=300)
        plt.close()
'''

