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
import numpy
import pylab
from scipy.signal import convolve2d
from scipy.optimize import curve_fit 
from scipy.stats import spearmanr
'''
This code analyses the (x,cos(Theta)) distribution of muons originating from semileptonic decay of the t quark.
Event selection cuts are applied to the data try to minimise background events e.g.from the semileptonic decay of B-mesons producing "fake leptons" to the semileptonic signal
'''
###
#Import data
###

#specify BSM modifications
#top_EWK_scenario = ["","ta_ttAdown_","ta_ttAup_","tv_ttAdown_","tv_ttAup_","vr_ttZup_","vr_ttZdown_"] # ""for no modification,i.e. SM coupling
top_EWK_scenario = ["","ta_ttAdown_"]

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

'''       
###
#Apply cut-flow to each decay channel and save the x and Theta values for later use
###

for BSM_coupling in top_EWK_scenario:
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

###
#Determine optimal binning for 2D (x,cosTheta) distribution
###

#Calculate optimal binwidths which minimises the integrated mean squared error (IMSE), i.e. how well the histogram resembles the true underlying distribution. It was shown by Scott (Multivariate Density Estimation, 2015) that for normal distributed values of a sample of dimension d a useful formala to calculate binwidth h_k of data k is h_k~3.5*sigma_k*n^(-1/(2+d)) with the error sigma_k of the subsample k of size n. In case the two datasets are correlated h_k has to be modified to h_i=3.504*sigma_i*(1-rho^2)*n^(-1/4).
m_t = 173.34
s = 365**2
beta = np.sqrt(1-4*m_t**2/s)
x_lim = 2*120/m_t*np.sqrt((1-beta)/(1+beta)) #Define artifical upper limit for all x_values

#Fit gaussian to (very likely normal distributed) energy distribution
def gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

path_SM = Path('/home/skeilbach/FCCee_topEWK/arrays/SM_')
with open(path_SM/"xThetaR_dic.pkl","rb") as f:
    xThetaR_dic = pickle.load(f)

x,Theta,weights = [],[],[]
N_x,N_cosTheta = 0,0 #expected number of events -> N_x = R_scaling*N_x,MC

for channel in ntuples:
    x_channel = xThetaR_dic[channel]["x_l{}".format(lepton_charge)]
    Theta_channel = xThetaR_dic[channel]["Theta_l{}".format(lepton_charge)]
    R_channel = xThetaR_dic[channel]["R"]
    N_x += int(R_channel*len(x_channel));N_cosTheta += int(R_channel*len(Theta_channel))
    x.append(x_channel)
    Theta.append(Theta_channel)
    weights.append(np.ones(len(x_channel))*R_channel)

x = np.concatenate(x);Theta = np.concatenate(Theta);weights = np.concatenate(weights)
#Dismiss outliers, i.e. entries with x>1
Theta = Theta[x<=x_lim]
weights = weights[x<=x_lim]
x=x[x<=x_lim]

#histogram reduced energy
counts_x,xedges = np.histogram(x,bins="fd",density=True)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.stairs(counts_x,xedges,color=kit_green100,label="MC data")
xpos = (xedges[:-1] + xedges[1:]) / 2

popt, pcov = curve_fit(gauss,xpos,counts_x)
gauss_fit = gauss(xpos,popt[0],popt[1],popt[2])
ax.plot(xpos,gauss_fit,c="r",label="Gaussian fit")
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
# place a text box in upper left in axes coords
textstr = '\n'.join((
    r'$a=%.3f$' % (popt[0], ),
    r'$\mu=%.3f$' % (popt[1], ),
    r'$\sigma=%.3f$' % (popt[2], )))
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
ax.legend()
ax.set_xlabel(r"x")
ax.set_ylabel(r"Number of events")
ax.set_title(r"Gaussian fit to reduced energy projection of MC_data for SM scenario")
fig.savefig("/home/skeilbach/FCCee_topEWK/figures/x_gaussian_fit.png",dpi=300)
plt.close()

#Use gained std of almost normal distributed x projection to calculate optimal binwidth
rho,_ = spearmanr(x,np.cos(Theta)) #calculate correlation between x and cosTheta data
h_k = 3.504*popt[2]*np.sqrt(len(x)/N_x)*(1-rho**2)**(3/8)*(N_x+N_cosTheta)**(-1/4) #optimal binwidth for correlated data#Note: h_k is ~10.5 so at least 11 bins for x

#cosTheta is far from normal distributed -> choose arbitrary n_bins=20
#now optimise n_bins for x axis by demanding that the minimum bin entry must not be <20 
yedges = np.histogram_bin_edges(np.cos(Theta),bins=20)
b=100
n_bins=24
xedges=(np.linspace(np.min(x)**(1/b), np.max(x)**(1/b),n_bins+1))**b #has more densely spaced bins for small x, i.e. where there is more data
counts,_,_ = np.histogram2d(x,np.cos(Theta),weights=weights,bins=(xedges,yedges))
counts=counts.astype(int)
while(np.any(counts.astype(int)<20)):
    n_bins -= 1
    xedges=(np.linspace(np.min(x)**(1/b), np.max(x)**(1/b),n_bins))**b
    counts,_,_ = np.histogram2d(x,np.cos(Theta),weights=weights,bins=(xedges,yedges))
    counts=counts.astype(int)

print(n_bins)
'''

###
#Optimal Binning results
###

n_bins_x = 14
n_bins_cosTheta = 20

#Define function that uses maximum minimum value of cosTheta and minimum maximum value cosTheta to ensure that the binning is later compatible with all BSM/SM scenarios.  
def calc_bin_edges(top_EWK_scenario,ntuples,lepton_charge,n_bins_x,n_bins_cosTheta):
    m_t = 173.34
    s = 365**2
    beta = np.sqrt(1-4*m_t**2/s)
    x_lim = 2*120/m_t*np.sqrt((1-beta)/(1+beta)) #w.l.o.g an upper limit on x is choosen so that outliers with x>1 do not led to empty bins when plotting the histograms (this is especially important for the Delta Chi2 fits where n_i = n_SM mustnt be 0 as we divide by n_i)
    x_min = []
    cosTheta_max,cosTheta_min = [],[]
    for coupling in top_EWK_scenario:
        x_min_tmp = []
        cosTheta_max_tmp,cosTheta_min_tmp= [],[]
        path = Path('/home/skeilbach/FCCee_topEWK/arrays/SM_{}'.format(coupling))
        with open(path/"xThetaR_dic.pkl","rb") as f:
            xThetaR_dic = pickle.load(f)
        for channel in ntuples:
            x_min_tmp.append(np.min(xThetaR_dic[channel]["x_l{}".format(lepton_charge)]))
            cosTheta_max_tmp.append(np.max(np.cos(xThetaR_dic[channel]["Theta_l{}".format(lepton_charge)])))
            cosTheta_min_tmp.append(np.min(np.cos(xThetaR_dic[channel]["Theta_l{}".format(lepton_charge)])))
        x_min.append(min(x_min_tmp))
        cosTheta_max.append(max(cosTheta_max_tmp))
        cosTheta_min.append(min(cosTheta_min_tmp))
    b=100 #Use same b value as was used when calculating optimal binwidth
    xedges=(np.linspace(max(x_min)**(1/b), x_lim**(1/b),n_bins_x+1))**b
    yedges = np.linspace(max(cosTheta_min),min(cosTheta_max),n_bins_cosTheta+1)
    return xedges,yedges


#Define function that returns either binned x or cosTheta data for a specific coupling with coupling, lepton charge and bin edges
def xcosTheta_hist(coupling,lepton_charge,decay_channel,x_bin_edges,cosTheta_bin_edges,projection=None):
    path = Path('/home/skeilbach/FCCee_topEWK/arrays/SM_{}'.format(coupling))
    output = 0
    with open(path/"xThetaR_dic.pkl","rb") as f:
        xThetaR_dic = pickle.load(f)
    for channel in decay_channel:
        x_channel = xThetaR_dic[channel]["x_l{}".format(lepton_charge)]
        Theta_channel = xThetaR_dic[channel]["Theta_l{}".format(lepton_charge)]
        R_channel = xThetaR_dic[channel]["R"]
        counts_channel,_,_ = np.histogram2d(x_channel,np.cos(Theta_channel),weights=np.full_like(x_channel, R_channel,dtype=np.double), bins=(x_bin_edges,cosTheta_bin_edges))
       #(important: as R_channel is dtype=float, counts contains non int values when scaled with R. Therefore a general,i.e. for all counts for all different couplings, conversion to int will be done 
        if projection==None:
            output += counts_channel.astype(int)
        elif projection=="x":
            output += np.sum(counts_channel.astype(int),axis=1)   
        elif projection=="cosTheta":
            output += np.sum(counts_channel.astype(int),axis=0)
    return output

'''
#Define custom std deviation
def std_scaled(coupling,ntuples,lepton_charge):
    path = Path('/home/skeilbach/FCCee_topEWK/arrays/SM_{}'.format(coupling))
    N_x,N_cosTheta = 0,0
    x,Theta = [],[]
    with open(path/"xThetaR_dic.pkl","rb") as f:
        xThetaR_dic = pickle.load(f)
    for channel in ntuples:
        x_channel = xThetaR_dic[channel]["x_l{}".format(lepton_charge)]
        Theta_channel = xThetaR_dic[channel]["Theta_l{}".format(lepton_charge)]
        R_channel = xThetaR_dic[channel]["R"]
        N_x += int(R_channel*len(x_channel));N_cosTheta += int(R_channel*len(Theta_channel))
        x.append(x_channel)
        Theta.append(Theta_channel)
    x = np.concatenate(x);Theta = np.concatenate(Theta)
    #compute rescaled std
    std_x = np.sqrt(1/(N_x-1)*np.sum((x-np.mean(x))**2))
    std_cosTheta = np.sqrt(1/(N_cosTheta-1)*np.sum((np.cos(Theta)-np.mean(np.cos(Theta)))**2))
    return std_x,std_cosTheta
'''

###
#Plot the x and cosTheta projections and Delta(BSM-SM) plots for negative and positive leptons
###



lepton_charge = ["minus","plus"]

#Define custom colours
kit_green100=(0,.59,.51)
kit_green15=(.85,.93,.93)


for BSM_coupling in top_EWK_scenario:
    for charge in lepton_charge:
        #Define proper bin edges
        x_bin_edges,cosTheta_bin_edges = calc_bin_edges(top_EWK_scenario,ntuples,charge,n_bins_x,n_bins_cosTheta)
        #load SM data       
        x_counts_SL_SM = xcosTheta_hist("",charge,["tlepThad","thadTlep"],x_bin_edges,cosTheta_bin_edges,"x")
        x_counts_AH_SM = xcosTheta_hist("",charge,["thadThad"],x_bin_edges,cosTheta_bin_edges,"x")
        cosTheta_counts_SL_SM = xcosTheta_hist("",charge,["tlepThad","thadTlep"],x_bin_edges,cosTheta_bin_edges,"cosTheta")
        cosTheta_counts_AH_SM = xcosTheta_hist("",charge,["thadThad"],x_bin_edges,cosTheta_bin_edges,"cosTheta")
        #x projection + SM-BSM diff plots
        fig = plt.figure()
        ax1 = plt.subplot2grid((3,1),(0,0),rowspan=2)
        ax2 = plt.subplot2grid((3,1),(2,0),rowspan=1,sharex=ax1)
        x_counts_SL = xcosTheta_hist(BSM_coupling,charge,["tlepThad","thadTlep"],x_bin_edges,cosTheta_bin_edges,"x")
        x_counts_AH = xcosTheta_hist(BSM_coupling,charge,["thadThad"],x_bin_edges,cosTheta_bin_edges,"x")
        ax1.stairs(x_counts_SL,x_bin_edges,hatch="///",color=kit_green100,fill=True,label="semileptonic signal")
        ax1.stairs(x_counts_AH,x_bin_edges,hatch="||",color=kit_green15,fill=True,label="full hadronic signal")
        ax1.set_yscale("log")
        ax1.set_yticks([1,10,10**2,10**3,10**4])
        ax1.set_yticklabels(["1","10",r"$10^2$",r"$10^3$",r"$10^4$"])
        ax1.set_ylabel(r"Number of events")
        ax1.legend()
        sign="+" if charge=="plus" else "-"
        if BSM_coupling=="":
            ax1.set_title(r"Reduced energy for $l\in\{{e^{{{}}},\mu^{{{}}}\}}$ and SM couplings".format(sign,sign),size="large") 
        if BSM_coupling!="":
            ax1.set_title(r"Reduced energy for $l\in\{{e^{{{}}},\mu^{{{}}}\}}$ and modified {} coupling".format(sign,sign,BSM_coupling[:-1]),size="large")  
        ax2.stairs(x_counts_SL-x_counts_SL_SM,x_bin_edges,color="red",label="semileptonic signal")
        ax2.stairs(x_counts_AH-x_counts_AH_SM,x_bin_edges,color="blue",label="full hadronic signal")
        ax2.axhline(0.,linestyle="--",color="grey")
        #plot errorbars
        x_bin_centers = (x_bin_edges[:-1] + x_bin_edges[1:]) / 2
        x_sigma_Delta_SL = np.sqrt(x_counts_SL+x_counts_SL_SM)
        x_sigma_Delta_AH = np.sqrt(x_counts_AH+x_counts_AH_SM)
        ax2.errorbar(x_bin_centers,x_counts_SL-x_counts_SL_SM,yerr=x_sigma_Delta_SL,fmt="r.")
        ax2.errorbar(x_bin_centers,x_counts_AH-x_counts_AH_SM,yerr=x_sigma_Delta_AH,fmt="b.")
        ax2.set_xticks(np.arange(0,1.1,0.1))
        ax2.set_xlabel(r"$x$")
        ax2.legend()
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=True)
        if BSM_coupling=="":
            ax2.set_ylabel(r"$\Delta(\mathrm{{SM-SM}})$")
        if BSM_coupling!="":
            ax2.set_ylabel(r"$\Delta(\mathrm{{mod-SM}})$")
        plt.savefig("/home/skeilbach/FCCee_topEWK/figures/ee_tt_SM_{}x_l{}.png".format(BSM_coupling,charge),dpi=300)
        plt.close()
        #cosTheta projection
        fig = plt.figure()
        ax1 = plt.subplot2grid((3,1),(0,0),rowspan=2)
        ax2 = plt.subplot2grid((3,1),(2,0),rowspan=1,sharex=ax1)
        cosTheta_counts_SL = xcosTheta_hist(BSM_coupling,charge,["tlepThad","thadTlep"],x_bin_edges,cosTheta_bin_edges,"cosTheta")
        cosTheta_counts_AH = xcosTheta_hist(BSM_coupling,charge,["thadThad"],x_bin_edges,cosTheta_bin_edges,"cosTheta")
        ax1.stairs(cosTheta_counts_SL,cosTheta_bin_edges,hatch="///",color=kit_green100,fill=True,label="semileptonic signal")
        ax1.stairs(cosTheta_counts_AH,cosTheta_bin_edges,hatch="||",color=kit_green15,fill=True,label="full hadronic signal")
        ax1.set_xticks=(np.arange(-1,1.2,0.2))
        ax1.set_yscale("log")
        ax1.set_yticks([1,10,10**2,10**3,10**4])
        ax1.set_yticklabels(["1","10",r"$10^2$",r"$10^3$",r"$10^4$"])
        ax1.set_ylabel(r"Number of events")
        ax1.legend()
        if BSM_coupling=="":
            ax1.set_title(r"Reduced energy for $l\in\{{e^{{{}}},\mu^{{{}}}\}}$ and SM couplings".format(sign,sign),size="large")
        if BSM_coupling!="":
            ax1.set_title(r"Reduced energy for $l\in\{{e^{{{}}},\mu^{{{}}}\}}$ and modified {} coupling".format(sign,sign,BSM_coupling[:-1]),size="large")
        ax2.stairs(cosTheta_counts_SL-cosTheta_counts_SL_SM,cosTheta_bin_edges,color="red",label="semileptonic signal")
        ax2.stairs(cosTheta_counts_AH-cosTheta_counts_AH_SM,cosTheta_bin_edges,color="blue",label="full hadronic signal")
        #plot errorbars
        ax2.axhline(0,linestyle="--",color="grey")
        cosTheta_bin_centers = (cosTheta_bin_edges[:-1] + cosTheta_bin_edges[1:]) / 2
        cosTheta_sigma_Delta_SL = np.sqrt(cosTheta_counts_SL+cosTheta_counts_SL_SM)
        cosTheta_sigma_Delta_AH = np.sqrt(cosTheta_counts_AH+cosTheta_counts_AH_SM)
        ax2.errorbar(cosTheta_bin_centers,cosTheta_counts_SL-cosTheta_counts_SL_SM,yerr=cosTheta_sigma_Delta_SL,fmt="r.")
        ax2.errorbar(cosTheta_bin_centers,cosTheta_counts_AH-cosTheta_counts_AH_SM,yerr=cosTheta_sigma_Delta_AH,fmt="b.")
        ax2.set_xticks(np.arange(-1,1.2,0.2))
        ax2.set_xlabel(r"$\cos\theta$")
        ax2.set_ylabel(r"Number of events")
        ax2.legend()
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=True)
        if BSM_coupling=="":
            ax2.set_ylabel(r"$\Delta(\mathrm{{SM-SM}})$")
        if BSM_coupling!="":
            ax2.set_ylabel(r"$\Delta(\mathrm{{mod-SM}})$")
        plt.savefig("/home/skeilbach/FCCee_topEWK/figures/ee_tt_SM_{}cosTheta_l{}.png".format(BSM_coupling,charge),dpi=300)
        plt.close()

###
#Chi square fit (only for BSM_coupling!="")
###
top_EWK_scenario_fit = top_EWK_scenario[1:]

for BSM_coupling in top_EWK_scenario_fit:
    for charge in lepton_charge:
        #histogram data
        x_bin_edges,cosTheta_bin_edges = calc_bin_edges(top_EWK_scenario,ntuples,charge,n_bins_x,n_bins_cosTheta) #choose binning
        n_SM = xcosTheta_hist("",charge,ntuples,x_bin_edges,cosTheta_bin_edges)
        n_i = n_SM #assume in this case that the "experimental" data does not include BSM physics
        n_mod = xcosTheta_hist(BSM_coupling,charge,ntuples,x_bin_edges,cosTheta_bin_edges)
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
        plt.savefig("/home/skeilbach/FCCee_topEWK/figures/Delta_Chi2_SM_{}_l{}.png".format(BSM_coupling[:-1],charge),dpi=300)
        plt.close()

###
#Plot (x,cosTheta) distribution for BSM = MOD - SM in the semileptonic channel (for positive and negative leptons) to compare with Patrick's plots for l^-
###
#BSM_scenario = ["","ta_ttAdown_","ta_ttAup_","tv_ttAdown_","tv_ttAup_","vr_ttZup_","vr_ttZdown_"]

def moving_average_2d(data, window):
    """Moving average on two-dimensional data.
    """
    # Makes sure that the window function is normalized.
    window /= window.sum()
    # Makes sure data array is a numpy array or masked array.
    if type(data).__name__ not in ['ndarray', 'MaskedArray']:
        data = numpy.asarray(data)
    # The output array has the same dimensions as the input data 
    # (mode='same') and symmetrical boundary conditions are assumed
    # (boundary='symm').
    return convolve2d(data, window, mode='same', boundary='symm')

win = numpy.ones((6,20))
for BSM_coupling in top_EWK_scenario:
    for charge in lepton_charge:
        sign="+" if charge=="plus" else "-"
        #calculate 2d (x,cosTheta) counts
        xedges,yedges = calc_bin_edges(top_EWK_scenario,["tlepThad","thadTlep"],charge,25,25)
        counts_SM_SL = xcosTheta_hist("",charge,["tlepThad","thadTlep"],xedges,yedges)
        counts_mod_SL = xcosTheta_hist(BSM_coupling,charge,["tlepThad","thadTlep"],xedges,yedges)
        #specify xpos,ypos and zpos to plot the distribution (x is equivalent to reduced energy axis and y to cosTheta axis)
        xpos, ypos = np.meshgrid(yedges[:-1]+yedges[1:], xedges[:-1]+xedges[1:]) #use this specific slicing to ensure the x and y position of the bars in the plot is almost in the middle of the chosen bins
        xpos /= 2.
        ypos /= 2.
        zpos = np.zeros_like (xpos)
        dx = xedges [1] - xedges [0]
        dy = yedges [1] - yedges [0]
        dz = (counts_mod_SL-counts_SM_SL).flatten() if BSM_coupling!="" else counts_SM_SL.flatten()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if BSM_coupling=="":
            ax.plot_surface(xpos, ypos,moving_average_2d(counts_SM_SL,win),color= "blue")
            #ax.plot_wireframe(xpos,ypos,moving_average_2d(counts_SM_SL,win),rstride=1,cstride=1)
            ax.set_title(r"$(x,\cos\theta)_{{\mathrm{{SM}}}}$ distribution for $l^{{{}}}$".format(sign))
        elif BSM_coupling!="":
            ax.plot_surface(xpos, ypos,moving_average_2d(counts_mod_SL-counts_SM_SL,win), color= "blue")
            #ax.plot_wireframe(xpos,ypos,moving_average_2d(counts_mod_SL-counts_SM_SL,win),rstride=1,cstride=1)
            ax.set_title(r"$(x,\cos\theta)_{{\mathrm{{{}}}}}$ distribution for $l^{{{}}}$".format(BSM_coupling[:-1].replace("_", r"\_"),sign))
        ax.set_xlabel(r"$\cos\theta$")
        ax.set_ylabel(r"$x$")
        ax.invert_xaxis()
        ax.set_xticks([-1,0,1])
        ax.set_yticks([0.2,0.4,0.6,0.8,1.])
        ax.set_zlabel("frequency")
        ax.set_zlim3d([np.min(dz),np.max(dz)])
        ax.view_init(25,-35)
        plt.savefig("/home/skeilbach/FCCee_topEWK/figures/xcosTheta_SM_{}l{}.png".format(BSM_coupling,charge),dpi=300)
        plt.close()

