#this file is solely dedicated on improving the cut-flow by plotting efficiency and purity for different upper or lower cuts on observables (e.g. momenta, ME, etc.) to find the best value to cut 
import numpy as np
import pickle
import pandas as pd
import awkward as ak
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy import constants
from sample_norms import N_expect
from pathlib import Path
from cut_flow_functions import events,cut1,cut2,cut4,cut5,cut6,calc_p

#load data
path_df = Path('/home/skeilbach/FCCee_topEWK')
df_lephad = pd.read_pickle(path_df/"wzp6_ee_SM_tt_tlepThad_noCKMmix_keepPolInfo_ecm365.pkl").sample(n=100000,ignore_index=True)
df_hadlep = pd.read_pickle(path_df/"wzp6_ee_SM_tt_thadTlep_noCKMmix_keepPolInfo_ecm365.pkl").sample(n=100000,ignore_index=True)
df_hadhad = pd.read_pickle(path_df/"wzp6_ee_SM_tt_thadThad_noCKMmix_keepPolInfo_ecm365.pkl").sample(n=100000,ignore_index=True)
jet_algo = "kt_exactly6"

#Scaling for each df 
N_exp_lephad = N_expect["wzp6_ee_SM_tt_tlepThad_noCKMmix_keepPolInfo_ecm365"]
N_exp_hadlep = N_expect["wzp6_ee_SM_tt_thadTlep_noCKMmix_keepPolInfo_ecm365"]
N_exp_hadhad = N_expect["wzp6_ee_SM_tt_thadThad_noCKMmix_keepPolInfo_ecm365"]


N_lephad = events(df_lephad,1)
N_hadlep = events(df_hadlep,1)
N_hadhad = events(df_hadhad,0)

R_lephad = N_exp_lephad/N_lephad
R_hadlep = N_exp_hadlep/N_hadlep
R_hadhad = N_exp_hadhad/N_hadhad

#apply isolation/leading cut and sanity cut first. They are to be untouched
df_lephad = cut1(df_lephad,jet_algo)
df_hadlep = cut1(df_hadlep,jet_algo)
df_hadhad = cut1(df_hadhad,jet_algo)
df_lephad = cut2(df_lephad)
df_hadlep = cut2(df_hadlep)
df_hadhad = cut2(df_hadhad)
df_lephad = cut5(df_lephad,23)
df_hadlep = cut5(df_hadlep,23)
df_hadhad = cut5(df_hadhad,23)

###
#Cut Optimisation (Use 100k entries per decay channel)
###

#First Cut -> Compare eff and pur for cut4/cut5 and choose the cut with the best ratio of eff and pur. Cut6 is dealt with seperately/at the end because it is dependent on the lower momentum cut of cut4
cuts = {
	#"cut4_<":cut4, #upper cut on p_leading lepton
	"cut4_>":cut4, #lower cut on p_leading lepton
	#"cut5":cut5
	}

cut_var = {
	   #"cut4_<": np.arange(5,131,1),
	   "cut4_>": np.arange(0,51,1),
	   #"cut5": np.arange(0,71,1)
	   }

cut_title = {
	     #"cut4_<": ["Upper cut on highest energy lepton","Highest energy lepton","Momentum in GeV"],
  	     "cut4_>": ["Lower cut on highest energy lepton","Highest energy lepton","Momentum in GeV"],
	     "cut5": ["Lower cut on Missing energy","Missing energy","ME in GeV"]
	    }

#define function to return number of events before cuts (n) and after the cut has been applied (k). Decay_channel hereby refers to the amount of leptons originating from a W or top, i.e. 1=semileptonic and 0=allhadronic decay channel
def df_cut(df_before_cut,df_after_cut,R_scaling,decay_channel):
    n = R_scaling*events(df_before_cut,decay_channel)
    k = R_scaling*events(df_after_cut,decay_channel)
    return n,k

path_save = Path("/home/skeilbach/FCCee_topEWK/arrays/cut_opt")
path_save.mkdir(parents=True, exist_ok=True)

for cut_name in cuts:
    cut_result = {}
    eff,pur = [],[] #eff and purity are only calculated with respect to the semileptonic signal as the event selection aims to enrich the sample with semileptonic signal and discard allhadronic signal
    var = cut_var[cut_name]
    cut_result[cut_name] = var
    if cut_name=="cut5":
        for i in range(len(var)):
            n_lephad,k_lephad = df_cut(df_lephad,cuts[cut_name](df_lephad,var[i]),R_lephad,1)
            n_hadlep,k_hadlep = df_cut(df_hadlep,cuts[cut_name](df_hadlep,var[i]),R_hadlep,1)
            n_hadhad,k_hadhad = df_cut(df_hadhad,cuts[cut_name](df_hadhad,var[i]),R_hadhad,0)
            n_s = n_lephad + n_hadlep
            k_s,k_b = k_lephad + k_hadlep, k_hadhad
            eff.append(np.round(k_s/n_s,3))
            pur.append(np.round(k_s/(k_s+k_b),3))
    else:
        for i in range(len(var)):
            n_lephad,k_lephad = df_cut(df_lephad,cuts[cut_name](df_lephad,var[i],cut_name[-1]),R_lephad,1)
            n_hadlep,k_hadlep = df_cut(df_hadlep,cuts[cut_name](df_hadlep,var[i],cut_name[-1]),R_hadlep,1)
            n_hadhad,k_hadhad = df_cut(df_hadhad,cuts[cut_name](df_hadhad,var[i],cut_name[-1]),R_hadhad,0)
            n_s = n_lephad + n_hadlep
            k_s,k_b = k_lephad + k_hadlep, k_hadhad
            eff.append(np.round(k_s/n_s,3))
            pur.append(np.round(k_s/(k_s+k_b),3))
    cut_result["epsilon"] = np.array(eff)
    cut_result["pi"] = np.array(pur)
    with open(path_save/'{}.pkl'.format(cut_name), 'wb') as f:
        pickle.dump(cut_result, f)

###
#Plot the results 
###

def calc_p(px,py,pz):
    index = px.index
    px,py,pz = ak.Array(px),ak.Array(py),ak.Array(pz)
    p = np.sqrt(px**2+py**2+pz**2)
    return pd.Series(data=p.to_list(),index=index)  

#Define custom colours
kit_green100=(0,.59,.51)
kit_green50 =(.50,.79,.75)

#Define optimise cut limits
cut_lim = {#"cut4_>": [18,0.963,0.907], #[cut_lim,epsilon value, pi value]
	   "cut5": [23,0.982,0.926]
          }

#Define function that calculates momentum/ME based on cut_name for a df
def df_array(df,cut_name):
    if cut_name[:-2]=="cut4":
        p_electron = calc_p(df["electron_px"],df["electron_py"],df["electron_pz"])
        p_muon = calc_p(df["muon_px"],df["muon_py"],df["muon_pz"])
        p_lepton = p_electron+p_muon
        p_HE = p_lepton.apply(lambda row: ak.max(row))
        return np.array(p_HE)
    elif cut_name=="cut5":
        return np.array(ak.flatten(df["Emiss_energy"]))

#Plot results for all cuts
for cut_name in cuts:
    with open(path_save/'{}.pkl'.format(cut_name),'rb') as f:
        cut_result = pickle.load(f)      
    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
    ax1.plot(cut_result[cut_name],cut_result["epsilon"],marker="o",color=kit_green100,label=r"efficiency $\epsilon$")
    ax1.plot(cut_result[cut_name],cut_result["pi"],marker="o",color=kit_green50,label=r"purity $\pi$")    
    ax1.plot(cut_result[cut_name],cut_result["epsilon"]*cut_result["pi"],marker=".",color="black",label=r"$\epsilon\cdot\pi$")
    ax1.set_yticks([0,0.2,0.4,0.6,0.8,1])
    ax1.legend()
    ax1.grid() 
    ax1.set_title(cut_title[cut_name][0])
    #plot the distribution of the cut variable (e.g. momentum/ME) before the cuts
    array_lephad, array_hadlep, array_hadhad = df_array(df_lephad,cut_name),df_array(df_hadlep,cut_name),df_array(df_hadhad,cut_name)
    counts_lephad, bin_edges = np.histogram(array_lephad,weights=np.full_like(array_lephad,R_lephad),bins=len(cut_result[cut_name]))
    counts_hadlep,_ = np.histogram(array_hadlep,weights=np.full_like(array_hadlep,R_hadlep),bins=bin_edges)
    counts_hadhad,_ = np.histogram(array_hadhad,weights=np.full_like(array_hadhad,R_hadhad),bins=bin_edges)
    #save counts
    np.save(path_save/"counts_SL_{}".format(cut_name),counts_hadlep+counts_lephad)
    np.save(path_save/"counts_AH_{}".format(cut_name),counts_hadhad)
    np.save(path_save/"bin_edges_{}".format(cut_name),bin_edges)
    counts_SL = np.load(path_save/"counts_SL_{}.npy".format(cut_name))
    counts_AH = np.load(path_save/"counts_AH_{}.npy".format(cut_name))
    bin_edges = np.load(path_save/"bin_edges_{}.npy".format(cut_name))
    ax2.stairs(counts_SL,bin_edges,color="red",label=r"semileptonic")
    ax2.stairs(counts_AH,bin_edges,color="blue",label="allhadronic")
    ax2.set_xlim(0,cut_result[cut_name][-1])
    ax2.set_xlabel(cut_title[cut_name][2])
    ax2.set_ylabel("frequency")
    ax2.set_title(cut_title[cut_name][1])
    ax2.legend()
    ax2.grid()
    plt.savefig("/home/skeilbach/FCCee_topEWK/figures/cut_opt/{}.png".format(cut_name),dpi=300)
    plt.close()



