#this file is solely dedicated on improving the cut-flow by plotting efficiency and purity for different upper or lower cuts on observables (e.g. momenta, ME, etc.) to find the best value to cut 
import numpy as np
import pickle
import pandas as pd
import awkward as ak
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tabulate import tabulate
from scipy import constants
from sample_norms import N_expect
from pathlib import Path
from cut_flow_functions import events_match,events,cut1,cut2,cut3,cut4,cut5,calc_p,df_filter

#Define data loader
def df_load(channel,BSM_mod,chunk=None):
    if chunk==None:
        filepath = "/ceph/skeilbach/FCCee_topEWK/wzp6_ee_SM_tt_{}_noCKMmix_keepPolInfo_{}ecm365.pkl".format(channel,BSM_mod)
    if chunk!=None:
        filepath = "/ceph/skeilbach/FCCee_topEWK/wzp6_ee_SM_tt_{}_noCKMmix_keepPolInfo_{}ecm365/{}.pkl".format(channel,BSM_mod,chunk)
    df = pd.read_pickle(filepath).sample(n=100000,ignore_index=True)
    df = events_match(df,channel)
    N_exp = N_expect["wzp6_ee_SM_tt_{}_noCKMmix_keepPolInfo_{}ecm365".format(channel,BSM_mod)]
    if (channel=="tlepThad")|(channel=="thadTlep"):
        N_df = events(df,1)
    else:
        N_df = events(df,0)
    R_df = N_exp/N_df
    return df,R_df

#load data
df_lephad,R_lephad = df_load("tlepThad","")
df_hadlep,R_hadlep = df_load("thadTlep","")
df_hadhad,R_hadhad = df_load("thadThad","")

jet_algo = "kt_exactly6"

#apply isolation/leading cut and sanity cut first. They are to be untouched
cut_dic = {"cut1": cut1,
	   "cut2": cut2,
           "cut3": cut3,
           "cut4": cut4,
           #"cut5": cut5
	  }
cut_limits = {"cut1": {"jet_algo":jet_algo},
	      "cut2": {"":""}, #cut2 doesnt require any additional cut limits            
              "cut3": {"ME_cut":23},
              "cut4": {"p_cut":13,"comparison": ">"},
              #"cut5": {"d0": 0.05, "d0_signif":50,"p_cut":13}
             }

for cut_name in cut_dic:
    df_lephad = cut_dic[cut_name](df_lephad,**cut_limits[cut_name])
    df_hadlep = cut_dic[cut_name](df_hadlep,**cut_limits[cut_name])
    df_hadhad = cut_dic[cut_name](df_hadhad,**cut_limits[cut_name])
   
###
#Cut Optimisation (Use 100k entries per decay channel)
###

#First Cut -> Compare eff and pur for cut4/cut5 and choose the cut with the best ratio of eff and pur. Cut6 is dealt with seperately/at the end because it is dependent on the lower momentum cut of cut4
cuts = {"cut5": cut5
	}

cut_var = {
	   "cut5": {"d0":0,"d0_signif":0,"p_cut":13}
	   }

cut_limits = {#"d0": np.arange(0.025,0.375,0.025),
              #"d0_signif": np.arange(1,51,1)
              "d0": np.arange(0.030,0.050,0.005),
              "d0_signif": np.arange(1,51,1)
             }
cut_title = {
	     "cut3_<": ["Upper cut on highest energy lepton","Highest energy lepton","Momentum in GeV"],
  	     "cut3_>": ["Lower cut on highest energy lepton","Highest energy lepton","Momentum in GeV"],
	     "cut4": ["Lower cut on Missing energy","Missing energy","ME in GeV"],
	    }

#define function to return number of events before cuts (n) and after the cut has been applied (k). Decay_channel hereby refers to the amount of leptons originating from a W or top, i.e. 1=semileptonic and 0=allhadronic decay channel
def df_cut(df_before_cut,df_after_cut,R_scaling,decay_channel):
    n = R_scaling*events(df_before_cut,decay_channel)
    k = R_scaling*events(df_after_cut,decay_channel)
    return n,k

path_save = Path("/home/skeilbach/FCCee_topEWK/arrays/cut_opt/PV")
path_save.mkdir(parents=True, exist_ok=True)

for cut_name in cuts:
    eff,pur = [],[]
    for d_0 in cut_limits["d0"]:
        cut_var[cut_name]["d0"]=d_0 
        for d0signif in cut_limits["d0_signif"]:
            cut_var[cut_name]["d0_signif"]=d0signif
            n_lephad,k_lephad = df_cut(df_lephad,cuts[cut_name](df_lephad,**cut_var[cut_name]),R_lephad,1)
            n_hadlep,k_hadlep = df_cut(df_hadlep,cuts[cut_name](df_hadlep,**cut_var[cut_name]),R_hadlep,1)
            n_hadhad,k_hadhad = df_cut(df_hadhad,cuts[cut_name](df_hadhad,**cut_var[cut_name]),R_hadhad,0)
            n_s = n_lephad + n_hadlep
            k_s,k_b = k_lephad + k_hadlep, k_hadhad
            eff.append((d_0,d0signif,np.round(k_s/n_s,3)))
            pur.append((d_0,d0signif,np.round(k_s/(k_s+k_b),3)))
    with open(path_save/'{}_eff.pkl'.format(cut_name), 'wb') as f:
        pickle.dump(eff, f)
    with open(path_save/'{}_pur.pkl'.format(cut_name), 'wb') as f:
        pickle.dump(pur, f)



###
#Plot the results 
###
path_save = Path("/home/skeilbach/FCCee_topEWK/arrays/cut_opt")
cut_title = {
             "cut4_<": ["Upper cut on highest energy lepton","Highest energy lepton","Momentum in GeV"],
             "cut4_>": ["Lower cut on highest energy lepton","Highest energy lepton","Momentum in GeV"],
             "cut5": ["Lower cut on Missing energy","Missing energy","ME in GeV"]
            }
cuts = ["cut4_>"]

def calc_p(px,py,pz):
    index = px.index
    px,py,pz = ak.Array(px),ak.Array(py),ak.Array(pz)
    p = np.sqrt(px**2+py**2+pz**2)
    return pd.Series(data=p.to_list(),index=index)  

#Define custom colours
kit_green100=(0,.59,.51)
kit_green50 =(.50,.79,.75)

#Define optimise cut limits
cut_lim = {"cut4_>": [18,0.963,0.907], #[cut_lim,epsilon value, pi value]
	   "cut5": [23,0.982,0.926]
          }

xticks = {"cut4_<": [0,20,40,60,80,100,120],
	  "cut4_>": [0,10,13,20,30,40,50],
	  "cut5": [0,10,20,23,30,40,50,60,70]
 	 }

xtick_labels = {"cut4_<": ["0","20","40","60","80","100","120"],
          	"cut4_>": ["0","10","13","20","30","40","50"],
          	"cut5": ["0","10","20","23","30","40","50","60","70"]
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
    ax1.plot(cut_result[cut_name],cut_result["epsilon"],color=kit_green100,label=r"efficiency $\epsilon$")
    ax1.plot(cut_result[cut_name],cut_result["pi"],color=kit_green50,label=r"purity $\pi$")    
    ax1.plot(cut_result[cut_name],cut_result["epsilon"]*cut_result["pi"],marker=".",color="black",label=r"$\epsilon\cdot\pi$")
    ax1.set_yticks([0,0.2,0.4,0.6,0.8,1])
    ax1.set_xticks(xticks[cut_name])
    if cut_name=="cut4_>":
        ax1.axvline(x=13,color="orange",linestyle="--")
    ax1.set_xticklabels(xtick_labels[cut_name])
    ax1.legend()
    ax1.grid() 
    ax1.set_title(cut_title[cut_name][0])
    #plot the distribution of the cut variable (e.g. momentum/ME) before the cuts
    array_lephad, array_hadlep, array_hadhad = df_array(df_lephad,cut_name),df_array(df_hadlep,cut_name),df_array(df_hadhad,cut_name)
    #Calculate nice bin lims
    binwidth = 1
    xmax = max(np.max(array_lephad),np.max(array_hadlep),np.max(array_hadhad))
    lim = (int(xmax/binwidth)+1)*binwidth
    bins = np.arange(0,lim+binwidth,binwidth)
    counts_lephad, bin_edges = np.histogram(array_lephad,weights=np.full_like(array_lephad,R_lephad),bins=bins)
    counts_hadlep,_ = np.histogram(array_hadlep,weights=np.full_like(array_hadlep,R_hadlep),bins=bin_edges)
    counts_hadhad,_ = np.histogram(array_hadhad,weights=np.full_like(array_hadhad,R_hadhad),bins=bin_edges)
    #save counts
    #np.save(path_save/"counts_SL_{}".format(cut_name),counts_hadlep+counts_lephad)
    #np.save(path_save/"counts_AH_{}".format(cut_name),counts_hadhad)
    #np.save(path_save/"bin_edges_{}".format(cut_name),bin_edges)
    #counts_SL = np.load(path_save/"counts_SL_{}.npy".format(cut_name))
    #counts_AH = np.load(path_save/"counts_AH_{}.npy".format(cut_name))
    counts_SL = counts_hadlep+counts_lephad
    counts_AH = counts_hadhad
    #bin_edges = np.load(path_save/"bin_edges_{}.npy".format(cut_name))
    ax2.stairs(counts_SL,bin_edges,color="red",label=r"semileptonic")
    ax2.stairs(counts_AH,bin_edges,color="blue",label="full hadronic")
    ax2.set_xlim(0,cut_result[cut_name][-1])
    ax2.set_xlabel(cut_title[cut_name][2])
    ax2.set_ylabel("Number of events")
    ax2.set_xticks(xticks[cut_name])
    if cut_name=="cut4_>":
        ax2.axvline(x=13,color="orange",linestyle="--")
        plt.setp(ax2.get_xticklabels()[2], color="orange")
    ax2.set_xticklabels(xtick_labels[cut_name])
    ax2.set_title(cut_title[cut_name][1])
    ax2.legend()
    ax2.grid()
    plt.savefig("/home/skeilbach/FCCee_topEWK/figures/cut_opt/{}.png".format(cut_name),dpi=300)
    plt.close()

###
#Plot results for cut5
###

#unpack x,y and z values for eff and pur
with open(path_save/"cut5_eff.pkl","rb") as f:
    eff = pickle.load(f)

with open(path_save/"cut5_pur.pkl","rb") as f:
    pur = pickle.load(f)

eff_d0 = [val[0] for val in eff]
eff_d0signif = [val[1] for val in eff]
eff_result = [val[2] for val in eff]

pur_d0 = [val[0] for val in pur]
pur_d0signif = [val[1] for val in pur]
pur_result = [val[2] for val in pur]

#plot eff
plt.scatter(eff_d0,eff_d0signif,c=eff_result,cmap="RdYlGn")
plt.colorbar(label="efficiency [%]")
plt.xlabel(r"$d_0$ in mm")
plt.ylabel(r"$d_0/\sigma_{d_0}$")
plt.title(r"Efficiency $\epsilon$ for different impact parameter and significance")
plt.savefig("/home/skeilbach/FCCee_topEWK/figures/cut_opt/eff_plot.png",dpi=300)
plt.close()

#plot eff
plt.scatter(pur_d0,pur_d0signif,c=pur_result,cmap="RdYlGn")
plt.colorbar(label="purity [%]")
plt.xlabel(r"$d_0$ in mm")
plt.ylabel(r"$d_0/\sigma_{d_0}$")
plt.title(r"Purity $\pi$ for different impact parameter and significance")
plt.savefig("/home/skeilbach/FCCee_topEWK/figures/cut_opt/pur_plot.png",dpi=300)
plt.close()


def df_p_filter(input_df):
    df = input_df.copy()
    mask_electron,mask_muon = [],[]
    for i,index in enumerate(df.index):
        mask_muon.append(np.array(df["muon_energy"][index])>13)
        mask_electron.append(np.array(df["electron_energy"][index])>13)
    df = df_filter(df,mask_electron,"electron","p_electron")
    df = df_filter(df,mask_muon,"muon","p_muon")
    return df

df_hadlep = df_p_filter(df_hadlep)
df_lephad = df_p_filter(df_lephad)
df_hadhad = df_p_filter(df_hadhad)

#Define data
d0_hadlep,d0signif_hadlep = np.concatenate((np.array(ak.flatten(df_hadlep["electron_d0"])),np.array(ak.flatten(df_hadlep["muon_d0"])))),np.concatenate((np.array(ak.flatten(df_hadlep["electron_d0signif"])),np.array(ak.flatten(df_hadlep["muon_d0signif"]))))

d0_lephad,d0signif_lephad = np.concatenate((np.array(ak.flatten(df_lephad["electron_d0"])),np.array(ak.flatten(df_lephad["muon_d0"])))),np.concatenate((np.array(ak.flatten(df_lephad["electron_d0signif"])),np.array(ak.flatten(df_lephad["muon_d0signif"]))))

d0_SL,d0signif_SL = np.concatenate((d0_hadlep,d0_lephad)),np.concatenate((d0signif_hadlep,d0signif_lephad))
d0_AH,d0signif_AH = np.concatenate((np.array(ak.flatten(df_hadhad["electron_d0"])),np.array(ak.flatten(df_hadhad["muon_d0"])))),np.concatenate((np.array(ak.flatten(df_hadhad["electron_d0signif"])),np.array(ak.flatten(df_hadhad["muon_d0signif"]))))

#Define nice bins for d0 and d0signif by hand 
xbins_SL = np.arange(-1,1,0.01)
xbins_AH = np.arange(-1,1,0.01)

#Define nice bins
ybins_SL = np.arange(0,75,1)
ybins_AH = np.arange(0,75,1)

#Calculate counts
counts_d0_SL,_ = np.histogram(d0_SL,bins=xbins_SL,weights=np.concatenate((np.full_like(d0_hadlep, R_hadlep,dtype=np.double),np.full_like(d0_lephad,R_lephad,dtype=np.double))))
counts_d0_AH,_ = np.histogram(d0_AH,bins=xbins_AH,weights=np.full_like(d0_AH, R_hadhad,dtype=np.double))

counts_d0signif_SL,_ = np.histogram(d0signif_SL,bins=ybins_SL,weights=np.concatenate((np.full_like(d0signif_hadlep, R_hadlep,dtype=np.double),np.full_like(d0signif_lephad,R_lephad,dtype=np.double))))
counts_d0signif_AH,_ = np.histogram(d0signif_AH,bins=ybins_AH,weights=np.full_like(d0signif_AH,R_hadhad,dtype=np.double))

#Plot histograms

plt.stairs(counts_d0_SL.astype(int),xbins_SL,color="red",label="semileptonic signal")
plt.stairs(counts_d0_AH.astype(int),xbins_AH,color="blue",label="full hadronic signal")
plt.xlabel(r"$d_0$ in mm")
plt.xlim(-1,1)
plt.yscale("log")
plt.ylabel("frequency")
plt.legend()
plt.title(r"Distribution of impact parameter $d_0$ for all leptons with $p>13\,\mathrm{GeV}$")
plt.savefig("/home/skeilbach/FCCee_topEWK/figures/cut_opt/d0_hist_pcut.png",dpi=300)
plt.close()

plt.stairs(counts_d0signif_SL.astype(int),ybins_SL,color="red",label="semileptonic signal")
plt.stairs(counts_d0signif_AH.astype(int),ybins_AH,color="blue",label="full hadronic signal")
plt.xlabel(r"$d_0/\sigma_{d_0}$")
plt.ylabel("frequency")
plt.yscale("log")
plt.xlim(0,75)
plt.legend()
plt.title(r"Distribution of significance $d_0/\sigma_{d_0}$ for all leptons with $p>13\,\mathrm{GeV}$")
plt.savefig("/home/skeilbach/FCCee_topEWK/figures/cut_opt/d0signif_hist_pcut.png",dpi=300)
plt.close()




