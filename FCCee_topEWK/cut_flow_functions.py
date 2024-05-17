import numpy as np
from operator import add
import pandas as pd
import awkward as ak
from tabulate import tabulate
from scipy import constants
from itertools import compress
from sample_norms import N_expect

'''
This file contains all functions needed for applying the event selection in the main "FCCee_topEWK.py" file. 
'''

###
#Define cuts for cut-flow
###

def compress_col(df_col,mask):
    tmp = []
    for i,index in enumerate(df_col.index):
        tmp.append(list(compress(df_col[index], mask[i])))
    return tmp

#Define function that only keeps those entries that are actual semileptonic events (according to the event function). This is done with genLepton_parentPDG information!
def events_match(df,channel):
    if (channel=="tlepThad")|(channel=="thadTlep"):
        n_Wleptons = 1
    if channel=="thadThad":
        n_Wleptons = 0
    Electron_Wplus = df["genElectron_parentPDG"].apply(lambda row: row.count(24)/2+row.count(6)/2)
    Muon_Wplus = df["genMuon_parentPDG"].apply(lambda row: row.count(24)/2+row.count(6)/2)
    Electron_Wminus = df["genElectron_parentPDG"].apply(lambda row: row.count(-24)/2+row.count(-6)/2)
    Muon_Wminus = df["genMuon_parentPDG"].apply(lambda row: row.count(-24)/2+row.count(-6)/2)
    Leptons_W = Electron_Wplus + Muon_Wplus + Electron_Wminus + Muon_Wminus
    return df[Leptons_W==n_Wleptons]

#Define df loader specifying the amount of events n that should be loaded. The function also yields the variables N_exp/N_df necessary to calculate the rescaling factors for each ntuple
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
    return df,N_exp,N_df
   


'''
Do not dabble with jet energies for now as kt_exactly6 jet algo is now being used where rejecting jets would hamper with later cut criteria, e.g. inverse W mass from two hadronic jet
#jet energy cut: throw away jets with E<10 GeV, i.e. do not consider them as jets
def cut0(input_df,jet_algo):
    print("---Applying cut0 (jet energy cut): throw away jets with E_jet < 10GeV---")
    df = input_df.copy()
    mask = []
    jet_energy = df["jet_{}_energy".format(jet_algo)]
    for i,row in enumerate(jet_energy):
        tmp = []
        for j in range(len(row)):
            tmp.append(row[j] > 10)
        mask.append(tmp)
    df["cut0"]=pd.Series(data=mask,index=df.index)
    jet_px, jet_py, jet_pz, jet_phi, jet_eta = df["jet_{}_px".format(jet_algo)], df["jet_{}_py".format(jet_algo)], df["jet_{}_pz".format(jet_algo)], df["jet_{}_phi".format(jet_algo)],df["jet_{}_eta".format(jet_algo)] 
    df["jet_{}_energy".format(jet_algo)] = pd.Series(data=compress_col(jet_energy,mask),index=df.index)
    df["jet_{}_px".format(jet_algo)] = pd.Series(data=compress_col(jet_px,mask),index=df.index)
    df["jet_{}_py".format(jet_algo)] = pd.Series(data=compress_col(jet_py,mask),index=df.index)
    df["jet_{}_pz".format(jet_algo)] = pd.Series(data=compress_col(jet_pz,mask),index=df.index)
    df["jet_{}_phi".format(jet_algo)] = pd.Series(data=compress_col(jet_phi,mask),index=df.index)
    df["jet_{}_eta".format(jet_algo)] = pd.Series(data=compress_col(jet_eta,mask),index=df.index)
    df["n_jets_{}".format(jet_algo)] = df["cut0"].apply(lambda row: sum(row))
    print("---cut0 applied!---")
    return df    
'''

#isolation/leading cut: Require lepton candidate to be isolated with dR>0.4 to all jets or leading particle if within a jet with E_l/E_jet > 0.5
def phi(px,py):
    tmp = []
    for i in range(len(px)):
        if (px[i]>0)&(py[i]>0):
            tmp.append(np.arctan(py[i]/px[i]))
        if (px[i]<0)&(py[i]>0):
            tmp.append(np.arctan(py[i]/px[i])+np.pi)
        if (px[i]<0)&(py[i]<0):
            tmp.append(np.arctan(py[i]/px[i])+np.pi)
        if (px[i]>0)&(py[i]<0):
            tmp.append(np.arctan(py[i]/px[i])+2*np.pi)
    return tmp

def azim_angle(px,py):
    output = []
    for i,index in enumerate(px.index):
        output.append(phi(px[index],py[index]))
    return output

#Theta not necessary to be calculated because eta (pseudorapidity) is already given as variable in df
def theta(pt,pz):
    theta = []
    for i, val in enumerate(pt):
        if pz[i]>0:
                theta.append(np.arctan(pt[i]/pz[i]))
        elif pz[i]<0:
                theta.append(np.arctan(pt[i]/pz[i])+np.pi)
    return theta

def genTheta(pt,pz):
    Theta = []
    for i,index in enumerate(pt.index):
       Theta.append(theta(pt[index],pz[index]))
    return Theta

def Theta(row):
    tmp = []
    if (len(row)!=0):
        for i,eta in enumerate(row):
            tmp.append(2*np.arctan(np.exp(-eta)))
    return tmp

def pseudorap(Theta):
    tmp = []
    for i, index in enumerate(Theta.index):
        hlp = [];theta=Theta[index];
        for j in range(len(theta)):
            if theta[j]<= (np.pi/2):
                hlp.append((-1)*np.log(np.tan(theta[j]/2)))
            if theta[j] > (np.pi/2):
                hlp.append(np.log(np.tan((np.pi-theta[j])/2)))
        tmp.append(hlp)
    return tmp

def leading_lep(lepton_energy,jet_energy,dr_bool,ratio_lepjet):
    tmp = []
    lepjet = list(compress(jet_energy,dr_bool)) #all jets which overlap with a semileptonic candidate
    if len(lepjet)==1:
        return ((lepton_energy/lepjet[0])>ratio_lepjet)
    elif len(lepjet)>1:
        for i in range(len(lepjet)):
            tmp.append((lepton_energy/lepjet[i])>ratio_lepjet)
        return all(tmp) #only return true (i.e. the lepton is the leading particle inside a jet) if the lepton is leading in all jets with whom it overlaps
    else:
        return True

def dR(phi_lepton,phi_jets,rap_lepton,rap_jets,jet_energy,lepton_energy,ratio_lepjet):
    tmp = []
    if len(phi_lepton)!=0:
        for j in range(len(phi_lepton)):
            dr_bool = []
            for k in range(len(phi_jets)):
                dr = np.sqrt((phi_lepton[j]-phi_jets[k])**2+(rap_lepton[j]-rap_jets[k])**2)
                dr_bool.append(dr<0.4)
            tmp.append(False|leading_lep(lepton_energy[j],jet_energy,dr_bool,ratio_lepjet))            
    else:
        tmp.append(False) #for empty entries [] automatically return False
    return tmp



def df_filter(input_df,mask,lepton_name,cut_name):
    df = input_df.copy()
    lepton_px, lepton_py, lepton_pz, lepton_phi, lepton_eta, lepton_theta, lepton_energy,lepton_charge,lepton_d0,lepton_d0signif = df["{}_px".format(lepton_name)], df["{}_py".format(lepton_name)], df["{}_pz".format(lepton_name)], df["{}_phi".format(lepton_name)],df["{}_eta".format(lepton_name)],df["{}_theta".format(lepton_name)], df["{}_energy".format(lepton_name)],df["{}_charge".format(lepton_name)],df["{}_d0".format(lepton_name)],df["{}_d0signif".format(lepton_name)]
    #save mask to df as column
    df["{}_{}".format(cut_name,lepton_name)] = pd.Series(data=mask,index=df.index)
    df["{}_energy".format(lepton_name)] = pd.Series(data=compress_col(lepton_energy,mask),index=df.index)
    df["{}_px".format(lepton_name)] = pd.Series(data=compress_col(lepton_px,mask),index=df.index)
    df["{}_py".format(lepton_name)] = pd.Series(data=compress_col(lepton_py,mask),index=df.index)
    df["{}_pz".format(lepton_name)] = pd.Series(data=compress_col(lepton_pz,mask),index=df.index)
    df["{}_phi".format(lepton_name)] = pd.Series(data=compress_col(lepton_phi,mask),index=df.index)
    df["{}_eta".format(lepton_name)] = pd.Series(data=compress_col(lepton_eta,mask),index=df.index)
    df["{}_theta".format(lepton_name)] = pd.Series(data=compress_col(lepton_theta,mask),index=df.index)
    df["{}_charge".format(lepton_name)] = pd.Series(data=compress_col(lepton_charge,mask),index=df.index)
    df["{}_d0".format(lepton_name)] = pd.Series(data=compress_col(lepton_d0,mask),index=df.index)
    df["{}_d0signif".format(lepton_name)] = pd.Series(data=compress_col(lepton_d0signif,mask),index=df.index)
    df["n_{}s".format(lepton_name)] = df["{}_{}".format(cut_name,lepton_name)].apply(lambda row: sum(row))
    return df

def cut1(input_df,**kwargs):
    jet_algo = kwargs["jet_algo"]
    print("---Applying cut1: Require lepton candidate to be isolated from all jets with dR > 0.4 or being the leading particle within the jet---")
    mask_electron,mask_muon = [],[]
    df = input_df.copy()
    df["muon_theta"] = df["muon_eta"].apply(lambda row: Theta(row))
    df["electron_theta"] = df["electron_eta"].apply(lambda row: Theta(row))
    #save df columns as variables to make code more comprehensive  
    phi_electron, rap_electron, electron_energy = df["electron_phi"],df["electron_eta"],df["electron_energy"]
    phi_muon, rap_muon, muon_energy = df["muon_phi"],df["muon_eta"],df["muon_energy"]
    phi_jet, rap_jet, jet_energy = df["jet_{}_phi".format(jet_algo)],df["jet_{}_eta".format(jet_algo)],df["jet_{}_energy".format(jet_algo)]
    for i,index in enumerate(df.index):
        mask_muon.append(dR(phi_muon[index],phi_jet[index],rap_muon[index],rap_jet[index],jet_energy[index],muon_energy[index],0.5))
        mask_electron.append(dR(phi_electron[index],phi_jet[index],rap_electron[index],rap_jet[index],jet_energy[index],electron_energy[index],0.5))
    #apply mask_electron/muon to df to throw away all leptons that do not fulfill the isolation/leading criteria prior to the cut-flow
    df = df_filter(df,mask_electron,"electron","cut1")
    df = df_filter(df,mask_muon,"muon","cut1")
    print("---cut1 applied!---")
    return df

#cut2: remove all events with 0 leptons
def cut2(input_df,**kwargs):
    print("---Applying cut2: Require n_muons(n_electrons) > 0---")
    df = input_df.copy()
    df["cut2_muon"] = df["n_muons"]!=0
    df["cut2_electron"] = df["n_electrons"]!=0
    df["cut2"] = df["cut2_muon"] | df["cut2_electron"]
    df = df[df["cut2"]]
    print("---cut2 applied!---")
    return df

'''
#dismiss n_jet cut as used jet algo uses exclusive 6 jet reco
#cut3: require >n_jets jets per event
def cut3(input_df,n_jets,jet_algo):
    print("---Applying cut3: Require n_jets>{} jets per event---".format(n_jets))
    df = input_df.copy()
    df["cut3"] = df["n_jets_{}".format(jet_algo)].apply(lambda row: row>n_jets)
    df = df[df["cut3"]]
    print("---cut3 applied!---")
    return df

#dismiss n_btag cut for now as no b reco algo is implemented yet
#cut: n_btag b-tagged jets per event
def btag(row):
    mask = []
    for i in range(len(row)):
        mask.append(np.abs(row[i])==5) #PDG ID for bottom quark is +-5 -> check if jet flavor that of a b quark or not, i.e. btag jets
    return mask

def cut2(input_df,n_btag,jet_algo):
    print("---Applying cut2: Require >= {} true b_tags per event---".format(n_btag))
    df = input_df.copy()
    if (jet_algo == "default"):
        df["cut2_{}".format(jet_algo)] = df["jet_{}_btag".format(jet_algo)].apply(lambda x: sum(x) >= n_btag)
    else:
        tmp = []
        for i,index in enumerate(df.index):
            tmp.append(btag(df["jet_{}_flavor".format(jet_algo)][index]))
        df["jet_{}_btag".format(jet_algo)] = pd.Series(data = tmp,index=df.index)
        df["cut2_{}".format(jet_algo)] = df["jet_{}_btag".format(jet_algo)].apply(lambda x: sum(x) >= n_btag)
    print("---cut2 applied!---")
    return df
'''

#cut3: ME cut (to filter out "fake" lepton events where a pi0 contained in a jet may deposit most of its energy in the ECAL faking the signature of a lepton - however without the necessary MET that is associated with the semileptonic decay of the W boson into a lepton and a neutrino)
def cut3(input_df,**kwargs):
    ME_cut = kwargs["ME_cut"]
    print("---Applying cut3: ME > {} GeV---".format(ME_cut))
    df = input_df.copy()
    df["cut3"] = df["Emiss_energy"] > ME_cut
    df = df[df["cut3"]]
    print("---cut3 applied!---")
    return df


#cut4: upper and lower momentum cut for leptons
#define cut4 for cut optimisation procedure to get best estimates for upper and lower momentum cut on highest energy lepton

def calc_p(px,py,pz):
    index = px.index
    px,py,pz = ak.Array(px),ak.Array(py),ak.Array(pz)
    p = np.sqrt(px**2+py**2+pz**2)
    return pd.Series(data=p.to_list(),index=index)  

def cut4(input_df,**kwargs):
    p_cut = kwargs["p_cut"]
    comparison = kwargs["comparison"]
    df = input_df.copy()
    electron_px,electron_py,electron_pz = df["electron_px"],df["electron_py"],df["electron_pz"]
    muon_px,muon_py,muon_pz = df["muon_px"],df["muon_py"],df["muon_pz"]
    df["p_muon"] = calc_p(muon_px,muon_py,muon_pz)
    df["p_electron"] = calc_p(electron_px,electron_py,electron_pz)
    p_leptons = df["p_muon"]+df["p_electron"]
    df["p_HE"] = p_leptons.apply(lambda row: ak.max(row)) #highest energy (HE) lepton per event
    if comparison == ">":
        print("---Applying cut4: lower cut on highest energy lepton with p > {} GeV---".format(p_cut))
        df["cut4_{}".format(comparison)] = df["p_HE"].apply(lambda p: p > p_cut)
    elif comparison == "<":
        print("---Applying cut4: upper cut on highest energy lepton with p < {} GeV---".format(p_cut))
        df["cut4_{}".format(comparison)] = df["p_HE"].apply(lambda p: p < p_cut)
    else:
        raise ValueError("Invalid comparison operator")
    df = df[df["cut4_{}".format(comparison)]]
    print("---cut4_{} applied!---".format(comparison))
    return df

#cut5: require semileptonic candidate to have impact parameter d_0 < 0.1mm (+and d0signif = d_0/sqrt(d_0variance) > 10) while having E_l > 20 GeV

def PV_check(row_d0,row_d0signif,row_energy,d0,d0signif,p_lim):
    tmp = []
    if len(row_d0)!=0:
        for i,val in enumerate(row_d0):
            tmp.append((np.abs(val)<d0)&(row_d0signif[i]<d0signif)&(row_energy[i]>p_lim))
    else:
        tmp.append(False)
    return tmp

def cut5(input_df,**kwargs):
    d0 = kwargs["d0"]
    d0signif = kwargs["d0_signif"]
    p_cut = kwargs["p_cut"]
    print("---Applying cut5: Require lepton candidate to have d0 < {} mm and d0_signif < {} plus possess p > {} GeV---".format(d0,d0signif,p_cut))
    mask_electron,mask_muon = [],[]
    df = input_df.copy()
    electron_d0,electron_d0signif,electron_energy = df["electron_d0"],df["electron_d0signif"],df["electron_energy"]
    muon_d0,muon_d0signif,muon_energy = df["muon_d0"],df["muon_d0signif"],df["muon_energy"]
    for i,index in enumerate(df.index):
        mask_electron.append(PV_check(electron_d0[index],electron_d0signif[index],electron_energy[index],d0,d0signif,p_cut))
        mask_muon.append(PV_check(muon_d0[index],muon_d0signif[index],muon_energy[index],d0,d0signif,p_cut))
    #apply filters to electrons and muons respectively
    df = df_filter(df,mask_electron,"electron","cut5")
    df = df_filter(df,mask_muon,"muon","cut5")
    df["cut5"] = df["cut5_electron"].apply(lambda row: any(row)) | df["cut5_electron"].apply(lambda row: any(row))
    df = df[df["cut5"]]
    print("---cut5 applied!---")
    return df

#cut6: Invariant mass from lepton and neutrino, i.e. ME



###
#Calculate efficiency and purity of cut-flow
###

#calculate total amount of signal events to be able to calculate the efficiency(i.e. what percentage of total signal events have been kept) for each cut

#requires the genElectron/genMuon to have the same charge as the W/t it originated from
def Wcharge(genLepton_charge,genLepton_PDG,genW_charge):
    mask= []
    for i,val in enumerate(genLepton_PDG):
        mask.append((val==(genW_charge*24))|(val==(genW_charge*6)))
    return int((all(genLepton_charge[mask]==genW_charge))&(len(genLepton_charge[mask])!=0))


def events_bruteforce(df,n_Wleptons):
    Electron_Wminus,Electron_Wplus,Muon_Wminus,Muon_Wplus = [],[],[],[]
    for i,index in enumerate(df.index):
        Electron_Wplus.append(Wcharge(df["genElectron_charge"][index],df["genElectron_parentPDG"][index],1.0))
        Electron_Wminus.append(Wcharge(df["genElectron_charge"][index],df["genElectron_parentPDG"][index],-1.0))
        Muon_Wplus.append(Wcharge(df["genMuon_charge"][index],df["genMuon_parentPDG"][index],1.0))
        Muon_Wminus.append(Wcharge(df["genMuon_charge"][index],df["genMuon_parentPDG"][index],-1.0))
    Electron_Wminus = pd.Series(data=Electron_Wminus,index=df.index)==n_Wleptons
    Electron_Wplus = pd.Series(data=Electron_Wplus,index=df.index)==n_Wleptons
    Muon_Wminus = pd.Series(data=Muon_Wminus,index=df.index)==n_Wleptons
    Muon_Wplus = pd.Series(data=Muon_Wplus,index=df.index)==n_Wleptons
    return (sum(Electron_Wminus)+sum(Electron_Wplus)+sum(Muon_Wminus)+sum(Muon_Wplus))


def events(df,n_Wleptons):
    if df.empty:
        return 0
    else:
        Electron_Wplus = df["genElectron_parentPDG"].apply(lambda row: row.count(24)/2+row.count(6)/2)
        Muon_Wplus = df["genMuon_parentPDG"].apply(lambda row: row.count(24)/2+row.count(6)/2)
        Electron_Wminus = df["genElectron_parentPDG"].apply(lambda row: row.count(-24)/2+row.count(-6)/2)
        Muon_Wminus = df["genMuon_parentPDG"].apply(lambda row: row.count(-24)/2+row.count(-6)/2)
        Leptons_W = Electron_Wplus + Muon_Wplus + Electron_Wminus + Muon_Wminus
        return sum(Leptons_W == n_Wleptons )


#define signal significance and signal purity (both semileptonic top decays as well as allhadronic ones are considered "signal" -> distinguish eff and pur for semileptonic and hadronic events in the cut-flow tho!)
 
#further, define uncertainties on efficiency and purity in the basis of the paper of Ullrich and Xu which calculates an uncertainty based on the binomially distributed values k which does not fail in the limits of k=0 or k=n. k_s hereby means the number of signal events after a cut and n_s refers to the total number of signal events prior to all cuts. The efficiency is calculated for signal events (tlepThad+thadTlep,tlepTlep) and for the all-hadronic thadThad channel that contributes to the background

def eff_std(k_s,n_s):
    return np.sqrt(((k_s+1)*(k_s+2))/((n_s+2)*(n_s+3))-(k_s+1)**2/(n_s+2)**2)

def pur_std(k_s,k_b,n_s,n_b):
    return np.sqrt(k_s**3/(k_s+k_b)**6 * (1-k_s/n_s)+k_s**2*k_b/(k_s+k_b)**4 * (1-k_b/n_b))

def signal_eff_pur(cut_dic,jet_algo,**kwargs):
    table_SL,table_AH = [],[]
    n_lephad,n_hadlep,n_hadhad = kwargs["tlepThad"],kwargs["thadTlep"],kwargs["thadThad"]
    for i,cut_name in enumerate(cut_dic):
        if i==0:
            n_tot_SL,n_tot_AH = n_lephad[i]+n_hadlep[i],n_hadhad[i] #total number of events before all cuts
        else:
            k_SL,k_AH = n_lephad[i]+n_lephad[i],n_hadhad[i] #number of events after each cut
            k = k_SL+k_AH
            dic_SL, dic_AH = {}, {}
            dic_SL["tT semileptonic"],dic_AH["tT full hadronic"] = cut_name, cut_name
            dic_SL[r"$\epsilon$ [%]"], dic_AH[r"$\epsilon$ [%]"] = np.round((k_SL/n_tot_SL)*100,5),np.round((k_AH/n_tot_AH)*100,5)
            dic_SL[r"$\sigma_{\epsilon}$ [%]"],dic_AH[r"$\sigma_{\epsilon}$ [%]"] = np.round(eff_std(k_SL,n_tot_SL)*100,5),np.round(eff_std(k_AH,n_tot_AH)*100,5)
            dic_SL[r"$\pi$ [%]"], dic_AH[r"$\pi$ [%]"] = np.round((k_SL/k)*100,5),np.round((k_AH/k)*100,5)
            dic_SL[r"$\sigma_{\pi}$ [%]"],dic_AH[r"$\sigma_{\pi}$ [%]"] = np.round(pur_std(k_SL,k_AH,n_tot_SL,n_tot_AH)*100,5),np.round(pur_std(k_AH,k_SL,n_tot_AH,n_tot_SL)*100,5)
            table_SL.append(dic_SL)
            table_AH.append(dic_AH)
    print("---Using jet_{} as jet_algo---".format(jet_algo))
    print("semileptonic efficiency and purity:")
    print(tabulate(table_SL,headers="keys",tablefmt="grid"))
    print("full hadronic efficiency and purity:")
    print(tabulate(table_AH,headers="keys",tablefmt="grid"))
    return table_SL,table_AH

#Define cut-flow -> specify decay channel (because cut flow is applied to tlepThad,thadTlep and thadThad respectively)  -> apply cut flow to df iteratively and calculate number of allhadronic/semileptonic events that remain after each cut to later calculate eff and pur with these numbers
def cut_flow(df,cut_dic,cut_limits_dic,decay_channel):
    table_s = [] #store amount of full hadronic/semileptonic signal events after each cut
    if (decay_channel=="tlepThad")|(decay_channel=="thadTlep"):
        n_Wleptons = 1
    elif (decay_channel=="thadThad"):
        n_Wleptons = 0
    for cut_name in cut_dic:
        df = cut_dic[cut_name](df,**cut_limits_dic[cut_name])
        table_s.append(events(df,n_Wleptons))
    return df,np.array(table_s)

###
#calculate energy and polar angle Theta for leptons
###

def lxcosTheta(df):
    s = 365**2 #square of centre of mass energy in GeV
    m_t = 173.34 #m_t in GeV (taken from literature)
    beta = np.sqrt(1-(4*m_t**2)/s) #top velocity
    c_0 = constants.speed_of_light
    #for electrons
    df_e = df[df["cut2_electron"]]
    electron_charge = np.array(ak.flatten(df_e["electron_charge"]))
    Theta_electron, electron_energy = np.array(ak.flatten(df_e["electron_theta"])), np.array(ak.flatten(df_e["electron_energy"])) 
    x_electron = 2*electron_energy/m_t*np.sqrt((1-beta)/(1+beta))
    Theta_eplus = Theta_electron[electron_charge == +1]
    Theta_eminus = Theta_electron[electron_charge == -1] #distinguish btw electrons and positrons in final state
    x_eplus = x_electron[electron_charge == +1]
    x_eminus = x_electron[electron_charge == -1]
    #for muons
    df_mu = df[df["cut2_muon"]]
    muon_charge = np.array(ak.flatten(df_mu["muon_charge"]))
    Theta_muon, muon_energy = np.array(ak.flatten(df_mu["muon_theta"])), np.array(ak.flatten(df_mu["muon_energy"]))
    x_muon = 2*muon_energy/m_t*np.sqrt((1-beta)/(1+beta))
    Theta_muplus = Theta_muon[muon_charge == +1]
    Theta_muminus = Theta_muon[muon_charge == -1]
    x_muplus = x_muon[muon_charge == +1]
    x_muminus = x_muon[muon_charge == -1]
    #for all leptons
    Theta_lplus,Theta_lminus = np.concatenate((Theta_eplus,Theta_muplus)), np.concatenate((Theta_eminus,Theta_muminus))
    x_lplus,x_lminus = np.concatenate((x_eplus,x_muplus)), np.concatenate((x_eminus,x_muminus))
    return x_lplus,x_lminus,Theta_lplus,Theta_lminus
 
