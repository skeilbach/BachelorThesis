import numpy as np
from operator import add
import pandas as pd
import awkward as ak
from tabulate import tabulate
from scipy import constants
from itertools import compress

###
#Define cuts for cut-flow
###

def compress_col(df_col,mask):
    return [list(compress(df_col[i], mask[i])) for i in range(len(df_col))]

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

def leading_lep(lepton_energy,jet_energy,dr_bool,E_lepjet):
    tmp = []
    lepjet = list(compress(jet_energy,dr_bool)) #all jets which overlap with a semileptonic candidate
    if len(lepjet)==1:
        return ((lepton_energy/lepjet[0])>E_lepjet)
    else:
        for i in range(len(lepjet)):
            tmp.append((lepton_energy/lepjet[i])>E_lepjet)
        return all(tmp) #only return true (i.e. the lepton is the leading particle inside a jet) if the lepton is leading in all jets with whom it overlaps

def dR(phi_lepton,phi_jets,rap_lepton,rap_jets,jet_energy,lepton_energy,E_lepjet):
    tmp = []
    if len(phi_lepton!=0):
        for j in range(len(phi_lepton)):
            dr_bool = []
            for k in range(len(phi_jets)):
                dr = np.sqrt((phi_lepton[j]-phi_jets[k])**2+(rap_lepton[j]-rap_jets[k])**2)
                dr_bool.append(dr<0.4)
            if sum(dr_bool)!=0:
                tmp.append(False|leading_lep(lepton_energy[j],jet_energy,dr_bool,E_lepjet))
            else:
                tmp.append(True) #if dr_bool contains only False,i.e. len(jet_energy[dr_bool])=0, then we directly know that the lepton is isolated from all jets with dR>0.4
    else:
        tmp.append(True)
    return tmp



def df_filter(input_df,mask,lepton_name):
    df = input_df.copy()
    lepton_px, lepton_py, lepton_pz, lepton_phi, lepton_eta, lepton_theta, lepton_energy,lepton_charge = df["{}_px".format(lepton_name)], df["{}_py".format(lepton_name)], df["{}_pz".format(lepton_name)], df["{}_phi".format(lepton_name)],df["{}_eta".format(lepton_name)],df["{}_theta".format(lepton_name)], df["{}_energy".format(lepton_name)],df["{}_charge".format(lepton_name)]
    df["{}_energy".format(lepton_name)] = pd.Series(data=compress_col(lepton_energy,mask),index=df.index)
    df["{}_px".format(lepton_name)] = pd.Series(data=compress_col(lepton_px,mask),index=df.index)
    df["{}_py".format(lepton_name)] = pd.Series(data=compress_col(lepton_py,mask),index=df.index)
    df["{}_pz".format(lepton_name)] = pd.Series(data=compress_col(lepton_pz,mask),index=df.index)
    df["{}_phi".format(lepton_name)] = pd.Series(data=compress_col(lepton_phi,mask),index=df.index)
    df["{}_eta".format(lepton_name)] = pd.Series(data=compress_col(lepton_eta,mask),index=df.index)
    df["{}_theta".format(lepton_name)] = pd.Series(data=compress_col(lepton_theta,mask),index=df.index)
    df["{}_charge".format(lepton_name)] = pd.Series(data=compress_col(lepton_charge,mask),index=df.index)
    df["n_{}".format(lepton_name)] = df["cut1_{}".format(lepton_name)].apply(lambda row: sum(row))
    return df

def cut1(input_df,jet_algo):
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
    #save masks to df as cut1 columns
    df["cut1_muon"] = pd.Series(data=mask_muon,index=df.index)
    df["cut1_electron"] = pd.Series(data=mask_electron,index=df.index)
    #modify masks to revert the prior "error" where empty rows ([]) where marked with True. This error was done to preserve the overall structure of the df
    electron_index,muon_index = df[df["n_electrons"]==0].index.tolist(),df[df["n_muons"]==0].index.tolist()
    for index in electron_index:
        df.at[index,"cut1_electron"] = [False]
    for index in muon_index:
        df.at[index,"cut1_muon"] = [False] 
    #apply mask_electron/muon to df to throw away all leptons that do not fulfill the isolation/leading criteria prior to the cut-flow
    df = df_filter(df,mask_electron,"electron")
    df = df_filter(df,mask_muon,"muon")
    print("---cut1 applied!---")
    return df

#cut2: remove all events with 0 leptons
def cut2(input_df):
    print("---Applying cut2: Require n_muons(n_electrons) > 0---")
    df = input_df.copy()
    df["cut2_muon"] = df["n_muons"]!=0
    df["cut2_electron"] = df["n_electrons"]!=0
    df["cut2"] = df["cut2_muon"] | df["cut2_electron"]
    df = df[df["cut2"]]
    print("---cut2 applied!---")
    return df

#cut3: require >n_jets jets per event
def cut3(input_df,n_jets,jet_algo):
    print("---Applying cut3: Require n_jets>{} jets per event---".format(n_jets))
    df = input_df.copy()
    df["cut3"] = df["n_jets_{}".format(jet_algo)].apply(lambda row: row>n_jets)
    df = df[df["cut3"]]
    print("---cut3 applied!---")
    return df

'''
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

#cut4: upper and lower momentum cut for leptons
#define cut4 for cut optimisation procedure to get best estimates for upper and lower momentum cut on highest energy lepton
def max_arr(arr):
    if len(arr)==0:
         return 0
    else:
        return ak.max(arr)

def calc_p(px,py,pz):
    index = px.index
    px,py,pz = ak.Array(px),ak.Array(py),ak.Array(pz)
    p = np.sqrt(px**2+py**2+pz**2)
    return pd.Series(data=p.to_list(),index=index)  

def cut4_opt(input_df,cut_l,comparison):
    df = input_df.copy()
    electron_px,electron_py,electron_pz = df["electron_px"],df["electron_py"],df["electron_pz"]
    muon_px,muon_py,muon_pz = df["muon_px"],df["muon_py"],df["muon_pz"]
    df["p_muon"] = calc_p(muon_px,muon_py,muon_pz)
    df["p_electron"] = calc_p(electron_px,electron_py,electron_pz)
    if comparison == ">":
        print("---Applying cut4: lower cut on highest energy lepton with p > {} GeV---".format(cut_l))
        df["cut4_muon"] = df["p_muon"].apply(lambda row: max_arr(row)>cut_l)
        df["cut4_electron"] = df["p_electron"].apply(lambda row: max_arr(row)>cut_l)
    elif comparison == "<":
        print("---Applying cut4: upper cut on highest energy lepton with p < {} GeV---".format(cut_l))
        df["cut4_muon"] = df["p_muon"].apply(lambda row: max_arr(row) < cut_l)
        df["cut4_electron"] = df["p_electron"].apply(lambda row: max_arr(row) < cut_l)
    else:
        raise ValueError("Invalid comparison operator")
    df["cut4"] = df["cut4_muon"] | df["cut4_electron"]
    print("---cut4 applied!---")
    return df

def cut4(input_df,lower_lim,upper_lim):
    df = input_df.copy()
    print("---Applying cut4: {} < p_leading < {} ---".format(lower_lim,upper_lim))
    electron_px,electron_py,electron_pz = df["electron_px"],df["electron_py"],df["electron_pz"]
    muon_px,muon_py,muon_pz = df["muon_px"],df["muon_py"],df["muon_pz"]
    df["p_muon"] = calc_p(muon_px,muon_py,muon_pz)
    df["p_electron"] = calc_p(electron_px,electron_py,electron_pz)
    df["cut4_electron"] = (df["p_electron"].apply(lambda row: max_arr(row)>lower_lim)) & df["p_electron"].apply(lambda row: max_arr(row)<upper_lim)
    df["cut4_muon"] = (df["p_muon"].apply(lambda row: max_arr(row)>lower_lim)) & df["p_muon"].apply(lambda row: max_arr(row)<upper_lim)
    df["cut4"] = df["cut4_electron"] | df["cut4_muon"]
    df = df[df["cut4"]]
    print("---cut4 applied!---")
    return df

#cut5: ME cut (to filter out "fake" lepton events where a pi0 contained in a jet may deposit most of its energy in the ECAL faking the signature of a lepton - however without the necessary MET that is associated with the semileptonic decay of the W boson into a lepton and a neutrino)
def cut5(input_df):
    print("---Applying cut5: ME > 40 GeV---")
    df = input_df.copy()
    df["cut5"] = df["Emiss_energy"] > 40
    df = df[df["cut5"]]
    print("---cut5 applied!---")
    return df

'''
#cut5: Consider only leptons whose reconstructed tracks lie close to the PV -> d0 < 0.1 mm and d0signif = d_0/sqrt(d_0variance) < 50 (tbs if including z0 variable improves purity and efficiency of cut)
def PV(d0,d0sig):
    mask = []
    for i,val in enumerate(d0):
        mask.append((d0[i]<0.1)&(d0sig[i]<50))
    return mask

def PV_TP(arr1,arr2):
    count_TP=0
    for i,val in enumerate(arr1):
        if(arr1[i] & arr2[i]):
            count_TP += 1
    return count_TP

def PV_TN(arr1,arr2):
    count_TN = 0
    for i,val in enumerate(arr1):
        if((arr1[i]==0)&(arr2[i]==0)):
            count_TN += 1
    return count_TN

#this code checks the rate of true positives (TP) and true negatives (TN) and prints the sensitivity(prob. that PV is true when PV_truth=True)  and specificity (prob. that PV is false when PV_truth=False)of the PV cut
def PV_check(df):
    n_PVtrue_electrons = df["electron_IsPrimary_truth"].apply(lambda row: sum(row)).sum()+0.001 #add 0.001 to ensure that we dont divide by 0 if there isnt any true PV at all
    n_PVtrue_muons = df["muon_IsPrimary_truth"].apply(lambda row: sum(row)).sum()+0.001
    n_PVfalse_muons = df["n_muons"].sum() - n_PVtrue_muons
    n_PVfalse_electrons = df["n_electrons"].sum() - n_PVtrue_electrons
    n_TP_electrons= df[["electron_IsPrimary_truth","cut5_electron"]].apply(lambda row: PV_TP(row["electron_IsPrimary_truth"],row["cut5_electron"]),axis=1).sum()
    n_TP_muons = df[["muon_IsPrimary_truth","cut5_muon"]].apply(lambda row: PV_TP(row["muon_IsPrimary_truth"],row["cut5_muon"]),axis=1).sum()
    n_TN_electrons= df[["electron_IsPrimary_truth","cut5_electron"]].apply(lambda row: PV_TN(row["electron_IsPrimary_truth"],row["cut5_electron"]),axis=1).sum()
    n_TN_muons = df[["muon_IsPrimary_truth","cut5_muon"]].apply(lambda row: PV_TN(row["muon_IsPrimary_truth"],row["cut5_muon"]),axis=1).sum()
    table = [{"": "electrons", "TP": n_TP_electrons,"TP/P_total [%]": np.round(n_TP_electrons/n_PVtrue_electrons*100,2), "TN/N_total [%]": np.round(n_TN_electrons/n_PVfalse_electrons*100,2)},
             {"": "muons", "TP": n_TP_muons, "TP/P_total [%]": np.round(n_TP_muons/n_PVtrue_muons*100,2), "TN/N_total [%]": np.round(n_TN_muons/n_PVfalse_muons*100,2)}]
    print("---PV check successfull:---")
    print(tabulate(table,headers="keys",tablefmt="grid"))

def cut5(input_df):
    print("---Applying cut5: PV criteria with d0&z0 < 0.1 mm and d0sig&z0sig < 50 ---")
    df = input_df.copy()
    mask_muon = [];mask_electron = []
    for i,index in enumerate(df.index):
        mask_muon.append(PV(df["muon_d0"][index],df["muon_d0signif"][index]))
        mask_electron.append(PV(df["electron_d0"][index],df["electron_d0signif"][index]))
    df["cut5_muon"] = pd.Series(data=mask_muon,index=df.index)
    df["cut5_electron"] = pd.Series(data=mask_electron,index=df.index)
    df["cut5"] = (df["cut5_muon"].apply(lambda row: any(row)))|(df["cut5_electron"].apply(lambda row: any(row))) #this tests if an event includes a lepton that fulfills the PV criterion which makes it a prime candidate for having originated from a semileptonic top decay. Thus it is assigned as a signal event 
    print("---cut5 applied!---")
    return df
'''

#cut7: inverse W mass cut: Compare chi2 values for all semileptonic candidates in each event
def BW_resonance(E,a,E_0):
    return a/(a**2/4+(E-E_0)**2)

def mass_W(E_l,E_nu,px_l,py_l,pz_l,px_nu,py_nu,pz_nu):
    tmp = []
    for i in range(len(E_l)):
        tmp.append(2*(E_l[i]*E_nu+px_l[i]*px_nu+py_l[i]*py_nu+pz_l[i]*pz_nu))
    return tmp
        
def m_W_lep(input_df):
    df = input_df.copy()
    df["m_W_electron"] = df[["Emiss","Emiss_px","Emiss_py","Emiss_pz","electron_energy","electron_px","electron_py","electron_pz"]].apply(lambda row: mass_W(row["electron_energy"],row["Emiss"],row["electron_px"],row["electron_py"],row["electron_pz"],row["Emiss_py"],row["Emiss_py"],row["Emiss_pz"]),axis=1)
    df["m_W_muon"] = df[["Emiss","Emiss_px","Emiss_py","Emiss_pz","muon_energy","muon_px","muon_py","muon_pz"]].apply(lambda row: mass_W(row["muon_energy"],row["Emiss"],row["muon_px"],row["muon_py"],row["muon_pz"],row["Emiss_py"],row["Emiss_py"],row["Emiss_pz"]),axis=1)
    m_W_electron = np.array(ak.flatten(df["m_W_electron"]))
    m_W_muon = np.array(ak.flatten(df["m_W_muon"]))
    return m_W_electron + m_W_muon

###
#Calculate efficiency and purity of cut-flow
###

#calculate total amount of signal events to be able to calculate the efficiency(i.e. what percentage of total signal events have been kept) for each cut
def events(df,n_Wleptons):
    Electron_Wplus = df["genElectron_parentPDG"].apply(lambda row: row.count(24)/2+row.count(6)/2)
    Electron_Wminus = df["genElectron_parentPDG"].apply(lambda row: row.count(-24)/2+row.count(-6)/2)
    Muon_Wplus = df["genMuon_parentPDG"].apply(lambda row: row.count(24)/2+row.count(6)/2)
    Muon_Wminus = df["genMuon_parentPDG"].apply(lambda row: row.count(-24)/2+row.count(-6)/2)
    Leptons_W = Electron_Wplus + Electron_Wminus + Muon_Wplus + Muon_Wminus
    return sum(Leptons_W == n_Wleptons )

#define signal significance and signal purity. In this context signal is referred to as an event containing a semileptonic top decay. Every other event (e.g. tt_hadhad) is assigned as background 
#further, define uncertainties on efficiency and purity in the basis of the paper of Ullrich and Xu which calculates an uncertainty based on the binomially distributed values k which does not fail in the limits of k=0 or k=n. k_s hereby means the number of signal events after a cut and n_s refers to the total number of signal events prior to all cuts. The efficiency is calculated for signal events (tlepThad+thadTlep,tlepTlep) and for the all-hadronic thadThad channel that contributes to the background

def eff_std(k_s,n_s):
    return np.sqrt(((k_s+1)*(k_s+2))/((n_s+2)*(n_s+3))-(k_s+1)**2/(n_s+2)**2)

def pur_std(k_s,n):
    return np.sqrt(((k_s+1)*(k_s+2))/((n+2)*(n+3))-(k_s+1)**2/(n+2)**2)

#signal_eff_pur(cut_names,table_lephad,table_hadlep,table_hadhad,jet_algo,N_exp,N_SL,N_AH,BR_SL,BR_AH)

def signal_eff_pur(cut_names,n_lephad,n_hadlep,n_hadhad,jet_algo,R_SL,R_AH):
    table_SL,table_AH = [],[]
    n_lephad,n_hadlep,n_hadhad = np.array(n_lephad),np.array(n_hadlep),np.array(n_hadhad)
    #rescale MC entries
    n_tot_SL,n_tot_AH = R_SL*(n_lephad[0]+n_hadlep[0]), R_AH*n_hadhad[0] #total number of events before all cuts
    k_SL,k_AH = R_SL*(n_lephad[1:]+n_hadlep[1:]),R_AH*n_hadhad[1:] #number of events after each cut
    k = k_SL + k_AH
    for i,cut_name in enumerate(cut_names):
        dic_SL, dic_AH = {}, {}
        dic_SL["tT semileptonic"],dic_AH["tT all-hadronic"] = cut_name, cut_name
        dic_SL[r"$\epsilon$ [%]"], dic_AH[r"$\epsilon$ [%]"] = np.round((k_SL[i]/n_tot_SL)*100,3),np.round((k_AH[i]/n_tot_AH)*100,3)
        dic_SL[r"$\sigma_{\epsilon}$ [%]"],dic_AH[r"$\sigma_{\epsilon}$ [%]"] = np.round(eff_std(k_SL[i],n_tot_SL)*100,3),np.round(eff_std(k_AH[i],n_tot_AH)*100,3)
        dic_SL[r"$\pi$ [%]"], dic_AH[r"$\pi$ [%]"] = np.round((k_SL[i]/k[i])*100,3),np.round((k_AH[i]/k[i])*100,3)
        dic_SL[r"$\sigma_{\pi}$ [%]"],dic_AH[r"$\sigma_{\pi}$ [%]"] = np.round(pur_std(k_SL[i],k[i])*100,3),np.round(pur_std(k_AH[i],k[i])*100,3)
        table_SL.append(dic_SL)
        table_AH.append(dic_AH)
    print("---Using jet_{} as jet_algo---".format(jet_algo))
    print("semileptonic efficiency and purity:")
    print(tabulate(table_SL,headers="keys",tablefmt="grid"))
    print("allhdronic efficiency and purity:")
    print(tabulate(table_AH,headers="keys",tablefmt="grid"))
    return table_SL,table_AH

#Define cut-flow
def cut_flow(df,jet_algo,decay_channel):
    table_s = []
    if (decay_channel=="semileptonic"):
        n_Wleptons = 1
    elif (decay_channel=="allhadronic"):
        n_Wleptons = 0
    table_s.append(events(df,n_Wleptons))
    df = cut0(df,jet_algo) #only filters out jets with E<10 GeV so doesnt influence number of lephad and hadhad events
    df = cut1(df,jet_algo) #rejects non leading/isolated leptons 
    df = cut2(df)
    table_s.append(events(df,n_Wleptons))
    df = cut3(df,3,jet_algo)
    table_s.append(events(df,n_Wleptons))
    df = cut4(df,10,120)
    table_s.append(events(df,n_Wleptons))
    df = cut5(df)
    table_s.append(events(df,n_Wleptons))
    return df,table_s

###
# apply filters and leptons
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

###
#apply filters for genLeptons
###

#create mask for leptons originating from a W+ or W- (PDG code: +-24)
def genMask(row):
    row_bool = (np.abs(np.array(row)) == 24) | (np.abs(np.array(row)) == 6)
    if(sum(row_bool)>1):
        row_bool[np.where(row_bool==1)[0][1:]]=False #this deletes duplicated semileptonic muons or electrons 
    return row_bool

def LxcosTheta(df):
    s = 365**2 #square of centre of mass energy in GeV
    m_t = 173 #top mass in GeV
    beta = np.sqrt(1-(4*m_t**2)/s) #top velocity
    c_0 = constants.speed_of_light
    df["pT_genElectron"] = np.sqrt(df["genElectron_px"]**2+df["genElectron_py"]**2)
    df["pT_genMuon"] = np.sqrt(df["genMuon_px"]**2+df["genMuon_py"]**2)
    df["Theta_genElectron"] = genTheta(df["pT_genElectron"],df["genElectron_pz"])
    df["Theta_genMuon"] = genTheta(df["pT_genMuon"],df["genMuon_pz"])          
    df["genElectron_ID_24"] = df["genElectron_parentPDG"].apply(lambda row: genMask(row)) 
    df["genMuon_ID_24"] = df["genMuon_parentPDG"].apply(lambda row: genMask(row)) 
    genElectron_mask = np.array(ak.flatten(df["genElectron_ID_24"]))
    genMuon_mask = np.array(ak.flatten(df["genMuon_ID_24"]))
    genElectron_charge = np.array(ak.flatten(df["genElectron_charge"]))
    genMuon_charge = np.array(ak.flatten(df["genMuon_charge"]))
    genElectron_plus = genElectron_mask & (genElectron_charge == 1)
    genElectron_minus = genElectron_mask & (genElectron_charge == -1)
    genMuon_plus = genMuon_mask & (genMuon_charge == 1)
    genMuon_minus = genMuon_mask & (genMuon_charge == -1) 
    #import arrays and apply masks
    Theta_genElectron, genElectron_energy = np.array(ak.flatten(df["Theta_genElectron"])),np.array(ak.flatten(df["genElectron_energy"]))
    Theta_genMuon, genMuon_energy = np.array(ak.flatten(df["Theta_genMuon"])), np.array(ak.flatten(df["genMuon_energy"]))
    x_genElectron = 2*genElectron_energy/m_t*np.sqrt((1-beta)/(1+beta))
    x_genMuon = 2*genMuon_energy/m_t*np.sqrt((1-beta)/(1+beta))
    Theta_genEplus = Theta_genElectron[genElectron_plus]
    Theta_genEminus = Theta_genElectron[genElectron_minus]
    Theta_genMuplus = Theta_genMuon[genMuon_plus]
    Theta_genMuminus = Theta_genMuon[genMuon_minus]
    x_genEplus = x_genElectron[genElectron_plus]
    x_genEminus = x_genElectron[genElectron_minus]
    x_genMuplus = x_genMuon[genMuon_plus]
    x_genMuminus = x_genMuon[genMuon_minus]
    x_genLplus,x_genLminus = np.concatenate((x_genEplus,x_genMuplus)),np.concatenate((x_genEminus,x_genMuminus))
    Theta_genLplus,Theta_genLminus = np.concatenate((Theta_genEplus,Theta_genMuplus)), np.concatenate((Theta_genEminus,Theta_genMuminus))
    return x_genLplus,x_genLminus,Theta_genLplus,Theta_genLminus

