import numpy as np
import pandas as pd
import awkward as ak
from tabulate import tabulate
from scipy import constants

#calculate total amount of signal events to be able to calculate the efficiency(i.e. what percentage of total signal events have been kept) for each cut
def events(df,n_Wleptons):
    Electron_Wplus = df["genElectron_parentPDG"].apply(lambda row: row.count(24)/2)
    Electron_Wminus = df["genElectron_parentPDG"].apply(lambda row: row.count(-24)/2)
    Muon_Wplus = df["genMuon_parentPDG"].apply(lambda row: row.count(24)/2)
    Muon_Wminus = df["genMuon_parentPDG"].apply(lambda row: row.count(-24)/2)
    Leptons_W = Electron_Wplus + Electron_Wminus + Muon_Wplus + Muon_Wminus
    return sum(Leptons_W == n_Wleptons )  

#define signal significance and signal purity. In this context signal is referred to as an event containing a semileptonic top decay. Every other event (e.g. tt_hadhad) is assigned as background 
#further, define uncertainties on efficiency and purity in the basis of the paper of Ullrich and Xu which calculates an uncertainty based on the binomially distributed values k which does not fail in the limits of k=0 or k=n. k_s hereby means the number of signal events after a cut and n_s refers to the total number of signal events prior to all cuts. The efficiency is calculated for signal events (tlepThad+thadTlep,tlepTlep) and for the all-hadronic thadThad channel that contributes to the background

def eff_std(k_s,n_s):
    return np.sqrt(((k_s+1)*(k_s+2))/((n_s+2)*(n_s+3))-(k_s+1)**2/(n_s+2)**2)

def pur_std(k_s,n):
    return np.sqrt(((k_s+1)*(k_s+2))/((n+2)*(n+3))-(k_s+1)**2/(n+2)**2)
    
def signal_eff_pur(cut_name,df,parentID,n_s,n_b,table_s,table_b):
    dic_s, dic_b = {}, {}
    k_s = events(df,1) + events(df,2)
    k_b = events(df,0)
    k = k_s + k_b
    dic_s["tT non-allhadronic"],dic_b["tT all-hadronic"] = cut_name, cut_name
    dic_s[r"$\epsilon$ [%]"], dic_b[r"$\epsilon$ [%]"] = np.round((k_s/n_s)*100,2),np.round((k_b/n_b)*100,2)
    dic_s[r"$\sigma_{\epsilon}$ [%]"],dic_b[r"$\sigma_{\epsilon}$ [%]"] = np.round(eff_std(k_s,n_s)*100,2),np.round(eff_std(k_b,n_b)*100,2)
    dic_s[r"$\pi$ [%]"], dic_b[r"$\pi$ [%]"] = np.round((k_s/k)*100,2),np.round((k_b/k)*100,2)
    dic_s[r"$\sigma_{\pi}$ [%]"],dic_b[r"$\sigma_{\pi}$ [%]"] = np.round(pur_std(k_s,k)*100,2),np.round(pur_std(k_b,k)*100,2)
    table_s.append(dic_s)
    table_b.append(dic_b)
    return None

#Define the cuts
#jet energy cut: throw away jets with E<10 GeV, i.e. do not consider them as jets
def E_jet_cut(input_df,jet_algo):
    df = input_df.copy()
    mask = []
    jet_energy = df["jet_{}_energy".format(jet_algo)]
    for i,index in enumerate(df.index):
        tmp = []
        for j in range(len(jet_energy[index])):
            tmp.append(jet_energy[index][j] > 10)
        mask.append(tmp)
    jet_px, jet_py, jet_pz, n_jets = df["jet_{}_px".format(jet_algo)], df["jet_{}_py".format(jet_algo)], df["jet_{}_pz".format(jet_algo)], df["n_jets_{}".format(jet_algo)] 
    #df["jet_{}_energy".format(jet_algo)] = pd.Series(data=,index=df.index).fillna(False)

#cut0: remove all events with 0 leptons
def cut0(input_df):
    print("---Applying cut0: Require n_muons(n_electrons) > 0---")
    df = input_df.copy()
    df["cut0_muon"] = df["n_muons"]!=0
    df["cut0_electron"] = df["n_electrons"]!=0
    df["cut0"] = df["cut0_muon"] | df["cut0_electron"]
    print("---cut0 applied!---")
    return df

#cut1: require >n_jets jets per event
def cut1(input_df,n_jets,jet_algo):
    print("---Applying cut1: Require more than {} jets per event---".format(n_jets))
    df = input_df.copy()
    if(jet_algo == "default"):
        df["cut1_{}".format(jet_algo)] = df["n_jets_{}".format(jet_algo)]>n_jets
    else:
        tmp = []
        for i,index in enumerate(df.index):
            tmp.append(len(df["jet_{}_flavor".format(jet_algo)][index]))
        df["n_jets_{}".format(jet_algo)] = pd.Series(data=tmp,index=df.index)
        df["cut1_{}".format(jet_algo)] = df["n_jets_{}".format(jet_algo)]>n_jets
    print("---cut1 applied!---")
    return df

#cut2: n_btag b-tagged jets per event
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

#cut3: ME cut (to filter out "fake" lepton events where a pi0 contained in a jet may deposit most of its energy in the ECAL faking the signature of a lepton - however without the necessary MET that is associated with the semileptonic decay of the W boson into a lepton and a neutrino)
def cut3(input_df):
    print("---Applying cut3: ME > 40 GeV---")
    df = input_df.copy()
    df["pT_electron"] = np.sqrt(df["electron_px"]**2+df["electron_py"]**2)
    df["pT_muon"] = np.sqrt(df["muon_px"]**2+df["muon_py"]**2)
    df["cut3"] = df["Emiss_energy"] > 40
    print("---cut3 applied!---")
    return df

#cut4: Require lepton candidate to be isolated with dR>0.4 to all jets or leading particle if within a jet with E_l/E_jet > 0.5
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

def theta(pt,pz):
    theta = []
    for i, val in enumerate(pt):
        if pz[i]>0:
                theta.append(np.arctan(pt[i]/pz[i]))
        elif pz[i]<0:
                theta.append(np.arctan(pt[i]/pz[i])+np.pi)
    return theta

def Theta(pt,pz):
    Theta = []
    for i,index in enumerate(pt.index):
       Theta.append(theta(pt[index],pz[index]))
    return Theta

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

def dR(phi_lepton,phi_jets,rap_lepton,rap_jets,jet_energy,lepton_energy):
    tmp = []
    for j in range(len(phi_lepton)):
        dr_bool = []
        for k in range(len(phi_jets)):
            dr = np.sqrt((phi_lepton[j]-phi_jets[k])**2+(rap_lepton[j]-rap_jets[k])**2)
            dr_bool.append(dr<0.4)
        if len(jet_energy[dr_bool])!=0:
            tmp.append((sum(dr_bool) == 0)|(lepton_energy[j]/(jet_energy[dr_bool][0]) > 0.5))
        else:
            tmp.append(True) #if dr_bool contains only False,i.e. len(jet_energy[dr_bool])=0, then we directly know that the lepton is isolated from all jets with dR>0.4
    return tmp

def cut4(input_df,jet_algo):
    print("---Applying cut4: Require lepton candidate to be isolated from all jets with dR > 0.4 or being the leading particle within the jet---")
    mask_muon = [];mask_electron = [];
    df = input_df.copy()
    df_e,df_mu = df[df["cut0_electron"]].copy(), df[df["cut0_muon"]].copy()
    df["pT_jet"] = np.sqrt((df["jet_{}_px".format(jet_algo)])**2+(df["jet_{}_py".format(jet_algo)])**2)
    df_mu["Theta_muon"] = Theta(df_mu["pT_muon"],df_mu["muon_pz"])
    df_e["Theta_electron"] = Theta(df_e["pT_electron"],df_e["electron_pz"])
    df["Theta_jet"] = Theta(df["pT_jet"],df["jet_{}_pz".format(jet_algo)])
    #save df columns as variables to make code more comprehensive  
    phi_electron, rap_electron, electron_energy = df_e["electron_phi"],df_e["electron_eta"],df_e["electron_energy"]
    phi_muon, rap_muon, muon_energy = df_mu["muon_phi"],df_mu["muon_eta"],df_mu["muon_energy"]
    phi_jet, rap_jet, jet_energy = df["jet_{}_phi".format(jet_algo)],df["jet_{}_eta".format(jet_algo)],df["jet_{}_energy".format(jet_algo)]
    for i,index in enumerate(df_mu.index):
        mask_muon.append(dR(phi_muon[index],phi_jet[index],rap_muon[index],rap_jet[index],jet_energy[index],muon_energy[index]))
    for i,index in enumerate(df_e.index):
        mask_electron.append(dR(phi_electron[index],phi_jet[index],rap_electron[index],rap_jet[index],jet_energy[index],electron_energy[index]))
    df_mu["cut4_muon"] = pd.Series(data=mask_muon,index=df_mu.index)
    df_e["cut4_electron"] = pd.Series(data=mask_electron,index=df_e.index)
    df["cut4_muon"] = pd.Series(data=df_mu["cut4_muon"],index=df.index).fillna(False)
    df["cut4_electron"] = pd.Series(data=df_e["cut4_electron"],index=df.index).fillna(False)
    df["cut4"] = pd.Series(data=df_e["cut4_electron"].apply(lambda row: ak.any(row)),index=df.index).fillna(False) | pd.Series(data=df_mu["cut4_muon"].apply(lambda row: ak.any(row)),index=df.index).fillna(False)      
    #save Theta_lepton as well to histogramm the distributions later
    df["Theta_muon"] = pd.Series(data=df_mu["Theta_muon"],index=df.index).fillna(False)
    df["Theta_electron"] = pd.Series(data=df_e["Theta_electron"],index=df.index).fillna(False)
    print("---cut4 applied!---")
    return df

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

#cut6: at least one muon with p>15 GeV per event or electron with p>20 GeV
def cut6(input_df):
    print("---Applying cut6: Require at least one muon(electron) with p>15(20) GeV per event---")
    df = input_df.copy()
    df["p_muon"] = np.sqrt(df["muon_px"]**2+df["muon_py"]**2+df["muon_pz"]**2)
    df["p_electron"] = np.sqrt(df["electron_px"]**2+df["electron_py"]**2+df["electron_pz"]**2)
    df["cut6_muon"] = df["p_muon"].apply(lambda row: any(val > 15 for val in row))
    df["cut6_electron"] = df["p_electron"].apply(lambda row: any(val > 20 for val in row))
    df["cut6"] = df["cut6_muon"] | df["cut6_electron"]
    print("---cut6 applied!---")
    return df

#cut7: Exactly one muon with p>15 GeV or electron with p>20 GeV per event
def cut7(input_df):
    print("---Applying cut7: Exactly one muon(electron) with p>15(20) GeV per event---")
    df = input_df.copy()
    mask = []
    df["pT_muon"] = np.sqrt(df["muon_px"]**2+df["muon_py"]**2)
    df["pT_electron"] = np.sqrt(df["electron_px"]**2+df["electron_py"]**2)
    df["cut7_muon"] = df[["n_muons","p_muon"]].apply(lambda row: all(val>15 for val in row["p_muon"]) & row["n_muons"]==1,axis=1)
    df["cut7_electron"] = df[["n_electrons","p_electron"]].apply(lambda row: all(val>20 for val in row["p_electron"]) & row["n_electrons"]==1,axis=1)
    df["cut7"] = df["cut7_muon"] | df["cut7_electron"] 
    print("---cut7 applied---")
    return df

#Apply the cuts to the df
def cut_flow(df,jet_algo):
    n_s = events(df,1) + events(df,2)
    n_b = events(df,0)
    table_s, table_b = [], []
    df = cut0(df)
    df = df[df["cut0"]]
    signal_eff_pur("cut0",df,24,n_s,n_b,table_s,table_b)
    df = cut1(df,2,jet_algo)
    df = df[df["cut1_{}".format(jet_algo)]]
    signal_eff_pur("cut1_{}".format(jet_algo),df,24,n_s,n_b,table_s,table_b)
    df = cut2(df,1,jet_algo)
    df = df[df["cut2_{}".format(jet_algo)]]
    signal_eff_pur("cut2_{}".format(jet_algo),df,24,n_s,n_b,table_s,table_b)
    df = cut3(df)
    df = df[df["cut3"]]
    signal_eff_pur("cut3",df,24,n_s,n_b,table_s,table_b)
    df = cut4(df,jet_algo)
    df = df[df["cut4"]]
    signal_eff_pur("cut4",df,24,n_s,n_b,table_s,table_b)
    print("---Using jet_{} as jet_algo---".format(jet_algo))
    print("signal efficiency and purity:")
    print(tabulate(table_s,headers="keys",tablefmt="grid"))
    print("background efficiency and purity:")
    print(tabulate(table_b,headers="keys",tablefmt="grid"))
    return df

###
# apply filters and leptons
###

def lxcosTheta(df):
    s = 365**2 #square of centre of mass energy in GeV
    m_t = 173 #top mass in GeV
    beta = np.sqrt(1-(4*m_t**2)/s) #top velocity
    c_0 = constants.speed_of_light
    #for electrons
    df_e = df[df["cut0_electron"]]
    electron_charge = np.array(ak.flatten(df_e["electron_charge"]))
    mask_electron = np.array(ak.flatten(df_e["cut4_electron"]))
    Theta_electron, electron_energy = np.array(ak.flatten(df_e["Theta_electron"])), np.array(ak.flatten(df_e["electron_energy"]))
    Theta_electron = Theta_electron[mask_electron]
    electron_energy = electron_energy[mask_electron]
    electron_charge = electron_charge[mask_electron]
    x_electron = 2*electron_energy/m_t*np.sqrt((1-beta)/(1+beta))
    Theta_eplus = Theta_electron[electron_charge == +1]
    Theta_eminus = Theta_electron[electron_charge == -1] #distinguish btw electrons and positrons in final state
    x_eplus = x_electron[electron_charge == +1]
    x_eminus = x_electron[electron_charge == -1]
    #for muons
    df_mu = df[df["cut0_muon"]]
    muon_charge = np.array(ak.flatten(df_mu["muon_charge"]))
    mask_muon = np.array(ak.flatten(df_mu["cut4_muon"]))
    Theta_muon, muon_energy = np.array(ak.flatten(df_mu["Theta_muon"])), np.array(ak.flatten(df_mu["muon_energy"]))
    Theta_muon = Theta_muon[mask_muon]
    muon_energy = muon_energy[mask_muon]
    muon_charge = muon_charge[mask_muon]
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
    row_bool = np.abs(np.array(row)) == 24
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
    df["Theta_genElectron"] = Theta(df["pT_genElectron"],df["genElectron_pz"])
    df["Theta_genMuon"] = Theta(df["pT_genMuon"],df["genMuon_pz"])          
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

