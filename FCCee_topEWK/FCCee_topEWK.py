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
from cut_flow_functions import cut_flow
'''
This code analyses the (x,cos(Theta)) distribution of muons originating from semileptonic decay of the t quark.
Event selection cuts are applied to the data try to minimise background events e.g.from the semileptonic decay of B-mesons producing "fake leptons" to the semileptonic signal
'''

#import dataframes
#path_tlepTlep = "/home/skeilbach/FCCee_topEWK/ee_Z_tt_tlepTlep_pol.pkl"
#path_thadThad = "/home/skeilbach/FCCee_topEWK/ee_SM_tt_thadThad.pkl"
#path_thadTlep = "/home/skeilbach/FCCee_topEWK/ee_SM_tt_thadTlep.pkl"
#path_tlepThad = "/home/skeilbach/FCCee_topEWK/ee_SM_tt_tlepThad.pkl"

#df_tlepTlep = pd.read_pickle(path_tlepTlep).sample(n=100000,ignore_index=True)
#df_thadThad = pd.read_pickle(path_thadThad)
#df_thadTlep = pd.read_pickle(path_thadTlep)
#df_tlepThad = pd.read_pickle(path_tlepThad)

#concatenate signal(tlepThad,thadTlep) and background (thadThad,tlepTlep) events and shuffle the dataframe
#df = pd.concat([df_thadThad,df_tlepThad,df_thadTlep]).sample(frac=1).reset_index(drop=True)
df = pd.read_pickle("/home/skeilbach/FCCee_topEWK/ee_SM_tt_300k.pkl")

jet_algo = "ee_genkt04"
df = cut_flow(df,jet_algo)
'''
###
#Plotting the genLepton and lepton distributions side by side
###

#creating the plots
hist, xedges, yedges = np.histogram2d(x_lplus,np.cos(Theta_lplus), bins=(25,25))
xpos, ypos = np.meshgrid(yedges[:-1]+yedges[1:], xedges[:-1]+xedges[1:]) #use this specific slicing to ensure the x and y position of the bars in the plot is almost in the middle of the chosen bins
xpos = xpos.flatten()/2.
ypos = ypos.flatten()/2.
zpos = np.zeros_like (xpos)

dx = xedges [1] - xedges [0]
dy = yedges [1] - yedges [0]
dz = hist.flatten()

fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
#ax1.plot_wireframe(xpos,ypos,hist, rstride=3, cstride=3)
ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
#ax1.plot_surface(xpos, ypos,hist, rstride=1, cstride=1, color= "blue")
ax1.set_title(r"$(x,cos(\Theta))$ for $l^+$")
ax1.set_xlabel(r"$cos(\Theta)$")
ax1.set_ylabel(r"$x$")
ax1.set_xticks([-1,0,1])
ax1.set_yticks([0.2,0.4,0.6,0.8,1.])
ax1.set_zlabel("frequency")
ax1.view_init(25,-35)

#now plotting the distribution for genLeptons
hist, xedges, yedges = np.histogram2d(x_genLplus,np.cos(Theta_genLplus), bins=(25,25))
xpos, ypos = np.meshgrid(yedges[:-1]+yedges[1:], xedges[:-1]+xedges[1:]) #use this specific slicing to ensure the x and y position of the bars in the plot is almost in the middle of the chosen bins
xpos = xpos.flatten()/2.
ypos = ypos.flatten()/2.
zpos = np.zeros_like (xpos)

dx = xedges [1] - xedges [0]
dy = yedges [1] - yedges [0]
dz = hist.flatten()

ax2 = fig.add_subplot(122, projection='3d')
ax2.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
ax2.set_title(r"$(x,cos(\Theta)$ for $L^+$")
ax2.set_xlabel(r"$cos(\Theta)$")
ax2.set_ylabel(r"$x$")
ax2.set_xticks([-1,0,1])
ax2.set_yticks([0.2,0.4,0.6,0.8,1.])
ax2.set_zlabel("frequency")
ax2.view_init(25,-35)

plt.suptitle(r"$(x,\mathrm{cos}(\Theta))$ for positive reconstructed leptons l and genLeptons L")
plt.savefig("/home/skeilbach/FCCee_topEWK/FCCee_xcosTheta_plus.png",dpi=300)
plt.close()

#now for the leptons originating from a W-
hist, xedges, yedges = np.histogram2d(x_lminus,np.cos(Theta_lminus), bins=(25,25))
xpos, ypos = np.meshgrid(yedges[:-1]+yedges[1:], xedges[:-1]+xedges[1:]) #use this specific slicing to ensure the x and y position of the bars in the plot is almost in the middle of the chosen bins
xpos = xpos.flatten()/2.
ypos = ypos.flatten()/2.
zpos = np.zeros_like (xpos)

dx = xedges [1] - xedges [0]
dy = yedges [1] - yedges [0]
dz = hist.flatten()

fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
#ax1.plot_wireframe(xpos,ypos,hist, rstride=3, cstride=3)
ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
#ax1.plot_surface(xpos, ypos,hist, rstride=1, cstride=1, color= "blue")
ax1.set_title(r"$(x,cos(\Theta))$ for $l^-$")
ax1.set_xlabel(r"$cos(\Theta)$")
ax1.set_ylabel(r"$x$")
ax1.set_xticks([-1,0,1])
ax1.set_yticks([0.2,0.4,0.6,0.8,1.])
ax1.set_zlabel("frequency")
ax1.view_init(25,-35)

#now plotting the distribution for the genLeptons
hist, xedges, yedges = np.histogram2d(x_genLminus,np.cos(Theta_genLminus), bins=(25,25))
xpos, ypos = np.meshgrid(yedges[:-1]+yedges[1:], xedges[:-1]+xedges[1:]) #use this specific slicing to ensure the x and y position of the bars in the plot is almost in the middle of the chosen bins
xpos = xpos.flatten()/2.
ypos = ypos.flatten()/2.
zpos = np.zeros_like (xpos)

dx = xedges [1] - xedges [0]
dy = yedges [1] - yedges [0]
dz = hist.flatten()

ax2 = fig.add_subplot(122, projection='3d')
ax2.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
ax2.set_title(r"$(x,cos(\Theta))$ for $L^-$")
ax2.set_xlabel(r"$cos(\Theta)$")
ax2.set_ylabel(r"$x$")
ax2.set_xticks([-1,0,1])
ax2.set_yticks([0.2,0.4,0.6,0.8,1.])
ax2.set_zlabel("frequency")
ax2.view_init(25,-35)

plt.suptitle(r"$(x,\mathrm{cos}(\Theta))$ for negative reconstructed leptons l and genLeptons L")
plt.savefig("/home/skeilbach/FCCee_topEWK/FCCee_xcosTheta_minus.png",dpi=300)
plt.close()

###
#Chi square fit
###

#Define chi2 function with the fit parameter k so that for k=0 the experimental data represents the SM couplings 
def chi2(k):
    return np.sum((x_i-(x_SM+k*s_i))**2/x_i_err**2)

#compare data for cos(Theta) and x projections
x_i, bins = np.histogram(x_lminus, bins=25)
plt.plot(bins[:-1], x_i, drawstyle='steps',label=r"$x_i$",alpha=0.5)
plt.plot(bins[:-1], x_i+np.abs(np.random.randn(25)), drawstyle='steps',label=r"$s_i$",alpha=0.5)
plt.xlabel(r"$x_{l^-}$")
plt.ylabel("frequency")
plt.title("MC and experimental data in x-projection")
plt.legend() 
plt.savefig("/home/skeilbach/FCCee_topEWK/xprojection.png",dpi=300)
plt.close()


x_i, bins = np.histogram(np.cos(Theta_lminus), bins=25)
plt.plot(bins[:-1], x_i, drawstyle='steps',label=r"$x_i$",alpha=0.5)
plt.plot(bins[:-1], x_i+np.abs(np.random.randn(25)), drawstyle='steps',label=r"$s_i$",alpha=0.5)
plt.xlabel(r"cos($\Theta$)")
plt.ylabel("frequency")
plt.title(r"MC and experimental data in cos($\Theta$)-projection")
plt.legend() 
plt.savefig("/home/skeilbach/FCCee_topEWK/cosThetaprojection.png",dpi=300)
plt.close()


#TEST
gaussian_noise = np.abs(np.random.randn(25))
s_i, bins_s = np.histogram(x_lminus,bins=25)
x_SM = s_i
x_i = s_i + gaussian_noise #produce "experimental data" by adding gaussian noise to the binned data
x_i_err = np.sqrt(x_i)
#minimise chi2 using the iminuit package
m = Minuit(chi2,k=1)
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
ax.set_xlabel(r"k")
ax.set_ylabel(r"$\Delta \chi^2$")
plt.title(r"k={:.4f}$\pm${:.4f}".format(k_min,k_std))
plt.savefig("/home/skeilbach/FCCee_topEWK/Delta_chi2_x.png",dpi=300)
plt.close()

s_i, bins_s = np.histogram(np.cos(Theta_lminus),bins=25)
x_SM = s_i
x_i = s_i + gaussian_noise #produce "experimental data" by adding gaussian noise to the binned data
x_i_err = np.sqrt(x_i)
#minimise chi2 using the iminuit package
m = Minuit(chi2,k=1)
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
ax.set_xlabel(r"k")
ax.set_ylabel(r"$\Delta \chi^2$")
plt.title(r"k={:.4f}$\pm${:.4f}".format(k_min,k_std))
plt.savefig("/home/skeilbach/FCCee_topEWK/Delta_chi2_cosTheta.png",dpi=300)
plt.close()



#Define data
gaussian_noise = np.abs(np.random.randn(25,25))
s_i, sample_x_edges, sample_y_edges = np.histogram2d(x_lminus,np.cos(Theta_lminus),bins=(25,25))
x_SM, _,_ = np.histogram2d(x_lminus,np.cos(Theta_lminus),bins=(sample_x_edges,sample_y_edges))
x_i = s_i + gaussian_noise #produce "experimental data" by adding gaussian noise to the binned data
x_i_err = np.sqrt(x_i)
x_i,x_SM,s_i,x_i_err = x_i.flatten(),x_SM.flatten(),s_i.flatten(),x_i_err.flatten()

#minimise chi2 using the iminuit package
m = Minuit(chi2,k=1)
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
ax.set_xlabel(r"k")
ax.set_ylabel(r"$\Delta \chi^2$")
plt.title(r"k={:.4f}$\pm${:.4f}".format(k_min,k_std))
plt.savefig("/home/skeilbach/FCCee_topEWK/Delta_chi2.png",dpi=300)
plt.close()
'''


