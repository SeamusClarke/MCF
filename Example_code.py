import MCF

import numpy as np
from astropy.convolution import convolve_fft, Gaussian2DKernel
import os
from numpy.random import random
from scipy.stats import circmean,circstd
import matplotlib.pyplot as plt
'''
### Make a random realisation for an angle map given a set of parameters

grid_size = 128 ### Number of pixels in one length of the map
total_disp = 20 ### Total angular dispersion in the map given in degrees
alpha = -4.0    ### Power law index for the map
k_min = 1       ### Wavenumber of the largest mode given power

seed = 1       ### Integer number given for seed

### Makes angle map given the parameters, as well as Stokes Q and U
a,q,u = MCF.make_angle_map(grid_size,total_disp,alpha,k_min,seed)



### Apply both unsharp-masking techniques to the angle map given a set filter size, and calculate data for a filter-size plot

filtersize = 5 ### Filter size in pixels for the unsharp-masking method - Here the filter will be 11x11 pixels (2*filtersize+1)
max_filtersize = 30 ### Maximum filter size will be 61x61 pixels

### Calculate the residual map from the unsharp-masking technique
res = MCF.unsharp_masking(a,filtersize)
### Calculate the local angular dispersion map using the unsharp-masking technique
disp = MCF.unsharp_masking_map(a,filtersize)

### Calculate the angular dispersion across multiple filter sizes
fs_out = MCF.filtersize_data(a,max_filtersize)



### Plot this information in two figures

### First convert everything to degrees
a = a*180/np.pi
res = res*180/np.pi
disp = disp*180/np.pi
fs_out[1,:] = fs_out[1,:]*180/np.pi

### Plot the angle map, residual angle map, and local dispersion map for the given filter size
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

i1 = axes[0].imshow(a,origin="lower",cmap="magma",vmin=-90,vmax=90)

ax = np.sin(a * np.pi/180)
ay = np.cos(a * np.pi/180)
vec_sep=5
xx,yy,ux,uy = MCF.segments(ax,ay,vec_sep)
axes[0].quiver(xx, yy, ux, uy, units='width', color="cyan", pivot='middle', scale=30, headlength=0, headwidth=1, width=0.004, alpha=0.7)

i2 = axes[1].imshow(res,origin="lower",cmap="viridis",vmin=-20,vmax=20)
i3 = axes[2].imshow(disp,origin="lower",cmap="cividis",vmin=0,vmax=40)

ims = [i1, i2, i3]
labels=["Angle [deg]","Residual angle [deg]","Local dispersion [deg]"]

for ax, im, lb in zip(axes, ims, labels):
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.02, label=lb)
    ax.axis('off')

plt.tight_layout()
plt.savefig("Test_angle_map.png",dpi=150)

### Make a filter-size plot for the angle map
fig, axes = plt.subplots(1, 1, figsize=(4, 4))

axes.plot(fs_out[0,:],fs_out[1,:],lw=2)
axes.set_xlabel("Filter size [pixels]")
axes.set_ylabel("Angular dispersion [deg]")
axes.axhline(y=total_disp,color="k",ls="--")

plt.tight_layout()
plt.savefig("Test_filtersize.png",dpi=150)



'''
### Determine the correction factor for an example beam-convolved angle map

### Load the example angle map included and calculate it's filter-size plot
ex_angle = np.load("Test_angle_map.npy")
max_filtersize = 20            ### Set maximum filter size to 41x41 pixels
fs_out = MCF.filtersize_data(ex_angle,max_filtersize)

### Set parameters for this search
grid_size = ex_angle.shape[0]  ### Take grid size as the example map size
total_disp = 20				   ### Set total dispersion 
alpha = -2.0                   ### Set power law index 
k_min = 1.0                    ### Set minimum wavenumber

bm_size = 6.0 				   ### Set beam size (FWHM) to 6 pixels
iter_count = 10                ### Calculate 20 different realisations

fs_search = MCF.filtersize_param(grid_size,total_disp,alpha,k_min,bm_size,iter_count,max_filtersize)

### Double check if something is wrong with any of the realisations
q = np.zeros(fs_search.shape[0],dtype=bool)
for ii in range(0,fs_search.shape[0]):
	if np.amin(fs_search[ii,2,:])>0:
		q[ii]=True
	else:
		q[ii]=False

### Plot the results
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

### Plot example map's filter-size plot
axes[0].plot(fs_out[0,:]/bm_size,fs_out[1,:]*180/np.pi,color="k",lw=2)
### Plot the random realisations to compare to and their average
axes[0].plot(fs_search[0,0,:]/bm_size,fs_search[q,2,:].T*180/np.pi,color="C0",alpha=0.2)
mean_disp = np.mean(fs_search[q,2,:],axis=0)
std_disp = np.std(fs_search[q,2,:],axis=0)
axes[0].errorbar(fs_search[0,0,:]/bm_size,mean_disp*180/np.pi,yerr=std_disp*180/np.pi,color="C1",lw=2)

axes[0].set_xlabel("Filter size [beams]")
axes[0].set_ylabel("Angular dispersion [deg]")

### Plot the correction factor for the random realisations
corr_fact = fs_search[q,1,:].T/fs_search[q,2,:].T
axes[1].plot(fs_search[0,0,:]/bm_size,corr_fact,color="C0",alpha=0.2)
mean_corrfact = np.mean(corr_fact,axis=1)
std_corrfact = np.std(corr_fact,axis=1) 
axes[1].errorbar(fs_search[0,0,:]/bm_size,mean_corrfact,yerr=std_corrfact,color="C1",lw=2)

axes[1].set_xlabel("Filter size [beams]")
axes[1].set_ylabel("Correction factor")
axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()

plt.tight_layout()
plt.savefig("Correction_factor.png",dpi=150)

### For a given filter size find the corrector factor and its uncertainty from the random realisations

goal_filtersize = 3.2 ### Find the correction factor for a filter size equal to 3.2 beams
cf,e_cf = MCF.Calc_correction_factor(fs_search,bm_size,goal_filtersize)
print("For a %.2f beam size filter the correction factor = %.2f +/- %.2f" %(goal_filtersize,cf,e_cf))
