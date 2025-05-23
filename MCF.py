import numpy as np
from astropy.convolution import convolve_fft, Gaussian2DKernel
import os
from numpy.random import random
from scipy.stats import circmean,circstd



### Function to return data for a filter-size plot give an angle map (in radians)
def filtersize_data(angle_map,max_filtersize):
	### Set up filter size array
	filsize = np.arange(1,max_filtersize+1,2,dtype=int)
	### Set up array to store the filter-size plot data 
	fs_data = np.zeros_like(filsize,dtype=float)
		
	### Loop over filter sizes
	for kk in range(0,len(filsize)):
		### Apply unsharp masking method using the current filter size to produce residual angle map
		res_map = unsharp_masking(angle_map,filsize[kk])
		### Remove empty array entries (if any) to avoid biases
		res_map = res_map[res_map!=0]
		### Calculate angular dispersion of the residual angle map and store it
		fs_data[kk]=circstd(res_map,low=-np.pi/2,high=np.pi/2)

	### Package output into a single array
	output = np.column_stack([2*filsize + 1,fs_data]).T

	return output


### Function to return data for filter-size plot given properties of a parametrised field, both with and without a beam
def filtersize_param(grid_size,total_disp,alpha,k_min,bm_size,iter_count,max_filtersize):

	### Set up filter size array
	filsize = np.arange(1,max_filtersize+1,2,dtype=int)
	### Set up kernel with the correct beam size
	kernel = Gaussian2DKernel(bm_size/np.sqrt(8*np.log(2)))
	### Set up output array to store the information of each random realisation
	output = np.zeros((iter_count,3,len(filsize)))

	### Loop over all random realisations using a different random seed
	for sd in range(1,iter_count+1):
		print("Iteration number = ",sd)

		### Make a random realisation of the angle map given the parameters
		a,q,u = make_angle_map(grid_size,total_disp,alpha,k_min,sd)

		### Set up array to store the filter-size plot data for this realisation
		fs_data = np.zeros_like(filsize,dtype=float)
		### Loop over filter sizes
		for kk in range(0,len(filsize)):
			### Apply unsharp masking method using the current filter size to produce residual angle map
			res_map = unsharp_masking(a,filsize[kk])
			### Remove empty array entries (if any) to avoid biases
			res_map = res_map[res_map!=0]
			### Calculate angular dispersion of the residual angle map and store it
			fs_data[kk]=circstd(res_map,low=-np.pi/2,high=np.pi/2)

		### Store filter size information in output array
		output[sd-1,0,:] = np.copy(2*filsize + 1)
		### Store filter size data for the map without beam convolution
		output[sd-1,1,:]=np.copy(fs_data)

		### Convolve the Q and U maps with the beam
		q = convolve_fft(q,kernel)
		u = convolve_fft(u,kernel)

		### Calculate the angle map from the beam convolved Q and U maps and set the mean to 0 degrees
		a = get_angle(q,u)
		a = a - circmean(a,low=-np.pi/2,high=np.pi/2)
		aa = a + np.pi/2
		aa = aa%np.pi
		a = aa - np.pi/2

		### Set up array to store the filter-size plot data for this realisation
		fs_data = np.zeros_like(filsize,dtype=float)
		### Loop over filter sizes
		for kk in range(0,len(filsize)):
			### Apply unsharp masking method using the current filter size to produce residual angle map
			res_map = unsharp_masking(a,filsize[kk])
			### Remove empty array entries (if any) to avoid biases
			res_map = res_map[res_map!=0]
			### Calculate angular dispersion of the residual angle map and store it
			fs_data[kk]=circstd(res_map,low=-np.pi/2,high=np.pi/2)

		### Store filter size data for the map with beam convolution
		output[sd-1,2,:] = np.copy(fs_data)

	return output


### Function to calculate the correction factor at a given filter size using the data from filtersize_param
def Calc_correction_factor(fs_search,bm_size,goal):

	if(goal<np.amin(fs_search[0,0,:]/bm_size)):
		print("Error! Given filter size (%.2f beams) is smaller than the smallest possible filter size for this data (%.2f beams)" %(goal,np.amin(fs_search[0,0,:]/bm_size)))
		return 0,0
	if(goal>np.amax(fs_search[0,0,:]/bm_size)):
		print("Error! Given filter size (%.2f beams) is bigger than the maximum filter size calculated (%.2f beams)" %(goal,np.amax(fs_search[0,0,:]/bm_size)))
		return 0,0

	q = np.zeros(fs_search.shape[0],dtype=bool)
	for ii in range(0,fs_search.shape[0]):
		if np.amin(fs_search[ii,2,:])>0:
			q[ii]=True
		else:
			q[ii]=False

	dx = (fs_search[0,0,1]-fs_search[0,0,0])/bm_size
	index = int((goal-fs_search[0,0,0]/bm_size)/dx)
	temp_a = np.zeros_like(fs_search[:,0,0],dtype=float)

	for ii in range(0,len(temp_a)):
		y0 = fs_search[ii,1,index]/fs_search[ii,2,index]
		dy = fs_search[ii,1,index+1]/fs_search[ii,2,index+1] - y0

		temp_a[ii] = y0 + dy/dx * (goal-fs_search[0,0,index]/bm_size)

	mean_corr = np.mean(temp_a[q])
	std_corr = np.std(temp_a[q])

	return mean_corr,std_corr


### Function to produce an angle map from a power-law model
### Returns angle map as well as Stokes Q and U
def make_angle_map(grid_size,total_disp,alpha,k_min,seed):

	### Make the Q and U maps with set parameters
	q = make_turb(alpha/2.,k_min,seed,grid_size)
	u = make_turb(alpha/2.,k_min,seed*1000,grid_size)

	### Find the normalisation constant required to obtain the required total dispersion in the map
	norm_const = find_norm_const(find_norm,(q,u,total_disp))
	u = u+np.fabs(np.amin(u))-norm_const

	### Calculate the angle map from the generated Q and U maps and set the mean to 0 degrees
	a = get_angle(q,u)
	a = a - circmean(a,low=-np.pi/2,high=np.pi/2)
	aa = a + np.pi/2
	aa = aa%np.pi
	a = aa - np.pi/2
	
	### Report if the normalisation has gone wrong and has not resulted in the correct total dispersion
	map_disp = circstd(a,low=-np.pi/2,high=np.pi/2) * 180/np.pi
	if(np.fabs(map_disp-total_disp)>0.1):
		print("Error! Can not produce a map with the requested angular dispersion.")

	return a,q,u

### Function to perform unsharp-masking on an angle map for a given filter size.
### Returns 2D array of the same shape as the angle map containing the residual angle at each pixel
def unsharp_masking(angle_map,filtersize):

	### Create output array of the same shape as the angle map
	out_a = np.zeros_like(angle_map)

	### Get the shape of angle map array
	nx = angle_map.shape[0]
	ny = angle_map.shape[1]

	### Loop over the angle map array
	for ii in range(0,nx):
		for jj in range(0,ny):

			### Skip pixel if it is NaN
			if(np.isnan(angle_map[ii,jj])):
				continue

			### Find array element range for pixels inside the filter
			minx = ii-filtersize
			maxx = ii+filtersize+1 
			miny = jj-filtersize
			maxy = jj+filtersize+1

			if(minx<0):
				minx = 0
			if(maxx>nx):
				maxx = nx+1
			if(miny<0):
				miny = 0 
			if(maxy>ny):
				maxy = ny+1 

			### Extract pixels inside the filter and exclude any NaNs
			temp_a = angle_map[minx:maxx,miny:maxy].flatten()
			temp_a = temp_a[np.isnan(temp_a)==False]

			### Skip pixels which only have NaN inside their filters
			if(len(temp_a)==0):
				continue

			### Remove the mean angle inside the filter from the central pixel and store it in the output array
			out_a[ii,jj] = angle_map[ii,jj] - circmean(temp_a,low=-np.pi/2,high=np.pi/2)

			### Handle cases when angles go beyond +/- 90 degrees
			if(out_a[ii,jj]<-np.pi/2):
				out_a[ii,jj] = np.pi + out_a[ii,jj]
			if(out_a[ii,jj]>np.pi/2):
				out_a[ii,jj] = -np.pi + out_a[ii,jj]

	return out_a

### Function to perform unsharp-masking on an angle map for a given filter size.
### Returns 2D array of the same shape as the angle map containing the residual angle at each pixel
def unsharp_masking(angle_map,filtersize):

	### Create output array of the same shape as the angle map
	out_a = np.zeros_like(angle_map)

	### Get the shape of angle map array
	nx = angle_map.shape[0]
	ny = angle_map.shape[1]

	### Loop over the angle map array
	for ii in range(0,nx):
		for jj in range(0,ny):

			### Skip pixel if it is NaN
			if(np.isnan(angle_map[ii,jj])):
				continue

			### Find array element range for pixels inside the filter
			minx = ii-filtersize
			maxx = ii+filtersize+1 
			miny = jj-filtersize
			maxy = jj+filtersize+1

			if(minx<0):
				minx = 0
			if(maxx>nx):
				maxx = nx+1
			if(miny<0):
				miny = 0 
			if(maxy>ny):
				maxy = ny+1 

			### Extract pixels inside the filter and exclude any NaNs
			temp_a = angle_map[minx:maxx,miny:maxy].flatten()
			temp_a = temp_a[np.isnan(temp_a)==False]

			### Skip pixels which only have NaN inside their filters
			if(len(temp_a)==0):
				continue

			### Remove the mean angle inside the filter from the central pixel and store it in the output array
			out_a[ii,jj] = angle_map[ii,jj] - circmean(temp_a,low=-np.pi/2,high=np.pi/2)

			### Handle cases when angles go beyond +/- 90 degrees
			if(out_a[ii,jj]<-np.pi/2):
				out_a[ii,jj] = np.pi + out_a[ii,jj]
			if(out_a[ii,jj]>np.pi/2):
				out_a[ii,jj] = -np.pi + out_a[ii,jj]

	return out_a

### Function to perform unsharp-masking on an angle map for a given filter size
### Returns a map showing the local dispersion within that filter size
def unsharp_masking_map(angle_map,filtersize):

	### Create output array of the same shape as the angle map
	out_a = np.zeros_like(angle_map)

	### Get the shape of angle map array
	nx = angle_map.shape[0]
	ny = angle_map.shape[1]

	### Loop over the angle map array
	for ii in range(0,nx):
		for jj in range(0,ny):

			### Skip pixel if it is NaN
			if(np.isnan(angle_map[ii,jj])):
				continue

			### Find array element range for pixels inside the filter
			minx = ii-filtersize
			maxx = ii+filtersize+1 
			miny = jj-filtersize
			maxy = jj+filtersize+1

			if(minx<0):
				minx = 0
			if(maxx>nx):
				maxx = nx+1
			if(miny<0):
				miny = 0 
			if(maxy>ny):
				maxy = ny+1 

			### Extract pixels inside the filter and exclude any NaNs
			temp_a = angle_map[minx:maxx,miny:maxy].flatten()
			temp_a = temp_a[np.isnan(temp_a)==False]

			### Skip pixels which only have NaN inside their filters
			if(len(temp_a)==0):
				continue

			### Remove the mean angle inside the filter from the central pixel and store it in the output array
			out_a[ii,jj] = circstd(temp_a,low=-np.pi/2,high=np.pi/2)

	return out_a












### Internal functions which are not meant to be called
### As such they are left uncommented

### Internal function to make a turbulent 2D field
def make_turb(law,kmin,sd,grid):

	np.random.seed(sd)

	A = np.zeros((grid,grid))
	phase = np.zeros((grid,grid))

	half_modes = int((grid-1)/2 + 1)

	for ii in range(0,half_modes):
		ky = ii
		
		if(ky==0):
			for jj in range(0,half_modes):
				kx = jj
				k = np.sqrt(kx**2 + ky**2)
				if(kx==0 or k<kmin):
					A[ii,jj] = 0
				else:
					A[ii,jj] = np.random.normal()*k**law

				phase[ii,jj] = 2*np.pi*random()

		if(ky>0):
			for jj in range(0,grid):
				kx = jj

				if(jj > (grid-1)/2):
					kx = jj - grid

				k = np.sqrt(kx**2 + ky**2)

				if(k<kmin):
					A[ii,jj] = 0
				else:
					A[ii,jj] = np.random.normal()*k**law

				phase[ii,jj] = 2*np.pi*random()


	B1 = A*np.cos(phase)
	B2 = 1j*A*np.sin(phase)

	v = np.fft.ifftn(B1 + B2)
	v = v.real

	v = v/np.std(v)

	return v

### Internal function to convert Stokes Q and U into angles
def get_angle(q,u):
	return 0.5 * np.arctan2(-u,q) 

### Internal function for the bisector method to find roots
def bisector_method(func,f_param,upper,lower):

	dx = 0.5*(upper-lower)
	mid = lower+dx

	upper_ans = func(upper,*f_param)
	mid_ans = func(mid,*f_param)
	lower_ans = func(lower,*f_param)

	if(abs(mid_ans) < 0.01):
		return mid

	else:

		if(upper_ans/mid_ans < 0 and lower_ans/mid_ans > 0):

			answer = bisector_method(func,f_param,upper,mid)
			return answer

		elif(lower_ans/mid_ans < 0 and upper_ans/mid_ans>0):

			answer = bisector_method(func,f_param,mid,lower)
			return answer

		else:

			return 0


### Internal function to find the constant to add to Stokes U to produce the desired angular dispersion for a given set of angle map parameters
def find_norm_const(func,param):

	q = param[0]
	u = param[1]
	aim = param[2]

	n = 100
	k = np.arange(1, n + 1)
	xx = np.cos((2*k - 1) * np.pi / (2*n))
	xx=-xx*13 - 6
	mn = np.amax(xx[xx<0])
	xx=xx[xx>mn-0.01]

	ii=0

	while(xx[ii]<6):

		diff = func(xx[ii],q,u,aim)

		if(diff>0):
			break

		ii=ii+1

	if(ii==0 and diff>0):
		ans = bisector_method(func,param,0,-100.)
	else:
		ans = bisector_method(func,param,xx[ii],xx[ii-1])

	return ans

### Internal function to find the constant to add to Stokes U to produce the desired angular dispersion for a given set of angle map parameters
def find_norm(x,q,u,aim):
	u = u+np.fabs(np.amin(u))-x

	a = get_angle(q,u)
	scale = circstd(a,low=-np.pi/2,high=np.pi/2) * 180/np.pi

	diff = scale-aim

	return diff

### Internal function to produce arrays to plot segments using the plt.quiver function
def segments(x, y, sep_range):

   nx=np.shape(x)[0]
   ny=np.shape(x)[1]

   l=np.sqrt(x**2+y**2)
   l[l==0] = 1.0

   xx=x/l
   yy=y/l

   X, Y = np.meshgrid(np.arange(0, ny-1, sep_range), np.arange(0, nx-1, sep_range))
   xx0=xx[Y,X]
   uy0=yy[Y,X]
   
   return X, Y, xx[Y,X], yy[Y,X]


