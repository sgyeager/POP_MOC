import xarray as xr               #netcdf multidim reading/writing/manipulation
import glob                       #globbing
import matplotlib.pyplot as plt   #plotting, 95% of the stuff you need 
import numpy as np                #numerics
import gsw  	                  #ocean equation of state
import os  	                  #operating system commands
import time as timer
#import cftime                     #netcdf time
#import matplotlib                 #plotting, the entire module, needed for following line
#matplotlib.use('Agg')             #backend necessary to save/plot from casper queue without crashing, optional for interactive session

time1=timer.time()
debug=True

# First, use f2py for MOCoffline fortran routines
f90mocroutines='/glade/u/home/yeager/POP_tools/computeMOCoffline/lib/MOCoffline.POP_1deg.f90'
if not os.path.isfile('moc_offline_1deg.cpython-37m-x86_64-linux-gnu.so'):  #check if compiled source code already exists
   print('MOCoffline compiling')
   os.system('f2py -c '+f90mocroutines+' -m moc_offline_1deg')
else: print('moc_offline already compiled')
import moc_offline_1deg

#Pseudocode:
#==============
#1. Define input/output streams
#	- single POP netcdf output file  OR collection of concatenated time series files
#	- template file that will provide template for MOC array
#	- netcdf file with time-dimension >= 1 
#2. Define target coordinates
#	- sigma dimension
#	- latitude dimension
#3. Read in needed fields
#	- 3D velocity (eulerian, at least)
#	- Grid info
#4. Compute PD referenced to 2000m
#	- regrid from T-grid to U-grid
#	- check for & fix any inversions in sigma_2 array
#5. Find sigma layer depths on U-grid
#	- check that full ocean volume is accounted for
#6. Calculate volume fluxes binned in sigma-coordinates
#	- compute flow convergence in sigma to get w_sigma
#	- check that vertically-integrated volume fluxes in
#		z-space and sigma-space give same answers
#7. Compute MOC in sigma-space
#	- integrate w_sigma in zonal direction
#	- partially-integrate w_sigma in meridional direction
#	- add MOC boundary condition at Atlantic southern boundary
#8. Write output
#

#1. Define input/output streams
in_dir='/glade/scratch/yeager/g.DPLE.GECOIAF.T62_g16.009.chey/'
out_dir=in_dir
in_file = in_dir+'g.DPLE.GECOIAF.T62_g16.009.chey.pop.h.0301-01.nc'
out_file = out_dir+'g.DPLE.GECOIAF.T62_g16.009.chey.pop.h.0301-01.MOCsig2.python.nc'

#2. Define target coordinates
#2a. 	Define sigma2 coordinates
tmp1 = np.linspace(28,34.8,35)
tmp2 = np.linspace(35,35.9,10)
#tmp3 = np.linspace(36,37.45,30)
tmp3 = np.linspace(36,38.0,41)
sig2 = np.concatenate((tmp1,tmp2,tmp3))
nsig = len(sig2)
sig2_top = sig2.copy()
tmp = 0.5*(sig2[1:]-sig2[0:-1])
sig2_top[0] = sig2[0]-tmp[0]
sig2_top[1:] = sig2[0:-1]+tmp
sig2_bot = sig2.copy()
sig2_bot[-1] = sig2[-1]+tmp[-1]
sig2_bot[0:-1] = sig2[0:-1]+tmp
#print(sig2)
#print(sig2_top)
#print(sig2_bot)

#2b. 	Define MOC template
ds = xr.open_dataset(in_file)
moc = ds['MOC'].isel(moc_comp=0)
time = moc['time'] 
lat_aux_grid = moc['lat_aux_grid'] 
transport_regions = moc['transport_regions'] 
ntr = transport_regions.shape[0]
nyaux = lat_aux_grid.shape[0]

#3. Read needed fields
pd     = ds['PD']
pd=pd.drop(['ULAT','ULONG'])		# this is a python bug that we are correcting
temp   = ds['TEMP']
temp=temp.drop(['ULAT','ULONG'])
salt   = ds['SALT']
salt=salt.drop(['ULAT','ULONG'])
u_e   = ds['UVEL']/100  
u_e.attrs['units']='m/s'
v_e   = ds['VVEL']/100   
v_e.attrs['units']='m/s'
w_e   = ds['WVEL']/100  
w_e.attrs['units']='m/s'
ulat   = ds['ULAT']
ulon   = ds['ULONG']
tlat   = ds['TLAT']
tlon   = ds['TLONG']
angle  = ds['ANGLE']
kmt  = ds['KMT']
kmu  = ds['KMU']
dxu    = ds['DXU']/100
dxu.attrs['units']='m'
dyu    = ds['DYU']/100
dyu.attrs['units']='m'
rmask  = ds['REGION_MASK']
tarea  = ds['TAREA']/100/100
tarea.attrs['units']='m^2'
uarea  = ds['UAREA']/100/100
uarea.attrs['units']='m^2'
time   = ds['time']
z_t   = ds['z_t']/100
z_t.attrs['units']='m'
z_w   = ds['z_w']/100
z_w.attrs['units']='m'
dz   = ds['dz']/100
dz.attrs['units']='m'
dzw   = ds['dzw']/100
dzw.attrs['units']='m'
hu   = ds['HU']/100
hu.attrs['units']='m'
ht   = ds['HT']/100
ht.attrs['units']='m'

dims = np.shape(temp)
nt = dims[0]
nz = dims[1]
ny = dims[2]
nx = dims[3]
km = int(np.max(kmt).values)

# Create a k-index array for masking purposes
kji = np.indices((nz,ny,nx))
kindices = kji[0,:,:,:] + 1

z_bot=z_w.values
z_bot=z_w.values+dz.values
z_top=z_w.values

#4. Compute PD referenced to 2000m
# for now, use gsw state function, later use POP mwjf:
#see http://www.teos-10.org/pubs/gsw/html/gsw_contents.html

# first, convert model depth to pressure
#    (gsw function require numpy arrays, not xarray, so use ".values") 
#    (use [:,None,None] syntax to conform the z_t and tlat into 3D arrays)
press = gsw.p_from_z(-z_t.values[:,None,None],tlat.values[None,:,:])
# compute absolute salinity from practical salinity
SA = gsw.SA_from_SP(salt,press[None,:,:,:],tlon.values[None,None,:,:],tlat.values[None,None,:,:])
# compute conservative temperature from potential temperature
CT = gsw.CT_from_pt(SA,temp.values)
# compute sigma2
sigma2 = gsw.sigma2(SA,CT)

# convert to DataArray
sigma2 = xr.DataArray(sigma2,name='Sigma2',dims=pd.dims,coords=pd.coords)
sigma2.attrs=pd.attrs
sigma2.attrs['long_name']='Sigma referenced to 2000m'
sigma2.attrs['units']='kg/m^3'

#4a. Regrid from T-grid to U-grid
tmp=sigma2
tmpe=tmp.roll(nlon=-1,roll_coords=False)        # wraparound shift to west, without changing coords
tmpn=tmp.shift(nlat=-1)                         # shift to south, without changing coords
tmpne=tmpn.roll(nlon=-1,roll_coords=False)      # wraparound shift to west, without changing coords
tmpall=xr.concat([tmp,tmpe,tmpn,tmpne],dim='dummy')
sigma2=tmpall.mean('dummy')
sigma2.attrs=tmp.attrs                          # sigma2 now on U-grid
del tmp,tmpe,tmpn,tmpne,tmpall

# apply U-grid mask
#sigma2 = sigma2.where(kindices<=kmu.values[None,:,:])
mask=kindices>kmu.values[None,:,:]
sigma2.values[mask[None,:,:,:]]=np.nan
#sigma2_masked = sigma2.where(kindices<=kmu.values[None,:,:])
#debug_ds=sigma2.to_dataset(name='SIG2')
#debug_ds['SIG2_masked']=sigma2_masked
#debug_ds.to_netcdf('/glade/scratch/yeager/python_debug.nc')

#4b. Remove density inversions
for iz in range(km-1,0,-1):
     print(iz)
     tmp1 = sigma2[:,iz,:,:].values
     tmp2 = sigma2[:,iz-1,:,:].values
     tmp3 = (~np.isnan(tmp1)) & (~np.isnan(tmp2)) & np.greater(tmp2,tmp1) 
     tmp2[tmp3] = tmp1[tmp3]-1.e-5
     sigma2.values[:,iz-1,:,:] = tmp2

#5. Find sigma2 layer interface depths on U-grid
mval=pd.encoding['_FillValue']
sigma2.values[np.isnan(sigma2.values)]=mval
tmpsig=np.transpose(sigma2.values,axes=[3,2,1,0])
tmpkmu=np.transpose(kmu.values,axes=[1,0])
zsigu_top,zsigu_bot = moc_offline_1deg.sig2z(tmpsig,tmpkmu,z_t.values,z_bot,sig2_top,sig2_bot,mval,[nt,nz,ny,nx,nsig])
del tmpsig,tmpkmu

# DEBUG: write sigma2 interface layer depth info to netcdf
if debug:
   tmparr1=xr.DataArray(np.transpose(zsigu_top,axes=[3,2,1,0]),dims=['time','sigma','nlat','nlon'], \
      coords={'time':time,'sigma':sig2,'TLAT':(('nlat','nlon'),tlat),'TLONG':(('nlat','nlon'),tlon)}, \
      name='SIG_top')
   tmparr2=xr.DataArray(np.transpose(zsigu_bot,axes=[3,2,1,0]),dims=['time','sigma','nlat','nlon'], \
      coords={'time':time,'sigma':sig2,'TLAT':(('nlat','nlon'),tlat),'TLONG':(('nlat','nlon'),tlon)}, \
      name='SIG_bot')
   tmparr3=tmparr2-tmparr1
   tmparr4=tmparr1.min(dim='sigma')
   tmparr5=tmparr2.where(tmparr2<1.e30).max(dim='sigma',skipna=True)
   tmparr6=tmparr5-hu
   debug_ds=tmparr3.to_dataset(name='SIG_zdiff')
   debug_ds['SIG_top']=tmparr1
   debug_ds['SIG_bot']=tmparr2
   debug_ds['SIG_top_min']=tmparr4
   debug_ds['SIG_bot_max']=tmparr5
   debug_ds['HU']=hu
   debug_ds['HUdiff']=tmparr6
   debug_ds['SIG2']=sigma2

#6. Calculate volume fluxes binned in sigma-coordinates
#6 a. Calculate volume fluxes: UDYDZ, VDXDZ, dWDXDY (m^3/s) for each layer in z-coord
uedydz = u_e*dyu*dz	# m^3/s
vedxdz = v_e*dxu*dz	# m^3/s
wedxdy = w_e*tarea	# m^3/s
tmpkmt=np.transpose(kmt.values,axes=[1,0])
tmpu=np.transpose(uedydz.values,axes=[3,2,1,0])
tmpv=np.transpose(vedxdz.values,axes=[3,2,1,0])
uflux,vflux,dwflux = moc_offline_1deg.sig2fluxconv(tmpkmt,z_top,z_bot,dz.values,zsigu_top,zsigu_bot,tmpu,tmpv,mval,[nt,nz,ny,nx,nsig])
time2=timer.time()
print('sig2fluxconv:  ',time2-time1,'s')
del tmpkmt,tmpu,tmpv

# DEBUG: write vertically-integrated volume fluxes, from both z-coord & sigma-coord, to netcdf
if debug:
   uflux_sig=xr.DataArray(np.transpose(uflux,axes=[3,2,1,0]),dims=['time','sigma','nlat','nlon'], \
      coords={'time':time,'sigma':sig2,'ULAT':(('nlat','nlon'),ulat),'ULONG':(('nlat','nlon'),ulon)}, \
      name='uedydz_sig')
   vflux_sig=xr.DataArray(np.transpose(vflux,axes=[3,2,1,0]),dims=['time','sigma','nlat','nlon'], \
      coords={'time':time,'sigma':sig2,'ULAT':(('nlat','nlon'),ulat),'ULONG':(('nlat','nlon'),ulon)}, \
      name='vedxdz_sig')
   dwflux_sig=xr.DataArray(np.transpose(dwflux,axes=[3,2,1,0]),dims=['time','sigma','nlat','nlon'], \
      coords={'time':time,'sigma':sig2,'TLAT':(('nlat','nlon'),tlat),'TLONG':(('nlat','nlon'),tlon)}, \
      name='dwedxdy_sig')
   dwflux_sig_zint=dwflux_sig.where(dwflux_sig<1.e30).sum(dim='sigma')
   uflux_sig_zint=uflux_sig.where(uflux_sig<1.e30).sum(dim='sigma')
   vflux_sig_zint=vflux_sig.where(vflux_sig<1.e30).sum(dim='sigma')
   uflux_zint=uedydz.where(uedydz<1.e30).sum(dim='z_t')
   vflux_zint=vedxdz.where(vedxdz<1.e30).sum(dim='z_t')
   debug_ds['dwflux_sig']=dwflux_sig
   debug_ds['uflux_zint_fromsigma']=uflux_sig_zint
   debug_ds['uflux_zint_fromz']=uflux_zint
   debug_ds['vflux_zint_fromsigma']=vflux_sig_zint
   debug_ds['vflux_zint_fromz']=vflux_zint
   debug_ds['wflux_srf_fromsigma']=dwflux_sig_zint
   debug_ds['wflux_srf_fromz']=wedxdy[:,0,:,:]
   debug_ds.to_netcdf('/glade/scratch/yeager/python_debug.nc')

#6 b. Compute WDXDY (m^3/s) as partial integral of dWDXDY from ocean bottom
tmpdwflux = dwflux_sig[:,::-1,:,:]
wflux_sig=tmpdwflux.where(tmpdwflux<1.e30).cumsum(dim='sigma',skipna=True)

#7. Compute MOC in sigma-space
#7a.    integrate w_sigma in zonal direction
rmlak = np.zeros((nx,ny,2),dtype=np.int)
tmprmask = np.transpose(rmask.values,axes=[1,0])
tmptlat = np.transpose(tlat.values,axes=[1,0])
tmpw = np.transpose(wflux_sig.values,axes=[3,2,1,0])
rmlak[:,:,0] = np.where(tmprmask>0,1,0)
rmlak[:,:,1] = np.where((tmprmask>=6) & (tmprmask<=11),1,0)
tmpmoc = moc_offline_1deg.moczonalint(tmptlat,lat_aux_grid,rmlak,tmpw,mval,[nyaux,nx,ny,nz,nt,ntr])

#7b.    do meridional integration
MOCnew = xr.DataArray(np.transpose(tmpmoc,axes=[3,2,1,0]),dims=['time','transport_reg','sigma','lat_aux_grid'], \
        coords={'time':time,'transport_reg':transport_regions,'sigma':sig2,'lat_aux_grid':lat_aux_grid}, \
        name='MOC',attrs={'units':'Sv','long_name':'Meridional Overturning Circulation'})
MOCnew = MOCnew.where(MOCnew<mval).cumsum(dim='lat_aux_grid')
MOCnew = MOCnew*1.0e-6

#7c.     add MOC boundary condition at Atlantic southern boundary
# Find starting j-index for Atlantic region
lat_aux_atl_start = nyaux
for n in range(1,nyaux):
    section = (tlat.values >= lat_aux_grid.values[n-1]) & (tlat.values < lat_aux_grid.values[n]) & (np.transpose(rmlak[:,:,1],axes=[1,0]) == 1)
    if (section.any() and (n < lat_aux_atl_start)): 
       lat_aux_atl_start = n-1

# Regrid VDXDZ in sigma-coord from ULONG,ULAT to TLONG,ULAT grid
tmp=vflux_sig
tmpw=tmp.roll(nlon=1,roll_coords=False)        
tmpall=xr.concat([tmp,tmpw],dim='dummy')
vflux_sig=tmpall.where(tmpall<1e30).mean('dummy')
del tmp,tmpw,tmpall
# Zonal sum of Atlantic points
atlmask=xr.DataArray(np.where(rmask==6,1,0),dims=['nlat','nlon'])
atlmask=atlmask.roll(nlat=-1,roll_coords=False)
vflux_sig_xint=vflux_sig.where(atlmask==1).sum(dim='nlon')
#amoc_s=-vflux_sig_xint[0,::-1,lat_aux_atl_start-1].cumsum(dim='sigma')
amoc_s_new=-np.cumsum(vflux_sig_xint.values[0,::-1,lat_aux_atl_start-1],axis=0)
amoc_s = amoc_s[::-1]*1.e-6
print("amoc_s=",amoc_s)
#Something's wrong... need to debug southern boundary addition:
MOCnew[:,1,:,:] = MOCnew[:,1,:,:] + amoc_s


#8.    Write output to netcdf
out_ds=MOCnew.to_dataset(name='MOC')
out_ds.to_netcdf(out_file)

time2=timer.time()
print('DONE creating ',out_file,':  ',time2-time1,'s')
