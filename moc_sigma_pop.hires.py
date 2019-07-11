import xarray as xr               #netcdf multidim reading/writing/manipulation
import glob                       #globbing
import numpy as np                #numerics
import os  	                  #operating system commands
import time as timer
import pop_tools
#import cftime                     #netcdf time

# Set Options
time1=timer.time()
zcoord=True		# True-->compute MOC(z), False-->compute MOC(sigma2)
debug=False		# Only applies for zcoord=False

# Define input/output streams
in_dir='/glade/p/cgd/oce/projects/JRA55/IAF/g.e20.G.TL319_t13.control.001/ocn/tavg/'
out_dir='/glade/scratch/yeager/g.e20.G.TL319_t13.control.001/'
in_file = in_dir+'g.e20.G.TL319_t13.control.001.pop.tavg.0042-0061.nc'
#out_file = out_dir+'MOCsig2.0042-0061.python.test.nc'
#out_file_db = out_dir+'MOCsig2.0042-0061.python.debug.nc'
out_file = out_dir+'MOCz.0042-0061.python.nc'
out_file_db = out_dir+'MOCz.0042-0061.python.debug.nc'

# Define needed data files
POP1deg_gridfile='/glade/p/cgd/oce/people/yeager/POP_grids/gx1v6_ocn.nc'
POP0p1deg_gridfile='/glade/p/cgd/oce/people/yeager/POP_grids/tx0.1v3_POPgrid.nc'

# Import offline MOC routines written in fortran (compile with f2py if necessary)
f90mocroutines='./MOCoffline.POP_0p1deg.f90'
if not os.path.isfile('moc_offline_0p1deg.cpython-37m-x86_64-linux-gnu.so'):  
   print('MOCoffline compiling')
   os.system('f2py -c '+f90mocroutines+' -m moc_offline_0p1deg')
else: print('moc_offline already compiled')
import moc_offline_0p1deg

# Open a 1-deg POP history file to get MOC template
ds = xr.open_dataset(POP1deg_gridfile)
moc = ds['MOC'].isel(moc_comp=0)
time = moc['time'] 
lat_aux_grid = moc['lat_aux_grid'] 
transport_regions = moc['transport_regions'] 
ntr = transport_regions.shape[0]
nyaux = lat_aux_grid.shape[0]

# Open a 0.1-deg POP grid file to get PBC information
ds = xr.open_dataset(POP0p1deg_gridfile)
dzt = ds['DZT']/100
dzt.attrs['units'] = 'm'

# Regrid PBC thicknesses to U-grid
tmp=dzt
tmpe=tmp.roll(nlon=-1,roll_coords=False)        # wraparound shift to west, without changing coords
tmpn=tmp.shift(nlat=-1)                         # shift to south, without changing coords
tmpne=tmpn.roll(nlon=-1,roll_coords=False)      # wraparound shift to west, without changing coords
tmpall=xr.concat([tmp,tmpe,tmpn,tmpne],dim='dummy')
dzu=tmpall.min('dummy')
dzu.attrs['units'] = 'm'
del tmp,tmpe,tmpn,tmpne,tmpall

# Open a 0.1-deg POP history file to get needed fields
ds = xr.open_dataset(in_file)
pd     = ds['PD']
pd=pd.drop(['ULAT','ULONG'])            # this is a python bug that we are correcting
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
kmt.values[np.isnan(kmt.values)]=0	# get rid of _FillValues
kmu  = ds['KMU']
kmu.values[np.isnan(kmu.values)]=0	# get rid of _FillValues
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
mval=pd.encoding['_FillValue']

# grid-oriented volume fluxes 
uflux_z = u_e*dyu*dzu    # m^3/s
uflux_z = uflux_z.drop(['TLAT','TLONG'])
vflux_z = v_e*dxu*dzu    # m^3/s
vflux_z = vflux_z.drop(['TLAT','TLONG'])
wflux_z = w_e*tarea     # m^3/s
wflux_z = wflux_z.drop(['ULAT','ULONG'])

# Create a k-index array for masking purposes
kji = np.indices((nz,ny,nx))
kindices = kji[0,:,:,:] + 1

# Define top/bottom depths of POP T-grid
z_bot=z_w.values
z_bot=z_w.values+dz.values
z_top=z_w.values

if zcoord:
  # Define target vertical coordinates for MOC computation
  #   zcoord:  use POP T-grid vertical coordinates
  sig2=z_t.values
  sig2_top=z_top
  sig2_bot=z_bot
  nsig = np.size(sig2)
  zsigu_top = np.zeros((nx,ny,nsig,nt)) + z_top[None,None,:,None]
  zsigu_bot = np.zeros((nx,ny,nsig,nt)) + z_bot[None,None,:,None]
  uflux=uflux_z
  vflux=vflux_z
  wflux=wflux_z
else:
  # Define target vertical coordinates for MOC computation
  #    not zcoord:  define a set of well-chosen sigma2 levels 
  tmp1 = np.linspace(28,34.8,35)
  tmp2 = np.linspace(35,35.9,10)
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

  # Compute POP sigma2 field (NOTE: this is only necessary if zcoord=False)
  # using pop_tools:
  depth=xr.DataArray(np.ones(np.shape(salt))*2000.,dims=salt.dims,coords=salt.coords)
  sigma2=pop_tools.eos(salt=salt,temp=temp,depth=depth)
  sigma2 = sigma2-1000.
  # using gsw functions:
  # first, convert model depth to pressure
  #    (gsw function require numpy arrays, not xarray, so use ".values") 
  #    (use [:,None,None] syntax to conform the z_t and tlat into 3D arrays)
  #press = gsw.p_from_z(-z_t.values[:,None,None],tlat.values[None,:,:])
  # compute absolute salinity from practical salinity
  #SA = gsw.SA_from_SP(salt,press[None,:,:,:],tlon.values[None,None,:,:],tlat.values[None,None,:,:])
  # compute conservative temperature from potential temperature
  #CT = gsw.CT_from_pt(SA,temp.values)
  #sigma2 = gsw.sigma2(SA,CT)

  # convert to DataArray & regrid from T-grid to U-grid
  sigma2 = xr.DataArray(sigma2,name='Sigma2',dims=pd.dims,coords=pd.coords)
  sigma2.attrs=pd.attrs
  sigma2.attrs['long_name']='Sigma referenced to 2000m'
  sigma2.attrs['units']='kg/m^3'
  tmp=sigma2
  tmpe=tmp.roll(nlon=-1,roll_coords=False)        # wraparound shift to west, without changing coords
  tmpn=tmp.shift(nlat=-1)                         # shift to south, without changing coords
  tmpne=tmpn.roll(nlon=-1,roll_coords=False)      # wraparound shift to west, without changing coords
  tmpall=xr.concat([tmp,tmpe,tmpn,tmpne],dim='dummy')
  sigma2=tmpall.mean('dummy')
  sigma2.attrs=tmp.attrs                          # sigma2 now on U-grid
  del tmp,tmpe,tmpn,tmpne,tmpall

  # apply U-grid mask & remove density inversions
  mask=kindices>kmu.values[None,:,:]
  sigma2.values[mask[None,:,:,:]]=np.nan
  for iz in range(km-1,0,-1):
     print(iz)
     tmp1 = sigma2[:,iz,:,:].values
     tmp2 = sigma2[:,iz-1,:,:].values
     tmp3 = (~np.isnan(tmp1)) & (~np.isnan(tmp2)) & np.greater(tmp2,tmp1) 
     tmp2[tmp3] = tmp1[tmp3]-1.e-5
     sigma2.values[:,iz-1,:,:] = tmp2

  # debug test: read in sigma2 from NCL code:
  #tmpds = xr.open_dataset('/glade/scratch/yeager/g.e20.G.TL319_t13.control.001/MOCsig2.ncl.0042-0061.debug.nc')
  #sigma2.values=tmpds.pdu.values
  #  NOTE: this test shows that sigma2 differences are to blame for python/NCL discrepancies in MOC(sig2)!

  # Find sigma2 layer interface depths on U-grid
  sigma2.values[np.isnan(sigma2.values)]=mval
  tmpsig=np.transpose(sigma2.values,axes=[3,2,1,0])
  tmpkmu=np.transpose(kmu.values,axes=[1,0])
  tmpdzu=np.transpose(dzu.values,axes=[2,1,0])
  zsigu_top,zsigu_bot = moc_offline_0p1deg.sig2z(tmpsig,tmpkmu,z_t.values,tmpdzu,sig2_top,sig2_bot,mval,[nt,nz,ny,nx,nsig])
  del tmpsig,tmpkmu

  # Calculate horizontal & vertical volume fluxes:
  tmpdzu=np.transpose(dzu.values,axes=[2,1,0])
  tmpkmt=np.transpose(kmt.values,axes=[1,0])
  tmpkmu=np.transpose(kmu.values,axes=[1,0])
  tmpu=np.transpose(uflux_z.values.copy(),axes=[3,2,1,0])
  tmpv=np.transpose(vflux_z.values.copy(),axes=[3,2,1,0])
  uflux,vflux,dwflux = moc_offline_0p1deg.sig2fluxconv(tmpkmt,tmpkmu,z_top,tmpdzu,zsigu_top,zsigu_bot,tmpu,tmpv,mval,[nt,nz,ny,nx,nsig])
  time2=timer.time()
  print('sig2fluxconv:  ',time2-time1,'s')

  # Compute WDXDY (m^3/s) as partial integral of dWDXDY from ocean bottom
  uflux_sig=xr.DataArray(np.transpose(uflux,axes=[3,2,1,0]),dims=['time','sigma','nlat','nlon'], \
     coords={'time':time,'sigma':sig2,'ULAT':(('nlat','nlon'),ulat),'ULONG':(('nlat','nlon'),ulon)}, \
     name='uedydz_sig')
  vflux_sig=xr.DataArray(np.transpose(vflux,axes=[3,2,1,0]),dims=['time','sigma','nlat','nlon'], \
     coords={'time':time,'sigma':sig2,'ULAT':(('nlat','nlon'),ulat),'ULONG':(('nlat','nlon'),ulon)}, \
     name='vedxdz_sig')
  dwflux_sig=xr.DataArray(np.transpose(dwflux.copy(),axes=[3,2,1,0]),dims=['time','sigma','nlat','nlon'], \
        coords={'time':time,'sigma':sig2,'TLAT':(('nlat','nlon'),tlat),'TLONG':(('nlat','nlon'),tlon)}, \
        name='dwedxdy_sig')
  tmpdwflux = dwflux_sig[:,::-1,:,:].copy()
  tmpdwflux.values[tmpdwflux.values>1e30]=np.nan
  wflux_sig=tmpdwflux.cumsum(dim='sigma',skipna=True).copy()
  wflux_sig.values[np.isnan(tmpdwflux.values)]=np.nan
  wflux_sig=wflux_sig[:,::-1,:,:]

  uflux=uflux_sig.copy()
  vflux=vflux_sig.copy()
  wflux=wflux_sig.copy()

  if debug:
     # DEBUG: write sigma2 interface layer depth info to netcdf
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
     # DEBUG: write vertically-integrated volume fluxes, from both z-coord & sigma-coord, to netcdf
     dwflux_sig_zint=dwflux_sig.where(dwflux_sig<1.e30).sum(dim='sigma').copy()
     uflux_sig_zint=uflux_sig.where(uflux_sig<1.e30).sum(dim='sigma')
     vflux_sig_zint=vflux_sig.where(vflux_sig<1.e30).sum(dim='sigma')
     uflux_zint=uflux_z.where(uflux_z<1.e30).sum(dim='z_t')
     vflux_zint=vflux_z.where(vflux_z<1.e30).sum(dim='z_t')
     debug_ds['uflux_zint_fromsigma']=uflux_sig_zint
     debug_ds['uflux_zint_fromz']=uflux_zint
     debug_ds['vflux_zint_fromsigma']=vflux_sig_zint
     debug_ds['vflux_zint_fromz']=vflux_zint
     debug_ds['wflux_srf_fromsigma']=dwflux_sig_zint
     debug_ds['wflux_srf_fromz']=wflux_z[:,0,:,:]
     debug_ds.to_netcdf(out_file_db)

# Compute MOC in sigma-space
#   a. integrate w_sigma in zonal direction
rmlak = np.zeros((nx,ny,2),dtype=np.int)
tmprmask = np.transpose(rmask.values,axes=[1,0])
tmptlat = np.transpose(tlat.values,axes=[1,0])
tmpw = np.transpose(np.where(~np.isnan(wflux.values.copy()),wflux.values.copy(),mval),axes=[3,2,1,0])
rmlak[:,:,0] = np.where(tmprmask>0,1,0)
rmlak[:,:,1] = np.where((tmprmask>=6) & (tmprmask<=12),1,0)  	# include Baltic for 0p1
tmpmoc = moc_offline_0p1deg.moczonalint(tmptlat,lat_aux_grid,rmlak,tmpw,mval,[nyaux,nx,ny,nz,nt,ntr])
print('tmpmoc shape',np.shape(tmpmoc))

#   b. integrate in meridional direction
MOCnew = xr.DataArray(np.transpose(tmpmoc,axes=[3,2,1,0]),dims=['time','transport_reg','sigma','lat_aux_grid'], \
        coords={'time':time,'transport_regions':transport_regions,'sigma':sig2_top,'lat_aux_grid':lat_aux_grid}, \
        name='MOC')
print('mocnewshape',np.shape(MOCnew))
MOCnew = MOCnew.where(MOCnew<mval).cumsum(dim='lat_aux_grid')
MOCnew = MOCnew*1.0e-6

# Add MOC boundary condition at Atlantic southern boundary
#    a. find starting j-index for Atlantic region
lat_aux_atl_start = ny
for n in range(1,ny):
    section = (rmlak[:,n-1,1] == 1)
    if (section.any()):
       lat_aux_atl_start = n-2
       break
# print("lat_aux_atl_start= ",lat_aux_atl_start)

#    b. regrid VDXDZ in sigma-coord from ULONG,ULAT to TLONG,ULAT grid
tmp=vflux
tmpw=tmp.roll(nlon=1,roll_coords=False)        
tmpall=xr.concat([tmp,tmpw],dim='dummy')
vflux=tmpall.where(tmpall<1e30).mean('dummy')
del tmp,tmpw,tmpall
#    c. zonal integral of Atlantic points
atlmask=xr.DataArray(np.where(rmask==6,1,0),dims=['nlat','nlon'])
atlmask=atlmask.roll(nlat=-1,roll_coords=False)
vflux_xint=vflux.where(atlmask==1).sum(dim='nlon')
if zcoord:
  amoc_s=-vflux_xint[0,::-1,lat_aux_atl_start].cumsum(dim='z_t')
else:
  amoc_s=-vflux_xint[0,::-1,lat_aux_atl_start].cumsum(dim='sigma')
amoc_s = amoc_s[::-1]*1.e-6
print("amoc_s=",amoc_s)
MOCnew.values[:,1,:,:] = MOCnew.values[:,1,:,:] + amoc_s.values[None,:,None]


#8.    Write output to netcdf
if zcoord:
   MOCnew=MOCnew.rename({'sigma':'moc_z'})
#MOCnew.encoding=moc.encoding
MOCnew.attrs={'units':'Sv','long_name':'Meridional Overturning Circulation'}
out_ds=MOCnew.to_dataset(name='MOC')
out_ds.to_netcdf(out_file)

time2=timer.time()
print('DONE creating ',out_file,':  ',time2-time1,'s')
