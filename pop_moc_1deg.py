import xarray as xr               #netcdf multidim reading/writing/manipulation
import glob                       #globbing
import numpy as np                #numerics
import os  	                  #operating system commands
import time as timer
import pop_tools
#import cftime                     #netcdf time

# Set Options
time1=timer.time()
zcoord=False		# True-->compute MOC(z), False-->compute MOC(sigma2)
debug=True		# If zcoord=True, compute MOCdiff
                        # If zcoord=False, compute vertical sums
computew=False		# Only applies for zcoord=True. True--> w will be computed from div(u,v)

# Define input/output streams
in_dir='./tests/data/'
out_dir='/glade/scratch/yeager/POP_MOC/'
in_file = in_dir+'cesm_pop.h_g17.nc'
moc_template_file = './moc_template.nc'
if zcoord:
  out_file = out_dir+'cesm_pop.h_g17.MOCz.python.nc'
  if computew:
    out_file = out_dir+'cesm_pop.h_g17.MOCz.python.computew.nc'
  if debug:
    moc_template_file = in_file
else:
  out_file = out_dir+'cesm_pop.h_g17.MOCsig2.python.nc'
  out_file_db = out_dir+'cesm_pop.h_g17.MOCsig2.python.debug.nc'

# Import offline MOC routines written in fortran (compile with f2py if necessary)
f90mocroutines='./MOCoffline.POP_1deg.f90'
if not os.path.isfile('moc_offline_1deg.cpython-37m-x86_64-linux-gnu.so'):  
   print('MOCoffline compiling')
   os.system('f2py -c '+f90mocroutines+' -m moc_offline_1deg')
else: print('moc_offline already compiled')
import moc_offline_1deg

# Define MOC template
ds = xr.open_dataset(moc_template_file)
#moc = ds['MOC'].isel(moc_comp=0)
moc = ds['MOC']
time = ds['time'] 
lat_aux_grid = ds['lat_aux_grid'] 
lat_aux_grid.encoding['_FillValue']=None 	# because xarray is weird
transport_regions = ds['transport_regions'] 
moc_components = ds['moc_components'] 
ncomp = moc_components.shape[0]
ntr = transport_regions.shape[0]
nyaux = lat_aux_grid.shape[0]

# Open a POP history file requiring MOC
ds = xr.open_dataset(in_file)
pd     = ds['PD']
pd=pd.drop(['ULAT','ULONG'])            # this is a python bug that we are correcting
temp   = ds['TEMP']
temp=temp.drop(['ULAT','ULONG'])
salt   = ds['SALT']
salt=salt.drop(['ULAT','ULONG'])
u_e   = ds['UVEL']/100
u_e=u_e.drop(['TLAT','TLONG'])
u_e.attrs['units']='m/s'
u_i   = ds['UISOP']/100
u_i=u_i.drop(['ULAT','TLONG'])
u_i.attrs['units']='m/s'
u_s   = ds['USUBM']/100
u_s=u_s.drop(['ULAT','TLONG'])
u_s.attrs['units']='m/s'
v_e   = ds['VVEL']/100
v_e=v_e.drop(['TLAT','TLONG'])
v_e.attrs['units']='m/s'
v_i   = ds['VISOP']/100
v_i=v_i.drop(['TLAT','ULONG'])
v_i.attrs['units']='m/s'
v_s   = ds['VSUBM']/100
v_s=v_s.drop(['TLAT','ULONG'])
v_s.attrs['units']='m/s'
w_e   = ds['WVEL']/100
w_e=w_e.drop(['ULAT','ULONG'])
w_e.attrs['units']='m/s'
w_i   = ds['WISOP']/100
w_i=w_i.drop(['ULAT','ULONG'])
w_i.attrs['units']='m/s'
w_s   = ds['WSUBM']/100
w_s=w_s.drop(['ULAT','ULONG'])
w_s.attrs['units']='m/s'
ulat   = ds['ULAT']
ulon   = ds['ULONG']
tlat   = ds['TLAT']
tlon   = ds['TLONG']
kmt  = ds['KMT']
kmu  = ds['KMU']
dxu    = ds['DXU']/100
dxu.attrs['units']='m'
dyu    = ds['DYU']/100
dyu.attrs['units']='m'
hte    = ds['HTE']/100
hte.attrs['units']='m'
htn    = ds['HTN']/100
htn.attrs['units']='m'
rmask  = ds['REGION_MASK']
tarea  = ds['TAREA']/100/100
tarea.attrs['units']='m^2'
uarea  = ds['UAREA']/100/100
uarea.attrs['units']='m^2'
time   = ds['time']
time.encoding['_FillValue']=None 	# because xarray is weird
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

# Create a k-index array for masking purposes
kji = np.indices((nz,ny,nx))
kindices = kji[0,:,:,:] + 1

# Zero out flow below bottom (overflow param):
mask=kindices>kmu.values[None,:,:]
u_e.values[mask[None,:,:,:]]=0.
v_e.values[mask[None,:,:,:]]=0.
mask=kindices>kmt.values[None,:,:]
w_e.values[mask[None,:,:,:]]=0.

# grid-oriented volume fluxes 
ueflux_z = u_e*dyu*dz	# m^3/s
veflux_z = v_e*dxu*dz	# m^3/s
weflux_z = w_e*tarea	# m^3/s
uiflux_z = u_i*hte*dz	# m^3/s
viflux_z = v_i*htn*dz	# m^3/s
wiflux_z = w_i*tarea	# m^3/s
usflux_z = u_s*hte*dz	# m^3/s
vsflux_z = v_s*htn*dz	# m^3/s
wsflux_z = w_s*tarea	# m^3/s

# Define top/bottom depths of POP T-grid
z_bot=z_w.values
z_bot=z_w.values+dz.values
z_top=z_w.values

if zcoord:
  # Define target vertical coordinates for MOC computation
  #   zcoord:  use POP T-grid vertical coordinates
  mocz=xr.DataArray(np.append(z_top[0],z_bot),dims=['moc_z'],attrs={'long_name':'depth from surface to top of layer','units':'m','positive':'down'})
  mocz.encoding['_FillValue']=None
  mocnz=nz+1
  # Define vertical volume flux 
  if computew:
     targnz=nz
     target_z_top = np.zeros((nx,ny,targnz,nt)) + z_top[None,None,:,None]
     target_z_bot = np.zeros((nx,ny,targnz,nt)) + z_bot[None,None,:,None]
     # Calculate horizontal & vertical volume fluxes:
     tmpkmt=np.transpose(kmt.values,axes=[1,0])
     tmpu=np.transpose(ueflux_z.values.copy(),axes=[3,2,1,0])
     tmpv=np.transpose(veflux_z.values.copy(),axes=[3,2,1,0])
     utmp,vtmp,wtmp = moc_offline_1deg.fluxconv(tmpkmt,z_top,z_bot,dz.values,target_z_top,target_z_bot,tmpu,tmpv,mval,[nt,nz,ny,nx,targnz])
     ueflux=xr.DataArray(np.transpose(utmp.copy(),axes=[3,2,1,0]),dims=['time','z_t','nlat','nlon'], \
           coords={'time':time,'z_t':z_t,'ULAT':(('nlat','nlon'),ulat),'ULONG':(('nlat','nlon'),ulon)}, \
           name='uedydz',attrs={'units':'m^3/s'})
     veflux=xr.DataArray(np.transpose(vtmp.copy(),axes=[3,2,1,0]),dims=['time','z_t','nlat','nlon'], \
           coords={'time':time,'z_t':z_t,'ULAT':(('nlat','nlon'),ulat),'ULONG':(('nlat','nlon'),ulon)}, \
           name='vedxdz',attrs={'units':'m^3/s'})
     weflux=xr.DataArray(np.transpose(wtmp.copy(),axes=[3,2,1,0]),dims=['time','z_top','nlat','nlon'], \
           coords={'time':time,'z_top':z_top,'TLAT':(('nlat','nlon'),tlat),'TLONG':(('nlat','nlon'),tlon)}, \
           name='wedxdy',attrs={'units':'m^3/s'})
  else:
     ueflux=ueflux_z
     veflux=veflux_z
     weflux=weflux_z
     uiflux=uiflux_z
     viflux=viflux_z
     wiflux=wiflux_z
     usflux=usflux_z
     vsflux=vsflux_z
     wsflux=wsflux_z
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
  mocz=xr.DataArray(np.append(sig2_top[0],sig2_bot),dims=['sigma'],attrs={'long_name':'Sigma2 at top of layer','units':'kg/m^3'})
  mocz.encoding['_FillValue']=None
  mocnz=nsig+1

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
  #tmpds = xr.open_dataset('/glade/scratch/yeager/g.DPLE.GECOIAF.T62_g16.009.chey/g.DPLE.GECOIAF.T62_g16.009.chey.pop.h.0301-01.MOCsig2.debug.nc')
  #sigma2.values=tmpds.pdu.values
  #  NOTE: this test shows that sigma2 differences are to blame for python/NCL discrepancies in MOC(sig2)!

  # Find sigma2 layer interface depths on U-grid
  sigma2.values[np.isnan(sigma2.values)]=mval
  tmpsig=np.transpose(sigma2.values,axes=[3,2,1,0])
  tmpkmu=np.transpose(kmu.values,axes=[1,0])
  target_z_top,target_z_bot = moc_offline_1deg.sig2z(tmpsig,tmpkmu,z_t.values,z_bot,sig2_top,sig2_bot,mval,[nt,nz,ny,nx,nsig])
  del tmpsig,tmpkmu

  # Calculate horizontal & vertical volume fluxes:
  tmpkmt=np.transpose(kmt.values,axes=[1,0])

  tmpu=np.transpose(ueflux_z.values.copy(),axes=[3,2,1,0])
  tmpv=np.transpose(veflux_z.values.copy(),axes=[3,2,1,0])
  utmp,vtmp,wtmp = moc_offline_1deg.fluxconv(tmpkmt,z_top,z_bot,dz.values,target_z_top,target_z_bot,tmpu,tmpv,mval,[nt,nz,ny,nx,nsig])
  ueflux_sig=xr.DataArray(np.transpose(utmp.copy(),axes=[3,2,1,0]),dims=['time','sigma','nlat','nlon'], \
        coords={'time':time,'sigma':sig2,'ULAT':(('nlat','nlon'),ulat),'ULONG':(('nlat','nlon'),ulon)}, \
        name='uedydz_sig',attrs={'units':'m^3/s'})
  veflux_sig=xr.DataArray(np.transpose(vtmp.copy(),axes=[3,2,1,0]),dims=['time','sigma','nlat','nlon'], \
        coords={'time':time,'sigma':sig2,'ULAT':(('nlat','nlon'),ulat),'ULONG':(('nlat','nlon'),ulon)}, \
        name='vedxdz_sig',attrs={'units':'m^3/s'})
  weflux_sig=xr.DataArray(np.transpose(wtmp.copy(),axes=[3,2,1,0]),dims=['time','sigma_top','nlat','nlon'], \
        coords={'time':time,'sigma_top':sig2_top,'TLAT':(('nlat','nlon'),tlat),'TLONG':(('nlat','nlon'),tlon)}, \
        name='wedxdy_sig',attrs={'units':'m^3/s'})

  tmpu=np.transpose(uiflux_z.values.copy(),axes=[3,2,1,0])
  tmpv=np.transpose(viflux_z.values.copy(),axes=[3,2,1,0])
  utmp,vtmp,wtmp = moc_offline_1deg.fluxconv(tmpkmt,z_top,z_bot,dz.values,target_z_top,target_z_bot,tmpu,tmpv,mval,[nt,nz,ny,nx,nsig])
  uiflux_sig=xr.DataArray(np.transpose(utmp.copy(),axes=[3,2,1,0]),dims=['time','sigma','nlat','nlon'], \
        coords={'time':time,'sigma':sig2,'ULAT':(('nlat','nlon'),ulat),'ULONG':(('nlat','nlon'),ulon)}, \
        name='uidydz_sig',attrs={'units':'m^3/s'})
  viflux_sig=xr.DataArray(np.transpose(vtmp.copy(),axes=[3,2,1,0]),dims=['time','sigma','nlat','nlon'], \
        coords={'time':time,'sigma':sig2,'ULAT':(('nlat','nlon'),ulat),'ULONG':(('nlat','nlon'),ulon)}, \
        name='vidxdz_sig',attrs={'units':'m^3/s'})
  wiflux_sig=xr.DataArray(np.transpose(wtmp.copy(),axes=[3,2,1,0]),dims=['time','sigma_top','nlat','nlon'], \
        coords={'time':time,'sigma_top':sig2_top,'TLAT':(('nlat','nlon'),tlat),'TLONG':(('nlat','nlon'),tlon)}, \
        name='widxdy_sig',attrs={'units':'m^3/s'})

  tmpu=np.transpose(usflux_z.values.copy(),axes=[3,2,1,0])
  tmpv=np.transpose(vsflux_z.values.copy(),axes=[3,2,1,0])
  utmp,vtmp,wtmp = moc_offline_1deg.fluxconv(tmpkmt,z_top,z_bot,dz.values,target_z_top,target_z_bot,tmpu,tmpv,mval,[nt,nz,ny,nx,nsig])
  usflux_sig=xr.DataArray(np.transpose(utmp.copy(),axes=[3,2,1,0]),dims=['time','sigma','nlat','nlon'], \
        coords={'time':time,'sigma':sig2,'ULAT':(('nlat','nlon'),ulat),'ULONG':(('nlat','nlon'),ulon)}, \
        name='usdydz_sig',attrs={'units':'m^3/s'})
  vsflux_sig=xr.DataArray(np.transpose(vtmp.copy(),axes=[3,2,1,0]),dims=['time','sigma','nlat','nlon'], \
        coords={'time':time,'sigma':sig2,'ULAT':(('nlat','nlon'),ulat),'ULONG':(('nlat','nlon'),ulon)}, \
        name='vsdxdz_sig',attrs={'units':'m^3/s'})
  wsflux_sig=xr.DataArray(np.transpose(wtmp.copy(),axes=[3,2,1,0]),dims=['time','sigma_top','nlat','nlon'], \
        coords={'time':time,'sigma_top':sig2_top,'TLAT':(('nlat','nlon'),tlat),'TLONG':(('nlat','nlon'),tlon)}, \
        name='wsdxdy_sig',attrs={'units':'m^3/s'})

  ueflux=ueflux_sig.copy()
  veflux=veflux_sig.copy()
  weflux=weflux_sig.copy()
  uiflux=uiflux_sig.copy()
  viflux=viflux_sig.copy()
  wiflux=wiflux_sig.copy()
  usflux=usflux_sig.copy()
  vsflux=vsflux_sig.copy()
  wsflux=wsflux_sig.copy()

  if debug:
     # DEBUG: write sigma2 interface layer depth info to netcdf
     tmparr1=xr.DataArray(np.transpose(target_z_top,axes=[3,2,1,0]),dims=['time','sigma','nlat','nlon'], \
       coords={'time':time,'sigma':sig2,'TLAT':(('nlat','nlon'),tlat),'TLONG':(('nlat','nlon'),tlon)}, \
       name='SIG_top')
     tmparr2=xr.DataArray(np.transpose(target_z_bot,axes=[3,2,1,0]),dims=['time','sigma','nlat','nlon'], \
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
     ueflux_sig_zint=ueflux_sig.where(ueflux_sig<1.e30).sum(dim='sigma')
     veflux_sig_zint=veflux_sig.where(veflux_sig<1.e30).sum(dim='sigma')
     ueflux_zint=ueflux_z.where(ueflux_z<1.e30).sum(dim='z_t')
     veflux_zint=veflux_z.where(veflux_z<1.e30).sum(dim='z_t')
     debug_ds['ueflux_zint_fromsigma']=ueflux_sig_zint
     debug_ds['ueflux_zint_fromz']=ueflux_zint
     debug_ds['veflux_zint_fromsigma']=veflux_sig_zint
     debug_ds['veflux_zint_fromz']=veflux_zint
     debug_ds['weflux_srf_fromsigma']=weflux_sig[:,0,:,:]
     debug_ds['weflux_srf_fromz']=weflux_z[:,0,:,:]
     debug_ds['uedydz_z']=ueflux_z
     debug_ds['vedxdz_z']=veflux_z
     debug_ds['weflux_sig']=weflux_sig.copy()
     debug_ds.to_netcdf(out_file_db)

# Compute MOC
#   a. integrate wflux in zonal direction (zonal sum of wflux already in m^3/s)
rmlak = np.zeros((nx,ny,2),dtype=np.int)
tmprmask = np.transpose(rmask.values,axes=[1,0])
tmptlat = np.transpose(tlat.values,axes=[1,0])
rmlak[:,:,0] = np.where(tmprmask>0,1,0)
rmlak[:,:,1] = np.where((tmprmask>=6) & (tmprmask<=11),1,0)
tmpw = np.transpose(np.where(~np.isnan(weflux.values.copy()),weflux.values.copy(),mval),axes=[3,2,1,0])
tmpmoc_e = moc_offline_1deg.wzonalsum(tmptlat,lat_aux_grid,rmlak,tmpw,mval,[nyaux,nx,ny,nz,nt,ntr])
tmpmoc_e=np.single(np.append(tmpmoc_e,np.zeros((nyaux,1,ntr,nt)),axis=1))	# add ocean bottom
tmpw = np.transpose(np.where(~np.isnan(wiflux.values.copy()),wiflux.values.copy(),mval),axes=[3,2,1,0])
tmpmoc_i = moc_offline_1deg.wzonalsum(tmptlat,lat_aux_grid,rmlak,tmpw,mval,[nyaux,nx,ny,nz,nt,ntr])
tmpmoc_i=np.single(np.append(tmpmoc_i,np.zeros((nyaux,1,ntr,nt)),axis=1))	# add ocean bottom
tmpw = np.transpose(np.where(~np.isnan(wsflux.values.copy()),wsflux.values.copy(),mval),axes=[3,2,1,0])
tmpmoc_s = moc_offline_1deg.wzonalsum(tmptlat,lat_aux_grid,rmlak,tmpw,mval,[nyaux,nx,ny,nz,nt,ntr])
tmpmoc_s=np.single(np.append(tmpmoc_s,np.zeros((nyaux,1,ntr,nt)),axis=1))	# add ocean bottom
#print('tmpmoc shape',np.shape(tmpmoc))

#   b. integrate in meridional direction
if zcoord:
   MOCnew = xr.DataArray(np.zeros((nt,ntr,ncomp,mocnz,nyaux),dtype=np.single),dims=['time','transport_reg','moc_comp','moc_z','lat_aux_grid'], \
      coords={'time':time,'transport_regions':transport_regions,'moc_comp':moc_components,'moc_z':mocz,'lat_aux_grid':lat_aux_grid}, \
      name='MOC')
else:
   MOCnew = xr.DataArray(np.zeros((nt,ntr,ncomp,mocnz,nyaux),dtype=np.single),dims=['time','transport_reg','moc_comp','sigma','lat_aux_grid'], \
      coords={'time':time,'transport_regions':transport_regions,'moc_comp':moc_components,'sigma':mocz,'lat_aux_grid':lat_aux_grid}, \
      name='MOC')
MOCnew.values[:,:,0,:,:] = np.transpose(tmpmoc_e,axes=[3,2,1,0])
MOCnew.values[:,:,1,:,:] = np.transpose(tmpmoc_i,axes=[3,2,1,0])
MOCnew.values[:,:,2,:,:] = np.transpose(tmpmoc_s,axes=[3,2,1,0])
print('mocnewshape',np.shape(MOCnew))
MOCnew = MOCnew.where(MOCnew<mval).cumsum(dim='lat_aux_grid')
MOCnew = MOCnew*1.0e-6
MOCnew.attrs={'units':'Sverdrups','long_name':'Meridional Overturning Circulation'}
MOCnew.encoding['_FillValue']=mval

# Add MOC boundary condition at Atlantic southern boundary
#    a. find starting j-index for Atlantic region
lat_aux_atl_start = ny
for n in range(1,ny):
    section = (rmlak[:,n-1,1] == 1)
    if (section.any()): 
       lat_aux_atl_start = n-2
       break
# print("lat_aux_atl_start= ",lat_aux_atl_start)

#    b. regrid VDXDZ from ULONG,ULAT to TLONG,ULAT grid
tmp=veflux
tmpw=tmp.roll(nlon=1,roll_coords=False)        
tmpall=xr.concat([tmp,tmpw],dim='dummy')
veflux=tmpall.where(tmpall<1e30).mean('dummy')
del tmp,tmpw,tmpall
#    c. zonal integral of Atlantic points
atlmask=xr.DataArray(np.where(rmask==6,1,0),dims=['nlat','nlon'])
atlmask=atlmask.roll(nlat=-1,roll_coords=False)
veflux_xint=veflux.where(atlmask==1).sum(dim='nlon')
viflux_xint=viflux.where(atlmask==1).sum(dim='nlon')
vsflux_xint=vsflux.where(atlmask==1).sum(dim='nlon')
if zcoord:
  amoc_s_e=-veflux_xint[0,::-1,lat_aux_atl_start].cumsum(dim='z_t')
  amoc_s_i=-viflux_xint[0,::-1,lat_aux_atl_start].cumsum(dim='z_t')
  amoc_s_s=-vsflux_xint[0,::-1,lat_aux_atl_start].cumsum(dim='z_t')
else:
  amoc_s_e=-veflux_xint[0,::-1,lat_aux_atl_start].cumsum(dim='sigma')
  amoc_s_i=-viflux_xint[0,::-1,lat_aux_atl_start].cumsum(dim='sigma')
  amoc_s_s=-vsflux_xint[0,::-1,lat_aux_atl_start].cumsum(dim='sigma')
amoc_s_e = amoc_s_e[::-1]*1.e-6
amoc_s_i = amoc_s_i[::-1]*1.e-6
amoc_s_s = amoc_s_s[::-1]*1.e-6
MOCnew.values[:,1,0,0:mocnz-1,:] = MOCnew.values[:,1,0,0:mocnz-1,:] + amoc_s_e.values[None,:,None]
MOCnew.values[:,1,1,0:mocnz-1,:] = MOCnew.values[:,1,1,0:mocnz-1,:] + amoc_s_i.values[None,:,None]
MOCnew.values[:,1,2,0:mocnz-1,:] = MOCnew.values[:,1,2,0:mocnz-1,:] + amoc_s_s.values[None,:,None]


#8.    Write output to netcdf
#MOCnew.encoding=moc.encoding
out_ds=MOCnew.to_dataset(name='MOC')
if zcoord and debug:
   MOCdiff=MOCnew.copy()
   MOCdiff.values = MOCnew.values - moc.values
   out_ds['MOCdiff']=MOCdiff
#out_ds.to_netcdf(out_file,encoding={'MOC':{'_FillValue':mval}})
out_ds.to_netcdf(out_file)

time2=timer.time()
print('DONE creating ',out_file,':  ',time2-time1,'s')
