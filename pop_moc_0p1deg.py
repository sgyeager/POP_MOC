import xarray as xr               #netcdf multidim reading/writing/manipulation
import glob                       #globbing
import numpy as np                #numerics
import os  	                  #operating system commands
import subprocess
import time as timer
import pop_tools
import sys
#import cftime                     #netcdf time

time1=timer.time()

# Set Default Options
verbose_output=False	# True with "-v"
append_to_infile=False	# True with "-a"
sigmacoord=False	# True with "-s", otherwise compute MOC(z)
debug=False		# True with "-d" (only applies for sigmacoord=True)
computew=False          # True with "-w" (only applies for zcoord=True, w will be computed from div(u,v))

# Define input/output settings & options based on command line input
moc_template_file = './moc_template.nc'
nargs=len(sys.argv)
in_file=sys.argv[-1]

if ('-a' in sys.argv[:]):
   append_to_infile=True
if ('-v' in sys.argv[:]):
   verbose_output=True
if ('-s' in sys.argv[:]):
   sigmacoord=True
if ('-d' in sys.argv[:]):
   debug=True
if ('-w' in sys.argv[:]):
   computew=True

if sigmacoord:
   out_file=in_file[:-2]+'MOCsig.nc'
else:
   out_file=in_file[:-2]+'MOCz.nc'
if computew:
   out_file=out_file[:-2]+'computew.nc'
if debug:
   out_file_db=out_file[:-2]+'debug.nc'

#in_dir='/glade/scratch/yeager/iHesp/'
#out_dir='/glade/scratch/yeager/iHesp/'
#in_file = in_dir+'B.E.13.B1850C5.ne120_t12.sehires38.003.sunway_02.pop.h.0116-01.nc'
#out_file = out_dir+'B.E.13.B1850C5.ne120_t12.sehires38.003.sunway_02.pop.h.0116-01.MOC.nc'
#out_file_db = out_dir+'MOCz.0042-0061.python.debug.nc'

# Regression test for MOC(z) computation:
# in_dir='/glade/p/cgd/oce/projects/JRA55/IAF/g.e20.G.TL319_t13.control.001/ocn/tavg/'
# in_file=in_dir+'g.e20.G.TL319_t13.control.001.pop.tavg.0042-0061.nc'
#  
# should reproduce vetted MOC(z) here (verified to within roundoff of Frank Bryan's fortran code):
# out_file = '/glade/p/cgd/oce/projects/JRA55/IAF/g.e20.G.TL319_t13.control.001/ocn/tavg/MOCz.0042-0061.nc'

# Import offline MOC routines written in fortran (compile with f2py if necessary)
f90mocroutines='./MOCoffline.POP_0p1deg.f90'
if not os.path.isfile('moc_offline_0p1deg.cpython-37m-x86_64-linux-gnu.so'):  
   print('MOCoffline compiling')
   cmd = ['f2py','-c',f90mocroutines,'-m','moc_offline_0p1deg']
   subprocess.call(cmd)
else: print('moc_offline already compiled')
import moc_offline_0p1deg

# Define MOC coordinates
ds = xr.open_dataset(moc_template_file)
#moc = ds['MOC'].isel(moc_comp=0)
lat_aux_grid = ds['lat_aux_grid'] 
lat_aux_grid.encoding['_FillValue']=None  
transport_regions = ds['transport_regions'] 
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
v_e   = ds['VVEL']/100
v_e=v_e.drop(['TLAT','TLONG'])
v_e.attrs['units']='m/s'
w_e   = ds['WVEL']/100
w_e=w_e.drop(['ULAT','ULONG'])
w_e.attrs['units']='m/s'
ulat   = ds['ULAT']
ulon   = ds['ULONG']
tlat   = ds['TLAT']
tlon   = ds['TLONG']
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
time.encoding['_FillValue']=None    
z_t   = ds['z_t']/100
z_t.attrs['units']='m'
z_w   = ds['z_w']/100
z_w.attrs['units']='m'
z_w_bot   = ds['z_w_bot']/100
z_w_bot.attrs['units']='m'
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

# Define top/bottom depths of POP T-grid
z_bot=z_w.values
z_bot=z_w.values+dz.values
z_top=z_w.values

time2=timer.time()
print('Timing:  Input data read =  ',time2-time1,'s')

# File containing POP 0.1deg partial bottom cell (pbc) information
#POP0p1deg_pbc='./dzt_t13.nc'
#ds = xr.open_dataset(POP0p1deg_pbc)
#dzt = ds['DZT']/100
#dzt.attrs['units'] = 'm'
# Compute PBC from grid info:
dzt = np.zeros((nz,ny,nx)) + dz.values[:,None,None]
for iz in range(0,nz):
   bottom = kmt.values==(iz+1)
   belowbottom = kmt.values<(iz+1)
   count1 = np.count_nonzero(bottom)
   count2 = np.count_nonzero(belowbottom)
   if (count1 > 0):
      tmp2 = dzt[iz,:,:]
      tmp2[bottom] = ht.values[bottom] - z_w_bot.values[iz-1]
      dzt[iz,:,:]=tmp2
   if (count2 > 0):
      tmp2 = dzt[iz,:,:]
      tmp2[belowbottom] = mval
      dzt[iz,:,:]=tmp2
dzt = xr.DataArray(dzt,dims=['z_t','nlat','nlon'])
dzt.encoding['_FillValue']=mval

# Regrid PBC thicknesses to U-grid
tmp=dzt
tmpe=tmp.roll(nlon=-1,roll_coords=False)        # wraparound shift to west, without changing coords
tmpn=tmp.shift(nlat=-1)                         # shift to south, without changing coords
tmpne=tmpn.roll(nlon=-1,roll_coords=False)      # wraparound shift to west, without changing coords
tmpall=xr.concat([tmp,tmpe,tmpn,tmpne],dim='dummy')
dzu=tmpall.min('dummy')
dzu.attrs['units'] = 'm'
del tmp,tmpe,tmpn,tmpne,tmpall

time2a=timer.time()
print('Timing:  Computed partial bottom cells =  ',time2a-time2,'s')

# grid-oriented volume fluxes 
ueflux_z = u_e*dyu*dzu    # m^3/s
ueflux_z = ueflux_z.drop(['TLAT','TLONG'])
veflux_z = v_e*dxu*dzu    # m^3/s
veflux_z = veflux_z.drop(['TLAT','TLONG'])
weflux_z = w_e*tarea     # m^3/s
weflux_z = weflux_z.drop(['ULAT','ULONG'])

if not sigmacoord:
  # Define target vertical coordinates for MOC computation
  #   not sigmacoord:  use POP T-grid vertical coordinates
  mocz=xr.DataArray(np.append(z_top[0],z_bot),dims=['moc_z'],attrs={'long_name':'depth from surface to top of layer','units':'m','positive':'down'})
  mocz.encoding['_FillValue']=None
  mocnz=nz+1
  # Define vertical volume flux 
  if computew:
     targnz=nz
     target_z_top = np.zeros((nx,ny,targnz,nt)) + z_top[None,None,:,None]
     target_z_bot = np.zeros((nx,ny,targnz,nt)) + z_bot[None,None,:,None]
     # Calculate horizontal & vertical volume fluxes:
     tmpdzu=np.transpose(dzu.values,axes=[2,1,0])
     tmpkmu=np.transpose(kmu.values,axes=[1,0])
     tmpkmt=np.transpose(kmt.values,axes=[1,0])
     tmpu=np.transpose(ueflux_z.values.copy(),axes=[3,2,1,0])
     tmpv=np.transpose(veflux_z.values.copy(),axes=[3,2,1,0])
     utmp,vtmp,wtmp = moc_offline_0p1deg.fluxconv(tmpkmt,tmpkmu,z_top,tmpdzu,target_z_top,target_z_bot,tmpu,tmpv,mval,[nt,nz,ny,nx,targnz])
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
else:
  # Define target vertical coordinates for MOC computation
  #    sigmacoord:  define a set of well-chosen sigma2 levels 
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

  # Compute POP sigma2 field (NOTE: this is only necessary if sigmacoord=True)
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
  time3=timer.time()
  print('Timing:  EOS call =  ',time3-time2a,'s')

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
     tmp1 = sigma2[:,iz,:,:].values
     tmp2 = sigma2[:,iz-1,:,:].values
     tmp3 = (~np.isnan(tmp1)) & (~np.isnan(tmp2)) & np.greater(tmp2,tmp1) 
     tmp2[tmp3] = tmp1[tmp3]-1.e-5
     sigma2.values[:,iz-1,:,:] = tmp2
  time4=timer.time()
  print('Timing:  Sigma remapping & correcting inversions =  ',time4-time3,'s')

  # debug test: read in sigma2 from NCL code:
  #tmpds = xr.open_dataset('/glade/scratch/yeager/g.e20.G.TL319_t13.control.001/MOCsig2.ncl.0042-0061.debug.nc')
  #sigma2.values=tmpds.pdu.values
  #  NOTE: this test shows that sigma2 differences are to blame for python/NCL discrepancies in MOC(sig2)!

  # Find sigma2 layer interface depths on U-grid
  sigma2.values[np.isnan(sigma2.values)]=mval
  tmpsig=np.transpose(sigma2.values,axes=[3,2,1,0])
  tmpkmu=np.transpose(kmu.values,axes=[1,0])
  tmpdzu=np.transpose(dzu.values,axes=[2,1,0])
  target_z_top,target_z_bot = moc_offline_0p1deg.sig2z(tmpsig,tmpkmu,z_t.values,tmpdzu,sig2_top,sig2_bot,mval,[nt,nz,ny,nx,nsig])
  del tmpsig,tmpkmu
  time5=timer.time()
  print('Timing:  sig2z call =  ',time5-time4,'s')

  # Calculate horizontal & vertical volume fluxes:
  tmpdzu=np.transpose(dzu.values,axes=[2,1,0])
  tmpkmt=np.transpose(kmt.values,axes=[1,0])
  tmpkmu=np.transpose(kmu.values,axes=[1,0])
  tmpu=np.transpose(ueflux_z.values.copy(),axes=[3,2,1,0])
  tmpv=np.transpose(veflux_z.values.copy(),axes=[3,2,1,0])
  utmp,vtmp,wtmp = moc_offline_0p1deg.fluxconv(tmpkmt,tmpkmu,z_top,tmpdzu,target_z_top,target_z_bot,tmpu,tmpv,mval,[nt,nz,ny,nx,nsig])
  ueflux_sig=xr.DataArray(np.transpose(utmp.copy(),axes=[3,2,1,0]),dims=['time','sigma','nlat','nlon'], \
     coords={'time':time,'sigma':sig2,'ULAT':(('nlat','nlon'),ulat),'ULONG':(('nlat','nlon'),ulon)}, \
     name='uedydz_sig',attrs={'units':'m^3/s'})
  veflux_sig=xr.DataArray(np.transpose(vtmp.copy(),axes=[3,2,1,0]),dims=['time','sigma','nlat','nlon'], \
     coords={'time':time,'sigma':sig2,'ULAT':(('nlat','nlon'),ulat),'ULONG':(('nlat','nlon'),ulon)}, \
     name='vedxdz_sig',attrs={'units':'m^3/s'})
  weflux_sig=xr.DataArray(np.transpose(wtmp.copy(),axes=[3,2,1,0]),dims=['time','sigma','nlat','nlon'], \
        coords={'time':time,'sigma':sig2,'TLAT':(('nlat','nlon'),tlat),'TLONG':(('nlat','nlon'),tlon)}, \
        name='wedxdy_sig',attrs={'units':'m^3/s'})
  ueflux=ueflux_sig.copy()
  veflux=veflux_sig.copy()
  weflux=weflux_sig.copy()
  time6=timer.time()
  print('Timing:  fluxconv call =  ',time6-time5,'s')

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
     debug_ds.to_netcdf(out_file_db)
     time7=timer.time()
     print('Timing:  debug stuff =  ',time7-time6,'s')

# Compute MOC
#   a. integrate wflux in zonal direction (zonal sum of wflux already in m^3/s)
time8=timer.time()
rmlak = np.zeros((nx,ny,2),dtype=np.int)
tmprmask = np.transpose(rmask.values,axes=[1,0])
tmptlat = np.transpose(tlat.values,axes=[1,0])
rmlak[:,:,0] = np.where(tmprmask>0,1,0)
rmlak[:,:,1] = np.where((tmprmask>=6) & (tmprmask<=12),1,0)  	# include Baltic for 0p1
tmpw = np.transpose(np.where(~np.isnan(weflux.values.copy()),weflux.values.copy(),mval),axes=[3,2,1,0])
tmpmoc_e = moc_offline_0p1deg.wzonalsum(tmptlat,lat_aux_grid,rmlak,tmpw,mval,[nyaux,nx,ny,nz,nt,ntr])
tmpmoc_e=np.single(np.append(tmpmoc_e,np.zeros((nyaux,1,ntr,nt)),axis=1))       # add ocean bottom
#print('tmpmoc shape',np.shape(tmpmoc))
time9=timer.time()
print('Timing:  wzonalsum call =  ',time9-time8,'s')

#   b. integrate in meridional direction
if not sigmacoord:
   MOCnew = xr.DataArray(np.zeros((nt,ntr,mocnz,nyaux),dtype=np.single),dims=['time','transport_reg','moc_z','lat_aux_grid'], \
      coords={'time':time,'transport_regions':transport_regions,'moc_z':mocz,'lat_aux_grid':lat_aux_grid}, \
      name='MOC')
else:
   MOCnew = xr.DataArray(np.zeros((nt,ntr,mocnz,nyaux),dtype=np.single),dims=['time','transport_reg','sigma','lat_aux_grid'], \
      coords={'time':time,'transport_regions':transport_regions,'sigma':mocz,'lat_aux_grid':lat_aux_grid}, \
      name='MOC')
MOCnew.values[:,:,:,:] = np.transpose(tmpmoc_e,axes=[3,2,1,0])
#print('mocnewshape',np.shape(MOCnew))
MOCnew = MOCnew.where(MOCnew<mval).cumsum(dim='lat_aux_grid')
MOCnew = MOCnew*1.0e-6
MOCnew.attrs={'units':'Sverdrups','long_name':'Meridional Overturning Circulation'}
MOCnew.encoding['_FillValue']=mval
time10=timer.time()
print('Timing:  meridional integration =  ',time10-time9,'s')

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
#tmp=veflux
#tmpw=tmp.roll(nlon=1,roll_coords=False)        
#tmpall=xr.concat([tmp,tmpw],dim='dummy')
#veflux=tmpall.where(tmpall<1e30).mean('dummy')
#del tmp,tmpw,tmpall
veflux = veflux.where(veflux<1e30)
#    c. zonal integral of Atlantic points
atlmask=xr.DataArray(np.where(rmask==6,1,0),dims=['nlat','nlon'])
atlmask=atlmask.roll(nlat=-1,roll_coords=False)
veflux_xint=veflux.where(atlmask==1).sum(dim='nlon')
if not sigmacoord:
  amoc_s_e=-veflux_xint[0,::-1,lat_aux_atl_start].cumsum(dim='z_t')
else:
  amoc_s_e=-veflux_xint[0,::-1,lat_aux_atl_start].cumsum(dim='sigma')
amoc_s_e = amoc_s_e[::-1]*1.e-6
MOCnew.values[:,1,0:mocnz-1,:] = MOCnew.values[:,1,0:mocnz-1,:] + amoc_s_e.values[None,:,None]
time11=timer.time()
print('Timing:  Atl southern boundary stuff =  ',time11-time10,'s')


#8.    Write output to netcdf
out_ds=MOCnew.to_dataset(name='MOC')
out_ds.to_netcdf(out_file,unlimited_dims='time')
if append_to_infile:
   cmd = ['ncks','-A','-h','-v','MOC',out_file,in_file]
   subprocess.call(cmd)
   cmd = ['rm','-f',out_file]
   subprocess.call(cmd)
time12=timer.time()
print('Timing:  writing output =  ',time12-time11,'s')

time13=timer.time()
print('DONE creating ',out_file,'.  Total time = ',time13-time1,'s')
