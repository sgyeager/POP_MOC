import xarray as xr               #netcdf multidim reading/writing/manipulation
import glob                       #globbing
import numpy as np                #numerics
import os                         #operating system commands
import subprocess
import time as timer
import pop_tools
import sys
import util

time1=timer.time()

# Set Default Options
append_to_infile=False  # True with "-a"

# Define input/output streams
popmocdir = '/glade/u/home/yeager/analysis/python/POP_MOC'
os.chdir(popmocdir)
moc_template_file = './moc_template.nc'
nargs=len(sys.argv)
in_file=sys.argv[-1]
out_file=in_file[:-2]+'N_HEAT.nc'
#in_file = '/glade/scratch/yeager/iHesp/B.E.13.B1850C5.ne120_t12.sehires38.003.sunway_02.pop.h.0125-06.nc'
#out_file=in_file[:-2]+'N_HEAT.nc'

if ('-a' in sys.argv[:]):
   append_to_infile=True

# Define N_HEAT, N_SALT templates
ds = xr.open_dataset(moc_template_file)
lat_aux_grid = ds['lat_aux_grid']
lat_aux_grid.encoding['_FillValue']=None        # because xarray is weird
transport_regions = ds['transport_regions']
transport_components = ds['transport_components']
ncomp = transport_components.shape[0]
ntr = transport_regions.shape[0]
nyaux = lat_aux_grid.shape[0]

# Open a POP history file requiring N_HEAT, N_SALT
ds = xr.open_dataset(in_file)
uet = ds['UET']
vnt = ds['VNT']
ues = ds['UES']
vns = ds['VNS']
ulat   = ds['ULAT']
ulat=ulat.drop(['TLAT','TLONG'])            # this is a python bug that we are correcting
tlat   = ds['TLAT']
tlat=tlat.drop(['ULAT','ULONG'])            # this is a python bug that we are correcting
tlat.values[np.isnan(tlat.values)]=300.     # to avoid error messages if TLAT has _FillValues
kmt  = ds['KMT']
kmt.values[np.isnan(kmt.values)]=0      # get rid of _FillValues
kmt.values[kmt<0]=0
rmask  = ds['REGION_MASK']
tarea  = ds['TAREA']
time   = ds['time']
time.encoding['_FillValue']=None        # because xarray is weird
z_t   = ds['z_t']
z_w_bot = ds['z_w_bot']
ht = ds['HT']
dz   = ds['dz']
cpsw = ds['cp_sw']       # erg/g/K
cpsw = cpsw/1.e7         # J/g/K
rhosw = ds['rho_sw']     # g/cm^3
tflux_factor = (rhosw*cpsw)/1.e15    #  PW/(degC*cm^3/s)
nt = np.shape(time)[0]
nz = np.shape(z_t)[0]
ny = np.shape(kmt)[0]
nx = np.shape(kmt)[1]
km = int(np.max(kmt).values)
mval=uet.encoding['_FillValue']

# Create ocean/land masks
kji = np.indices((nz,ny,nx))
kindices = kji[0,:,:,:] + 1
kmt3d = np.broadcast_to(kmt.values,kindices.shape)
ocean = (kindices <= kmt3d)
land = (ocean==False)
land_east = np.roll(land,-1,2)
land_north = np.roll(land,-1,1)
land_north[:,ny-1,:] = False

# Compute partial bottom cell thickness
dzt = util.pbc_dzt(dz,kmt,ht,z_w_bot,mval)

# Zero out missing values
uet.values[np.isnan(uet.values)]=0.0
vnt.values[np.isnan(vnt.values)]=0.0
ues.values[np.isnan(ues.values)]=0.0
vns.values[np.isnan(vns.values)]=0.0

# Zero out flux into/out of land (following Frank's mht.f90)
uet.values[np.broadcast_to(land,uet.shape)] = 0.0
uet.values[np.broadcast_to(land_east,uet.shape)] = 0.0
ues.values[np.broadcast_to(land,ues.shape)] = 0.0
ues.values[np.broadcast_to(land_east,ues.shape)] = 0.0
vnt.values[np.broadcast_to(land,vnt.shape)] = 0.0
vnt.values[np.broadcast_to(land_north,vnt.shape)] = 0.0
vns.values[np.broadcast_to(land,vns.shape)] = 0.0
vns.values[np.broadcast_to(land_north,vns.shape)] = 0.0

# grid-oriented tracer fluxes 
uet = uet*tarea*dzt   # degC*cm^3/s
vnt = vnt*tarea*dzt   # degC*cm^3/s
ues = ues*tarea*dzt   # (g/kg)*cm^3/s
vns = vns*tarea*dzt   # (g/kg)*cm^3/s

# unit conversion
uet = uet*tflux_factor   # PetaWatts
vnt = vnt*tflux_factor   # PetaWatts

# vertical integrals
hflux_e = uet.sum(dim='z_t')
hflux_n = vnt.sum(dim='z_t')
sflux_e = ues.sum(dim='z_t')
sflux_n = vns.sum(dim='z_t')

# flux divergence terms
hflux_div = hflux_e - hflux_e.roll(nlon=1,roll_coords=False)      
hflux_div = hflux_div + hflux_n - hflux_n.shift(nlat=1)
sflux_div = sflux_e - sflux_e.roll(nlon=1,roll_coords=False)      
sflux_div = sflux_div + sflux_n - sflux_n.shift(nlat=1)

# define transport regions
reg_glo = np.where(rmask>0,1,0)
reg_atl = np.where(rmask>=6,1,0) & np.where(rmask<=12,1,0)   # include Baltic for 0p1

# find starting j-index for Atlantic region
south_ind = ny
for n in range(1,ny):
    section = (reg_atl[n-1,:] == 1)
    if (section.any()):
       south_ind = n-2
       south_lat = ulat.values[n-2,0]
       break
print("south_ind= ",south_ind, "south_lat=",south_lat)

# Atlantic southern boundary transports
tmpmask = np.roll(reg_atl,-1,0)
hflux_n_xint = hflux_n.where(tmpmask==1).sum(dim='nlon')
mht0 = hflux_n_xint[:,south_ind]
sflux_n_xint = sflux_n.where(tmpmask==1).sum(dim='nlon')
mst0 = sflux_n_xint[:,south_ind]

# set up output arrays
MHT = xr.DataArray(np.zeros((nt,ntr,ncomp,nyaux),dtype=np.single),dims=['time','transport_reg','transport_comp','lat_aux_grid'], \
      coords={'time':time,'transport_regions':transport_regions,'transport_comp':transport_components,'lat_aux_grid':lat_aux_grid}, \
      name='N_HEAT',attrs={'long_name':'Northward Heat Transport','units':'Pwatt'})
MST = xr.DataArray(np.zeros((nt,ntr,ncomp,nyaux),dtype=np.single),dims=['time','transport_reg','transport_comp','lat_aux_grid'], \
      coords={'time':time,'transport_regions':transport_regions,'transport_comp':transport_components,'lat_aux_grid':lat_aux_grid}, \
      name='N_SALT',attrs={'long_name':'Northward Salt Transport','units':'gram centimeter^3/kg/s'})

# Initialize to fill value
MHT[:,:,:,:] = np.nan
MST[:,:,:,:] = np.nan

# loop over lat_aux_grid
lat_aux_atl_start = ny
for n in range(1,nyaux):
    regionmask = (tlat.values < lat_aux_grid.values[n] ) & (reg_glo==1)
    if (np.any(regionmask)):
        MHT.values[:,0,1,n] = hflux_div.where(regionmask==True).sum(dim='nlon').sum(dim='nlat')
        MST.values[:,0,1,n] = sflux_div.where(regionmask==True).sum(dim='nlon').sum(dim='nlat')
    regionmask = (tlat.values < lat_aux_grid.values[n] ) & (reg_atl==1)
    if (np.any(regionmask)):
        MHT.values[:,1,1,n] = hflux_div.where(regionmask==True).sum(dim='nlon').sum(dim='nlat')
        MST.values[:,1,1,n] = sflux_div.where(regionmask==True).sum(dim='nlon').sum(dim='nlat')
        if (lat_aux_grid.values[n] > south_lat):
            MHT.values[:,1,1,n] = MHT.values[:,1,1,n] + mht0
            MST.values[:,1,1,n] = MST.values[:,1,1,n] + mst0

# Write to netcdf
MHT.encoding['_FillValue']=1.e30
MST.encoding['_FillValue']=1.e30
ds = MHT.to_dataset(name='N_HEAT')
ds['N_SALT']=MST
ds.to_netcdf(out_file)
if append_to_infile:
   cmd = ['ncks','-A','-h','-v','N_HEAT,N_SALT,transport_regions,transport_comp',out_file,in_file]
   subprocess.call(cmd)
   cmd = ['rm','-f',out_file]
   subprocess.call(cmd)
   time2=timer.time()
   print('DONE appending to ',in_file,'.  Total time = ',time2-time1,'s')
else:
   time2=timer.time()
   print('DONE creating ',out_file,'.  Total time = ',time2-time1,'s')
