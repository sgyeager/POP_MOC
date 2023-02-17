import numpy as np
import xarray as xr
from xhistogram.xarray import histogram

def pbc_dzt(dz,kmt,ht,z_w_bot,mval):
   nz = np.shape(dz)[0]
   ny,nx = np.shape(kmt)
   dzt = np.zeros((nz,ny,nx)) + dz.values[:,None,None]
   for iz in range(0,nz):
      bottom = (kmt.values==(iz+1))
      belowbottom = (kmt.values<(iz+1))
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
   return dzt

def tx0p1v3_dztdzu(dz,kmt):
    dzt, dummy = xr.broadcast(dz, kmt)
    kidx = xr.DataArray(np.arange(1,len(dz)+1),dims=['z_t'])
    dzt = dzt.where(kidx<=kmt).fillna(0)
    dzt = dzt.drop_vars(['z_t','ULONG', 'ULAT', 'TLONG', 'TLAT'])
    dzbc = np.fromfile('/glade/p/cesmdata/cseg/inputdata/ocn/pop/tx0.1v3/grid/dzbc_pbc_s2.0_20171019.ieeer8',  dtype='>f8', count=-1)
    dzbc = dzbc.reshape(kmt.shape) / 100.
    pkmt = kmt.where(kmt == 0, kmt-1).astype('int').load()
    dzt.loc[dict(z_t=pkmt)] = dzbc
    dzt = dzt.assign_coords({'z_t':dz.z_t})
    dzt.attrs['units'] = 'meter'
    tmp=dzt.copy()
    tmp[:,0, :] = tmp[:,-1, ::-1]                   # tripole grid periodicity at the top
    tmpe=tmp.roll(nlon=-1,roll_coords=False)        # wraparound shift to west, without changing coords
    tmpn=tmp.roll(nlat=-1,roll_coords=False)        # wraparound shift to south, without changing coords
    tmpne=tmpn.roll(nlon=-1,roll_coords=False)      # wraparound shift to west, without changing coords
    tmpall=xr.concat([tmp,tmpe,tmpn,tmpne],dim='dummy')
    dzu=tmpall.min('dummy')
    dzu.attrs['units'] = 'meter'
    return dzt,dzu

def wflux_zonal_int(wflux,regionmask,lat):
    """ 
    Compute zonally-integrated vertical volume flux for each subdomain 
    defined by regionmask, using lat as the target latitude grid.
    """
    rmaskdict = regionmask.legend
    regions=xr.DataArray(list(rmaskdict.values()),dims=["transport_reg"],name="transport_reg")
    regionvals = list(rmaskdict.keys())
    ny = len(lat)
    field_lat = wflux.TLAT.data
    xrlist_all = []
    for key in rmaskdict:
        region = rmaskdict[key]
        xrlist = [wflux.sum(['nlon','nlat'])*0.]
        for i in range(ny-1):
            if (region=='Global'):
                latstrip = (field_lat>=lat.data[i]) & (field_lat<lat.data[i+1]) & (regionmask.data>key)
            else:
                latstrip = (field_lat>=lat.data[i]) & (field_lat<lat.data[i+1]) & (regionmask.data==key)
            xrlist.append(wflux.where(latstrip).sum(['nlon','nlat']))
        xrlist_all.append(xr.concat(xrlist,lat))
    xr_out = xr.concat(xrlist_all,regions).transpose("time","transport_reg","z_w_top",...)
    return xr_out 

def sigma2_grid_86L():
    """ 
    Defines an 86-layer sigma2 grid for MOC(sigma2) computation, returning mid-points and edge-points
    as xarrays (for use with xhistogram).
    """
    tmp1 = np.arange(28,35,0.2)
    tmp2 = np.arange(35,36,0.1)
    tmp3 = np.arange(36,38.05,0.05)
    sig2 = np.concatenate((tmp1,tmp2,tmp3)).astype('float32')
    nsig = len(sig2)
    # Define density of midpoint, top, and bottom of isopycnal layers
    sigma_mid=xr.DataArray(sig2,coords={'sigma':sig2},
                           attrs={'long_name':'Sigma2 at middle of layer','units':'kg/m^3'})
    sigma_edge = (sigma_mid+sigma_mid.shift(sigma=1))/2.
    sigma_edge[0] = 0.
    sigma_edge = np.append(sigma_edge.values,[50.])
    sigma_edge=xr.DataArray(sigma_edge,coords={'sigma':sigma_edge.astype('float32')},
                           attrs={'long_name':'Sigma2 at edges of layer','units':'kg/m^3'})
    return sigma_mid.astype('float32'),sigma_edge.astype('float32')

def latitude_grid_1deg():
    """ 
    Defines a 1-degree latitude grid for MOC(sigma2) computation, returning mid-points and edge-points
    as xarrays (for use with xhistogram).
    """
    midvals = np.arange(-89.5,90.5,1)
    edgevals = np.arange(-90,91,1)
    lat_mid=xr.DataArray(midvals,coords={'lat':midvals},
                           attrs={'long_name':'latitude','units':'degrees_north'})
    lat_edge=xr.DataArray(edgevals,coords={'lat':edgevals},
                           attrs={'long_name':'latitude','units':'degrees_north'})
    return lat_mid,lat_edge


def remove_inversions(sigma,zdim):
    """ 
    Removes density inversions from sigma field.
    """
    deltasig = 1.e-5
    km = sigma.sizes[zdim]
    for ik in range(km-1):
        k = km-(ik+2)
        kp1 = k+1
        sigma_kp1 = sigma.isel({zdim:kp1})
        sigma_k = sigma.isel({zdim:k})
        diff = sigma_k - sigma_kp1
        count = (diff>0).sum()
        sigma_k = xr.where(diff>0,sigma_kp1-deltasig,sigma_k)
        #print("comparing k-indices {} and {}".format(k,kp1)+".  found : {}".format(count.values))
        sigma = xr.where(sigma.coords[zdim]==sigma.coords[zdim][k],sigma_k,sigma)
    return sigma

def fluxdiv_B(uflux,vflux):
    # B-grid divergence
    # Assumes uflux=U*DY*DZ, vflux=V*DX*DZ
    UTE = 0.5*(uflux+uflux.shift(nlat=1))
    UTW = UTE.roll(nlon=1,roll_coords=False)
    VTN = 0.5*(vflux+vflux.roll(nlon=1,roll_coords=False))
    VTS = VTN.shift(nlat=1)
    fluxdiv = (UTE-UTW+VTN-VTS)
    return fluxdiv

def fluxdiv_C(uflux,vflux):
    # C-grid divergence
    # Assumes uflux=U*DY*DZ, vflux=V*DX*DZ
    UTE = uflux
    UTW = UTE.roll(nlon=1,roll_coords=False)
    VTN = vflux
    VTS = VTN.shift(nlat=1)
    fluxdiv = (UTE-UTW+VTN-VTS)
    return fluxdiv

def wflux(uflux,vflux,densdim,densedges,grid='B'):
    """ 
    Computes vertical volume flux given u,v volume fluxes in density-space.
    
    Parameters
    ----------
    uflux : array of Grid-zonal volume flux (m^3/s)
    vflux : array of Grid-meridional volume flux (m^3/s)
    densdim : string
        name of density dimension
    densedges : array of density coordinates corresponding to layer tops
    grid : 'B'==> U,V both at northeast corner of tracer cell
           'C'==> U at east face and V at north face of tracer cell
        
    Returns
    -------
    wflux : array of vertical volume flux in density coordinates (densedges) at T-point (m^3/s)
    """
    
    if (grid=='B'):
        dwflux = -fluxdiv_B(uflux,vflux)
    else:
        dwflux = -fluxdiv_C(uflux,vflux)
    
    # Bottom-up Vertical Integral to compute W:
    kwargs = {densdim:slice(None,None,-1)}
    wflux = dwflux.sel(kwargs).cumsum(densdim).sel(kwargs)
    kwargs = {densdim:slice(0,-1)}
    wflux['sigma'] = densedges.isel(kwargs)
    
    return wflux

def wflux_zonal_sum(wflux,regionmask,lat):
    """ 
    Compute zonally-integrated vertical volume flux using simple xhistogram binning by latitude.
    
    Parameters
    ----------
    wflux : array of vertical volume flux (m^3/s)
    regionmask : array of region masks (0 to exclude, 1 to include)
    lat : array defining target latitudes for meridional binning
        
    Returns
    -------
    wfluxzonalsum : array of zonally-integrated wflux for each region mask
    """
    wgts = (wflux*regionmask).astype('float32')
    
    # Use workaround for xhistogram bug (https://github.com/xgcm/xhistogram/pull/79). In future,
    # use keep_coords=True.
    xr_out = histogram(wflux.TLAT, bins=[lat.data],weights=wgts,dim=['nlat','nlon'],density=False)
    xr_out = xr_out.assign_coords(wgts.drop(['TLAT','TLONG']).coords)
    
    # Add zeros at southern edge in preparation for meridional integral:
    xr_out[{'TLAT_bin':0}] = 0
    xr_out = xr_out.rename({'TLAT_bin':lat.name})
    xr_out[lat.name] = lat[1:]
    #tmp = xr.zeros_like(xr_out.isel(TLAT_bin=0))
    #tmp['TLAT_bin'] = tmp['TLAT_bin'] - 1.
    #xr_out = xr.concat([tmp,xr_out],dim='TLAT_bin').rename({'TLAT_bin':lat.name})
    #xr_out[lat.name] = lat
    return xr_out 

def compute_MOC(wflux,regionmask,lat):
    """ 
    Use w-method to compute MOC.
    
    Parameters
    ----------
    wflux : array of vertical volume flux (m^3/s)
    regionmask : array of region masks (0 to exclude, 1 to include)
    lat : array defining target latitudes for meridional binning
        
    Returns
    -------
    MOC : array of MOC (Sv) fields for each region mask
    """
    # first compute wflux zonal sum binned by latitude using histogram
    zonsum = wflux_zonal_sum(wflux,regionmask,lat)
    
    # compute cumulative meridional sum (south to north)
    moc = zonsum.cumsum(dim=lat.name)
    
    # convert to Sv
    moc = moc/1.e6
    moc = moc.assign_attrs({'long_name':'Meridional Overturning Circulation','units':'Sv'})
    moc.name = 'MOC'
    
    return moc

