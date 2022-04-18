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
    sig2 = np.concatenate((tmp1,tmp2,tmp3))
    nsig = len(sig2)
    # Define density of midpoint, top, and bottom of isopycnal layers
    sigma_mid=xr.DataArray(sig2,coords={'sigma':sig2},
                           attrs={'long_name':'Sigma2 at middle of layer','units':'kg/m^3'})
    sigma_edge = (sigma_mid+sigma_mid.shift(sigma=1))/2.
    sigma_edge[0] = 0.
    sigma_edge = np.append(sigma_edge.values,[50.])
    sigma_edge=xr.DataArray(sigma_edge,coords={'sigma':sigma_edge},
                           attrs={'long_name':'Sigma2 at edges of layer','units':'kg/m^3'})
    return sigma_mid,sigma_edge

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

def pop_isowflux(uflux_ne,vflux_ne,densdim,densedges):
    """ 
    Computes horizontal flux convergence for POP u,v volume fluxes in density-space.
    
    Parameters
    ----------
    uflux_ne : array of Grid-zonal volume flux at U-point (m^3/s)
    vflux_ne : array of Grid-meridional volume flux at U-point (m^3/s)
    densdim : string
        name of density dimension
    densedges : array of density coordinates corresponding to layer tops
        
    Returns
    -------
    wflux : array of vertical volume flux in density coordinates (densedges) at T-point (m^3/s)
    """
    
    # Fluxes on Tracer-grid corners:
    uflux_nw = uflux_ne.roll(nlon=1,roll_coords=False)
    vflux_nw = vflux_ne.roll(nlon=1,roll_coords=False)
    uflux_sw = uflux_nw.shift(nlat=1)
    vflux_sw = vflux_nw.shift(nlat=1)
    uflux_se = uflux_ne.shift(nlat=1)
    vflux_se = vflux_ne.shift(nlat=1)
    
    # Fluxes on Tracer-grid faces:
    uflux_e = 0.5*(uflux_ne + uflux_se)
    uflux_w = 0.5*(uflux_nw + uflux_sw)
    vflux_n = 0.5*(vflux_ne + vflux_nw)
    vflux_s = 0.5*(vflux_se + vflux_sw)
    
    # Convergence (= -Divergence) gives dW/dz:
    dwflux = -(uflux_e-uflux_w+vflux_n-vflux_s)
    
    # Bottom-up Vertical Integral to compute W:
    kwargs = {densdim:slice(None,None,-1)}
    wflux = dwflux.sel(kwargs).cumsum(densdim).sel(kwargs)
    kwargs = {densdim:slice(0,-1)}
    wflux['sigma'] = densedges.isel(kwargs)
    
    return wflux

def mesh_zonalavg(da_in, grid_area, grid_lat, rmask, rmaskdict, lat_edge, sum=False):
    """
    Uses xhistogram to calculate zonal averages of a field on a mesh grid (2D lat/lon), 
    mapping onto a target latitude grid (lat_targ). Separate averages are computed for each integer 
    value in rmask (described by rmaskdict). Set sum==True to compute zonal sum instead of average.
    
    Input
    ----------
    da_in : xarray.DataArray
       DataArray to calculate a zonal average from. This should be your data variable.
    grid_area : xarray.DataArray
       Grid area field, matching horizontal dims of da_in
    grid_lat : xarray.DataArray
       Grid latitude field, matching horizontal dims of da_in
    rmask: xarray.DataArray
       DataArray containing region mask information (integers>0)
    rmaskdict: dictionary
       Dictionary that relates region mask values to region description strings
    lat_edge : xarray.DataArray
       Latitude axis to use for latitude binning (edge values)
    sum : logical
       False==>compute zonal average. True==>compute zonal sum.
       
    Returns
    -------
    da_out : xarray.DataArray
       Resultant zonally averaged field, with the same input name and a new latitude bin axis
    """
    grid_dims = list(grid_area.dims)
    area = xr.ones_like(da_in)*grid_area.where(~da_in.isnull())
    lat = xr.ones_like(da_in)*grid_lat.where(~da_in.isnull())
    lat.name = 'latitude'
    zmlist = []
    # Iterate over region mask regions:
    for i in rmaskdict:
        if i==0:
            da_masked = da_in.where(rmask>0)
            area_masked = area.where(rmask>0)
            lat_masked = lat.where(rmask>0)
        else:
            da_masked = da_in.where(rmask==i)
            area_masked = area.where(rmask==i)
            lat_masked = lat.where(rmask==i)
        
        if sum:
            # histogram-binned field:
            histdata = histogram(lat_masked, bins=[lat_edge.values], 
                             weights=da_masked.fillna(0), dim=grid_dims)
            zm = histdata.rename(da_in.name)
        else:
            # histogram-binned area weights:
            histarea = histogram(lat_masked, bins=[lat_edge.values], 
                             weights=area_masked, dim=grid_dims)
            # histogram-binned field:
            histdata = histogram(lat_masked, bins=[lat_edge.values], 
                             weights=(area_masked*da_masked).fillna(0), dim=grid_dims)
            zm = (histdata/histarea).rename(da_in.name)
        zm = zm.assign_coords({'region':rmaskdict[i]})
        zmlist.append(zm)
    da_out = xr.concat(zmlist,dim='region').rename({'latitude_bin':'lat'})
    da_out['lat'] = da_out['lat'].assign_attrs({'long_name':'latitude','units':'degrees_north'})
    return da_out
