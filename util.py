import numpy as np
import xarray as xr

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
