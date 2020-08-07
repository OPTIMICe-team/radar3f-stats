import xarray as xr
import numpy as np

filepath = '/net/broebroe/lvonterz/tripex_pol/output/2019/01/22/'
filename = filepath + '20190122_00_tripex_pol_poldata_L0_spec_regridded.nc'
data = xr.open_dataset(filename)

allna = data.sZDR.isnull().all(dim='Vel') # get the time-height of allNan
idx = data.sZDR.where(~allna, 0).argmax(dim='Vel') # temporarely fill the allNans with 0 and get the idxmaxs
idx = idx.where(~allna, np.nan) # perhaps instead of 0 you can fill again with NaNs the time-height where you had all nans
