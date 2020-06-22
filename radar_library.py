import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
from os import path

# some basics 

tripex_date_start = datetime(2015, 11, 11)
tripex_date_end = datetime(2016, 1, 4, 23, 59, 59)

tripexpol_date_start = datetime(2018, 11, 1)
tripexpol_date_end = datetime(2019, 2, 20, 23, 59, 59)

root_tripex = '/data/optimice/tripex/'
root_tripex_pol = '/data/obs/campaigns/tripex-pol/processed/'
root_pluvio = '/data/data_hatpro/jue/data/pluvio/netcdf/'

# Translate common variable names to campaign specific
tripex_vars = {'Z10':'dbz_x', 'Z35':'dbz_ka', 'Z94':'dbz_w', # reflectivity
               'time':'time', 'range':'range', # dimensions
               'T':'ta', 'P':'pa', 'RH':'hur', # thermodynamics
               'W10':'sw_x', 'W35':'sw_ka', 'W94':'sw_x', # spectral width
               'V10':'rv_x', 'V35':'rv_ka', 'V94':'rv_w', # mean doppler velocity
               'quality_x':'quality_flag_offset_w', 'quality_w':'quality_flag_offset_w',
               'S35':'sk', # skweness
               'RR':'r_accum_NRT' # rain rate from pluvio needs processing
              }
tripex_pol_vars = {'Z10':'X_DBZ_H', 'Z35':'Ka_DBZ_H', 'Z94':'W_DBZ_H', # reflectivity
                   'time':'time', 'range':'range', # dimensions
                   'T':'ta', 'P':'pa', 'RH':'hur', # thermodynamics
                   'W10':'X_WIDTH_H', 'W35':'Ka_WIDTH_H', 'W94':'W_WIDTH_H', # spectral width
                   'V10':'X_VEL_H', 'V35':'Ka_VEL_H', 'V94':'W_VEL_H', # mean doppler velocity
                   'quality_x':'quality_flag_offset_w', 'quality_w':'quality_flag_offset_w',
                   'S35':'Ka_SK_H', #skweness
                   'RR':'r_accum_NRT' # rain rate from pluvio needs processing
                  }                 
available_vars = [z for z in tripex_vars.keys()]

# Pluvio files are always loaded
def pluvio_filepath(date):
    dtstr = date.strftime('%Y%m%d')
    if date in pd.date_range(tripex_date_start, tripex_date_end):
        return root_pluvio + dtstr[2:-2] + '/pluvio2_jue_' + dtstr + '.nc'
    elif date in pd.date_range(tripexpol_date_start, tripexpol_date_end):
        return root_pluvio + dtstr[2:-2] + '/pluvio2_jue_' + dtstr + '.nc'
    else:
        return 'NOT_EXISTING_PATH...hopefully...otherwise,we,are,screwed'


def tripex_filepath(date):
    dtstr = date.strftime('%Y%m%d')
    return root_tripex + 'tripex_level_02_X_pol/tripex_joy_tricr00_l2_any_v00_' + dtstr + '000000.nc'
           

# Would be nice to add skweness to main tripex.nc files so we have unique resource
def tripex_sk_filepath(date):
    dtstr = date.strftime('%Y%m%d')
    return root_tripex + 'skewness/resampled/' + dtstr + '_sk_joyrad35.nc'


def tripol_filepath(date):
    dtstr = date.strftime('%Y%m%d')
    return root_tripex_pol + 'tripex_pol_level_2/' + dtstr + '_tripex_pol_3fr_L2_mom.nc'


# Define and efficient function that does running mean along one axis of a 2D array
def running_mean_2d(xx, N, minN=0, weights=None):
    if weights is None:
        weights = xx/xx
    xx = xx*weights
    add = int((N - (N & 1))/2)
    addition = np.zeros([xx.shape[0], add])*np.nan
    x = np.concatenate([addition, xx, addition], axis=1)
    w = np.concatenate([addition, weights, addition], axis=1)
    csumnan = np.cumsum((~np.isfinite(np.insert(x, 0, 0, axis=1))).astype(int),
                        axis=1)
    nannum = csumnan[:, N:] - csumnan[:, :-N]
    mask = ((N-nannum) >= minN)
    Filter = mask.astype(float)
    Filter[~mask] = np.nan
    csum = np.nancumsum(np.insert(x, 0, 0, axis=1), axis=1)
    wsum = np.nancumsum(np.insert(w, 0, 0, axis=1), axis=1)
    return Filter * (csum[:, N:] - csum[:, :-N]) / (wsum[:, N:] - wsum[:, :-N])


def Bd(x): # return linear if dB are given
    return 10.0**(0.1*x)


def dB(x): # return dB if linear are given
    return 10.0*np.log10(x)


#####################################################################################################################
# Functions to open the campaign data
#####################################################################################################################

# Function to open pluvio files and resample them on the 4s frequency of the tripex files
def open_resample_pluvio(dt_start, dt_end, accMins):
    pluvio_files = [pluvio_filepath(dt) for dt in pd.date_range(dt_start, dt_end) if path.exists(pluvio_filepath(dt))]
    if len(pluvio_files):
        save_vars = ['time', 'r_accum_NRT']
        x = xr.open_dataset(pluvio_files[0])
        varlist = list(x.variables)
        drops = [v for v in varlist if v not in save_vars]
        pluvio = xr.open_mfdataset(pluvio_files, concat_dim='dim',
                                   data_vars='minimal',
                                   drop_variables=drops).rename({'dim':'time'}).set_coords('time')
        pluvio = pluvio.loc[{'time':~pluvio.time.isnull()}] # eliminate missing times
        
        ############################################################
        # Resampling is very slow on this version of xarray (0.10) #
        ############################################################
        # Let's try old school pandas
        PLUVIO = pd.DataFrame(data=pluvio.r_accum_NRT.to_masked_array(),
                              index=pluvio.time.to_masked_array(),
                              columns=['RR'])
        PLUVIO = PLUVIO.resample(str(accMins)+'min').apply(np.nansum)*60/accMins # prbably better to do resampling and rolling
        PLUVIO = PLUVIO.resample('4s').nearest()
        PLUVIO.index.name='time'
        
        return xr.Dataset.from_dataframe(PLUVIO)
    return None


def generate_preprocess(translate, avgV):
    '''
    This is where dataset preprocess happens.
    I am still not sure if it is faster done like this or outside using lazy evaluation...
    '''
    def preprocess(ds):
        # First I find convenient to rename the variables early, so I do not have to deal with translations
        ds = ds.rename({i:v for (v,i) in translate.items() if i in ds.variables})
        # Second, it is very important to reorder the dimensions
        ds = ds.transpose('range', 'time') # this is robust with respect of the order used in the files
        if avgV: # perform moving window average for velocities
            Vvars = [v for v in ['V10', 'V35', 'V94'] if v in ds.variables]
            for m in avgV:
                steps = int(m*60/4) # I am always considering 4 seconds resampling
                steps = steps if steps % 2 else steps-1 # ensure it is odd number
                valid_steps = int(steps/5) # kind of arbitrary
                for v in Vvars:
                    ds[v+'m'+str(m)] = ds[v]*0.0 + running_mean_2d(ds[v], steps, valid_steps,
                                                                   Bd(ds['Z'+v[1:]]))
        return ds
    return preprocess


##############################################################################################################
# THIS IS THE MAIN FUNCTION
##############################################################################################################
def read_tripex(dt0, dt1, variables, accMins=5, avgV=[]):
    # always include dimensions in the list of variables to keep
    variables = list(set(variables) | set(['range', 'time']))

    # List of files to be loaded
    tripex_files = [tripex_filepath(dt) for dt in pd.date_range(dt0, dt1) if path.exists(tripex_filepath(dt))]
    tripsk_files = [tripex_sk_filepath(dt) for dt in pd.date_range(dt0, dt1) if path.exists(tripex_sk_filepath(dt))]
    tripol_files = [tripol_filepath(dt) for dt in pd.date_range(dt0, dt1) if path.exists(tripol_filepath(dt))]
    pluvio_files = [pluvio_filepath(dt) for dt in pd.date_range(dt0, dt1) if path.exists(pluvio_filepath(dt))]
    ##########################################################################################################
    # OPEN TRIPEX DATA 
    ##########################################################################################################
    if len(tripex_files): # open only if requested
        # open tripex
        save_vars = [tripex_vars[v] for v in variables]
        x = xr.open_dataset(tripex_files[0])
        varlist = list(x.variables)
        drops = [v for v in varlist if v not in save_vars]
        tripex = xr.open_mfdataset(tripex_files,
                                   concat_dim='time',
                                   data_vars='minimal',
                                   preprocess=generate_preprocess(tripex_vars, avgV), # drop_variables is called first
                                   drop_variables=drops)
        # open skweness of tripex and add it to the main tripex dataset
        if 'S35' in variables:
            tripex = xr.merge([tripex,
                               xr.open_mfdataset(tripsk_files,
                                                 preprocess=generate_preprocess(tripex_vars),
                                                 concat_dim='time')])
        #tripex.rename({i:v for (v,i) in tripex_vars.items() if i in tripex.variables}, inplace=True)
        if 'RR' in variables:
            tripex = xr.merge([tripex,
                               open_resample_pluvio(max([dt0, tripex_date_start]),
                                                    min([dt1, tripex_date_end]),
                                                   accMins)
                              ])        

    ###########################################################################################################
    # OPEN TRIPEX-POL  DATA 
    ###########################################################################################################
    if len(tripol_files): # open only if requested
        # open tripex-pol
        save_vars = [tripex_pol_vars[v] for v in variables]
        x = xr.open_dataset(tripol_files[0])
        varlist = list(x.variables)
        drops = [v for v in varlist if v not in save_vars]
        tripol = xr.open_mfdataset(tripol_files,
                                   concat_dim='time',
                                   data_vars='minimal',
                                   preprocess=generate_preprocess(tripex_pol_vars),
                                   drop_variables=drops)
        #tripol.rename({i:v for (v,i) in tripex_pol_vars.items() if i in tripol.variables}, inplace=True)
        if 'RR' in variables:
            tripol = xr.merge([tripol,
                               open_resample_pluvio(max([dt0, tripexpol_date_start]),
                                                    min([dt1, tripexpol_date_end]),
                                                    accMins)
                              ])

        try: # try to concatenate to tripex if existing
            tripex = xr.concat([tripex, tripol],
                               dim='time')
        except NameError: # otherwise initialize it
            tripex = tripol


    ###########################################################################################################
    # OPEN NEW TRIPEX CAMPAIGN ... 
    ###########################################################################################################

    # Following the same pattern we can open a new campaign data and concatenate it to tripex xr.dataset
    #
    #

    # Let's do one last dimension reordering, just in case I stop doing preprocessing at a certain point
    tripex = tripex.transpose('range', 'time')
    
    return tripex