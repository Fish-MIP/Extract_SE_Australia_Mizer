#!/usr/bin/python

## Extracting data for South East Australia - Mizer model from DKRZ server
## Date: 2023-11-07
## Author: Denisse Fierro Arcos

#Libraries
import numpy as np
import xarray as xr
import pandas as pd
from glob import glob
import os
import re
import itertools

#######################################################################################
#Base directory where outputs will be saved
base_out = 'Data_Extraction'
#Ensuring base directory exists
os.makedirs(base_out, exist_ok = True)

#Base directory where data is currently stored
base_dir = ['/work/bb0820/ISIMIP/ISIMIP3a/InputData/climate/ocean/obsclim/global/monthly/historical/GFDL-MOM6-COBALT2/',
            '/work/bb0820/ISIMIP/ISIMIP3a/InputData/climate/ocean/ctrlclim/global/monthly/historical/GFDL-MOM6-COBALT2/']

#Indicating location of masks
#1 degree - Subsetting to SE Australia
mask_1deg = xr.open_dataset('SE_Australia_Mizer_mask_1deg.nc').SE_Aust
mask_1deg = mask_1deg.rename({'latitude': 'lat', 'longitude': 'lon'}).sel(lat = slice(-30, -50), 
                                                                          lon = slice(120, 160))
#0.25 degree - Subsetting to SE Australia
mask_025deg = xr.open_dataset('SE_Australia_Mizer_mask_025deg.nc').SE_Aust
mask_025deg = mask_025deg.rename({'latitude': 'lat', 'longitude': 'lon'}).sel(lat = slice(-30, -50), 
                                                                              lon = slice(120, 160))

#Variables of interest
var_list = ['phyc-vint', 'phypico-vint', 'tos', 'tob', 'expc-bot', 'deptho', 'thetao_15']

###### Defining useful functions ######

## Loading data with dates that are not CF compliant
def load_ds_noncf(fn):
    '''
    This function loads non-CF compliant datasets where dates cannot be read. It needs one input:
    fn - ('string') refers to full filepath where the non-CF compliant dataset is located
    '''
    #Loading dataset without decoding times
    ds = xr.open_dataset(fn, decode_times = False)
    
    #Checking time dimension attributes
    #Extracting reference date from units 
    init_date = re.search('\\d{4}-\\d{1,2}-\\d{1,2}', ds.time.attrs['units']).group(0)
    
    #If month is included i the units calculate monthly timesteps
    if 'month' in ds.time.attrs['units']:
      #If month values include decimals, remove decimals
      if ds.time[0] % 1 != 0:
        ds['time'] = ds.time - ds.time%1
      #From the reference time, add number of months included in time dimension
      try:
        new_date = [pd.Period(init_date, 'M')+pd.offsets.MonthEnd(i) for i in ds.time.values]
        #Change from pandas period to pandas timestamp
        new_date =[pd.Period.to_timestamp(i) for i in new_date]
      #If any errors are encountered
      except:
        #If dates are before 1677, then calculate keep pandas period
        new_date = pd.period_range(init_date, periods = len(ds.time.values), freq ='M')
        #Add year and month coordinates in dataset
        ds.coords['year'] = ('time', new_date.year.values)
        ds.coords['month'] = ('time', new_date.month.values)
    
    #Same workflow as above but based on daily timesteps
    elif 'day' in ds.time.attrs['units']:
      if ds.time[0] % 1 != 0:
        ds['time'] = ds.time - ds.time%1
      try:
        new_date = [pd.Period(init_date, 'D')+pd.offsets.Day(i) for i in ds.time.values]
        new_date =[pd.Period.to_timestamp(i) for i in new_date]
      except:
        new_date = pd.period_range(init_date, periods = len(ds.time.values), freq ='D')
        ds.coords['year'] = ('time', new_date.year.values)
        ds.coords['month'] = ('time', new_date.month.values)
    
    #Replace non-cf compliant time to corrected time values
    ds['time'] = new_date
    return ds


## Extracting surface data
def masking_data(ds, var_int, mask, path_out):
  #Getting name of variables available in dataset
  var_ds = list(ds.keys())
  #If there are multiple variable, keep variable that is similar to variable in file name
  if len(var_ds) > 1:
    var_ds = [v for v in var_ds if v in var_int][0]
  else:
    var_ds = var_ds[0]
  #Extracting only variable of interest around SE Australia
  try:
    ds = ds[var_int].sel(lat = slice(-30, -50), lon = slice(120, 160))
  except:
    ds = ds[var_ds].sel(lat = slice(-30, -50), lon = slice(120, 160))
   
  #Applying regional mask
  ds_time = []
  for t, da in ds.groupby('time.year'):
    #Calculating  mean
    sub = xr.where(mask == 1, da, np.nan)
    ds_time.append(sub.mean(('lat', 'lon', 'time')))
  
  #Concatenating data back together
  ds_time = xr.concat(ds_time, dim = 'year')
  
  #Saving csv
  ds_time.to_pandas().to_csv(path_out, na_rep = np.nan)  


###### Applying functions to all files in directories of interest ######
###Loop through each directory
for bd in base_dir:
  #Find netcdf files for expriments and models of interest
  all_files = [glob(os.path.join(bd, f'*{var}*.nc')) for var in var_list]
  file_list = list(itertools.chain(*all_files))

  #Loop through variables of interest
  for dp in file_list:
    #Find the correct folder to store files
    dir_out =  base_out + "/" + re.split(bd, dp)[-1]
    ###Loop through each file
    var_int = re.split('_\\d{2,3}a', re.split('obsclim_', dp)[-1])[0]
    #Extracting base file name to create output
    base_file = re.split('/GFDL-MOM6-COBALT2/', 
                         dp)[-1].replace('global', 'SouthEastAustralia').replace('.nc', '.csv')
    path_out = os.path.join(base_out, base_file)
    
    #Loading data
    try:
        ds = xr.open_dataset(dp)
    except:
        print('Time in historical data is not cf compliant. Fixing dates based on years in file name.')
        try:
          ds = load_ds_noncf(dp)
        except:
          print(f'Time could not be decoded for file: {dp}')
          pass
    
    #Masking data and calculating means
    try:
      ds is not None
      try:
        if '60arcmin' in dp:
          mask = mask_1deg
        if '15arcmin' in dp:
          mask = mask_025deg
        #Extracting data
        masking_data(ds, var_int, mask, path_out)
        del ds
      except:
        print(f'File could not be processed: {f}')
        pass
    except:
      pass
      

