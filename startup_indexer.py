from functools import partial
import numpy as np
from sklearn import mixture
import pickle

from statistics import mode
import numpy as np

import os
import pandas as pd
from tqdm import tqdm
from CONSTANTS import *


def process_time_range(file_, time_start = "09:30:00", time_end = "16:00:00"):
    # convert times to datetime
    time_start = pd.to_datetime(time_start, format="%H:%M:%S")
    time_end = pd.to_datetime(time_end, format="%H:%M:%S")

    df = pd.read_csv(file_, header=None, names = ["date", "price"])
    # print(df["date"].head())
    # print(file_)
    
    try:
        df[['date','time']] = df["date"].str.split(" ", 1, expand=True)
    except: # dataframe is consistently empty. Ignore.
        raise KeyError("Dataframe is empty.")
    
    df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S")
    
    if len(df) < minimum_minute_samples:
        raise ValueError("Size of dataframe is too small.")
    # Add missing minutes
    # better job of interpolating missing minutes
    idx = pd.date_range(start=df['time'].min(), 
                        end=df['time'].max(), freq='min')

    df = df.set_index('time').reindex(idx).interpolate().reset_index()
    # filter time range
    df = df[df["index"] >= time_start]
    df = df[df["index"] <= time_end]
    
    # extract prices
    price_list = [float(price) for price in list(df["price"])]
    
    if max(price_list) == 0:
        raise ValueError("Check the price series data.")
        
    return price_list







def extract_data(path):
    subdirs = []
    for root, subFolders, _ in os.walk(path):
        subdirs = [os.path.join(root, subFolder) for subFolder in subFolders]
        break
    # print(subdirs) # list of stock index paths per year

    file_names = []
    total_data = {}

    for subdir in tqdm(subdirs[:]): # you might have to change this parameter, for each stock  # noqa: E501
                
        for root, subFolders, file_name in os.walk(subdir): # this only runs once
            
            total_series = []

            file_names = [os.path.join(root, file) for file in file_name]
            file_names.sort()
            # print(file_names) # per year, per stock all dates.
            
            real_file_count = 0
            bad_file = False
            
            for file in file_names: # for each day
                try:
                    series = process_time_range(file)
                    total_series.extend(series)
                    real_file_count += 1
                except ValueError:
                    bad_file = True
                    break
                except KeyError:
                    continue # accept the error and count the other days
            
            if bad_file: # bad stock
                continue
            
            total_data[subdir.split("/")[-1]] = total_series
            break
        # except:
        #     pass
    
    # incomplete dates
    mode_length: int
    lengths = []
    for k,v in total_data.items():
        lengths.append(len(v))
    mode_length = max(lengths, key=lengths.count)
    
    returned_list = []
    for k,v in total_data.items():
        if len(v) == mode_length:
            returned_list.append(v)
    return returned_list



for year in tqdm(years[-3:]):
    # try:
        data = extract_data(path=f"/mnt/SSD2/miyahara/mst_wrds_003/z_output_raw_1Min/z_output_raw_1Min_{year}/price") 
        data_numpy = np.array(data)
        np.save(f"{path}data_{year}.npy", data_numpy)
    # except:
    #     pass
        # print(year)

# import multiprocessing
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from tqdm import tqdm

# def extract_and_save_data(year):
#     try:
#         data = extract_data(path=f"/mnt/SSD2/miyahara/mst_wrds_003/z_output_raw_1Min/z_output_raw_1Min_{year}/price")
#         data_numpy = np.array(data)
#         np.save(f"{path}data_{year}.npy", data_numpy)
#     except:
#         pass

# if __name__ == '__main__':
#     with ThreadPoolExecutor(max_workers=3) as executor:
#         futures = [executor.submit(extract_and_save_data, year) for year in years]
#         for future in tqdm(as_completed(futures), total=len(futures)):
#             pass
