import sys
sys.path.append("../")

from Utils.helpers import *
from CONSTANTS import *
import numpy as np
import pickle
from tqdm import tqdm

if __name__ == "__main__": # only does yearly data
    for year in tqdm(years[:]):
        mat = np.load(f"{path}Preprocessed/data_{year}.npy", allow_pickle=True)
        for tau in tqdm(taus):
            response = correlation(mat, tau)
            with open(f"{path}Correlations/corr_res_{year}_{tau}.pkl", "wb") as f:
                pickle.dump(response, f, protocol=pickle.HIGHEST_PROTOCOL)


# BUGGY - REQUIRES FIX FOR MPI
# import numpy as np
# import multiprocessing as mp
# from tqdm import tqdm
# import pickle

# def compute_corr(year, tau, path):
#     mat = np.load(f"{path}Preprocessed/data_{year}.npy", allow_pickle=True)
#     response = correlation(mat, tau)
#     with open(f"{path}Correlations/corr_res_{year}_{tau}.pkl", "wb") as f:
#         pickle.dump(response, f, protocol=pickle.HIGHEST_PROTOCOL)


# pool = mp.Pool(processes=3)
# results = [pool.starmap(compute_corr, 
#                         [(year, tau, path) for tau in tqdm(taus)]) 
#            for year in tqdm(years)]
