from helpers_correlation import *
from CONSTANTS import *
import numpy as np
import pickle
from tqdm import tqdm

if __name__ == "__main__": # only does yearly data
    for year in tqdm(years):
        mat = np.load(f"{path}/Preprocessed/data_{year}.npy", allow_pickle=True)
        for tau in tqdm(taus):
            response = correlation(mat, tau)
            with open(f"{path}/Correlations/corr_res_{year}_{tau}.pkl", "wb") as f:
                pickle.dump(response, f, protocol=pickle.HIGHEST_PROTOCOL)