from helpers_correlation import *
from CONSTANTS import *
import numpy as np
import pickle
from tqdm import tqdm

if __name__ == "__main__": # only does yearly data
    for year in tqdm(years):
        for tau in tqdm(taus):
            with open(f"{path}/Correlations/corr_res_{year}_{tau}.pkl", "rb") as f:
                correlates = pickle.load(f)
            
            for mode in ["raw_correlations", "minus_market_mode", "normalized"]:
                vals = correlates[mode]
                response = histogram(np.tril(vals).flatten())
                
                with open(f"{path}/Histograms/hist_{mode}_{year}_{tau}.pkl", "wb") as f:
                    pickle.dump(response, f, protocol=pickle.HIGHEST_PROTOCOL)