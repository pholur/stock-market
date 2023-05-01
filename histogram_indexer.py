from helpers_correlation import *
from CONSTANTS import *
import numpy as np
import pickle
from tqdm import tqdm

if __name__ == "__main__": # only does yearly data
    for year in tqdm(years[:]):
        for tau in tqdm(taus):
            with open(f"{path}/Correlations/corr_res_{year}_{tau}.pkl", "rb") as f:
                correlates = pickle.load(f)
            
            for mode in ["raw_correlations", "minus_market_mode", "normalized"]:
                vals = correlates[mode]
                
                # print(vals.flatten().shape, np.tril(vals).flatten().shape)
                # exit()
                
                indices = np.triu_indices_from(vals, k=1)
                real_series = vals[indices]
                
                # print(vals.shape, len(real_series))
                # exit()
                
                response = histogram(real_series) # only bottom diagonal
                with open(f"{path}/Histograms/hist_{mode}_{year}_{tau}.pkl", "wb") as f:
                    pickle.dump(response, f, protocol=pickle.HIGHEST_PROTOCOL)