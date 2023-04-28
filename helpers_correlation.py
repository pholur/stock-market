import numpy as np
from CONSTANTS import *



# Compute the KL divergence
def kl(p, q, gap):
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    return np.sum(np.where(p != 0, gap * p * np.log(p+SMALL_CONSTANT / (q+SMALL_CONSTANT)), 0))



# fast correlation computation
def correlation(trajectory, tau=None, precision=precision):
    '''
    <<Computes the correlation of a trajectory across particles>>
    trajectory:     stocks x timesteps --> can be slightly different for different years
    '''

    dim_y = trajectory.shape[1]
    r_i_t_tau = np.log(trajectory[:,0+tau:dim_y]) - np.log(trajectory[:,:dim_y-tau])
    r_o_t_tau = np.mean(r_i_t_tau, axis=0)
    total_returns = np.vstack((r_o_t_tau, r_i_t_tau))
    correlations_i_j_tau_plus = np.corrcoef(total_returns)
    correlations_i_o_tau = correlations_i_j_tau_plus[1:,1:] # symmetric
    plus = correlations_i_j_tau_plus[:1,1:].reshape(-1,1) * correlations_i_j_tau_plus[:1,1:].reshape(1,-1)
    c_tau = correlations_i_o_tau - plus
    
    normed =  np.sqrt(1 - np.square(correlations_i_j_tau_plus[:1,1:]))
    c_tau_norm = np.divide(c_tau,normed.reshape(-1,1), normed.reshape(1,-1))

    c_tau_raw = correlations_i_o_tau
    d_tau = np.sqrt(2*(1-correlations_i_o_tau))

    return {"raw_correlations":np.round(c_tau_raw,precision), 
            "minus_market_mode":np.round(c_tau,precision), 
            "distances":np.round(d_tau,precision), 
            "market":np.round(plus,precision),
            "normalized": np.round(c_tau_norm,precision)
    }


# compute the histogram of a series
def histogram(array_rho, range_min = histogram_limits[0], 
              range_max = histogram_limits[1], bin_size = histogram_bins):

    mean_c_ij = np.mean(array_rho) # CHECK: This computes the absolute mean not the histogram mean
    var_c_ij = np.var(array_rho) # CHECK: Same
    sigma_cij = np.sqrt(var_c_ij) # CHECK: Same

    hist_array = np.histogram(array_rho, bins = np.linspace(range_min,range_max,bin_size), range = (range_min, range_max))
    prob_array = len(np.linspace(range_min,range_max,bin_size)) / (range_max - range_min) / sum(hist_array[0]) * hist_array[0] # CHECK: What is this?

    x_original_cij = []
    y_original_cij = []
    
    for i in range(len(hist_array[0])):
        x_original_cij.append(hist_array[1][i])
        y_original_cij.append(prob_array[i])

    x_scaled_cij = []
    y_scaled_cij = []
    for i in range(len(hist_array[0])):
        x_scaled_cij.append(hist_array[1][i] / sigma_cij - mean_c_ij / sigma_cij)
        y_scaled_cij.append(prob_array[i] * sigma_cij) # CHECK: sigma is multiplied here?
        
    return {
        "x": x_scaled_cij, 
        "y": y_scaled_cij, 
        "mu": mean_c_ij,
        "var": var_c_ij, 
        "sigma": sigma_cij
    }



# gmm fit 2-mode
from sklearn import mixture
def gmm_fit(series, n_components = gmm_components):
    g = mixture.GaussianMixture(n_components=n_components,covariance_type='spherical')
    g.fit(series)
    covar = g.covariances_
    std = np.sqrt(covar)
    means = g.means_
    weights = g.weights_
    return {
        "mus": means, 
        "stds": std, 
        "weights": weights
    }