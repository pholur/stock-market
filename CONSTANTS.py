# discard a stock if incomplete
minimum_minute_samples = 225

# Number of years for evaluation
years = range(2004,2021) 

# investment horizon
taus = [1,5,10,20,40] + list(range(60,1600,50)) + list(range(1600,50000,1000))

# checkpoint path
path = "/mnt/SSD5/pholur/Stock_Market/"

# precision for MSTs and other histogram values
precision=2

# small constant
SMALL_CONSTANT = 1e-8

# large constant
LARGE_CONSTANT = 100000

# kl_div_limits - resamples
kl_div_limits = [-5, 5]
kl_bins = 500
resampling_variance = 0.2
rasampling_count = 100000

# histogram_plotter_limits
histogram_limits = [-1, 1]
histogram_bins = 200
histogram_plot_limits = [-4,4] # due to rescaling

# gmm_components
gmm_components = 2

# Plotting specifics
plotting_year_cutoff = 2012
plotting_tau_cutoff = 1000
plotting_tau_upper_bound = 30000