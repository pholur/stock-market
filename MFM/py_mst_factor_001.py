
import os, sys
import IPython
import numpy as np
import math
import datetime
from datetime import timedelta
import pandas as pd

import matplotlib.pyplot as plt
import networkx as nx

from scipy import interpolate
import argparse

from calendar import monthrange

#
np.random.seed(0)

#
parser = argparse.ArgumentParser()
parser.add_argument('--num_bins', help='Store num_steps', type = int, default = 10000)
parser.add_argument('--num_stocks', help='Store num_stocks', type = int, default = 500)
parser.add_argument('--num_sectors', help='Store num_sectors', type = int, default = 2)
parser.add_argument('--num_steps', help='Store num_steps', type = int, default = 10000)
parser.add_argument('--std_market', help='Store std_market', type = float, default = 0.05)
parser.add_argument('--std_sectors', help='Store std_sectors', type = float, default = 0.1)
parser.add_argument('--std_stocks', help='Store std_stocks', type = float, default = 1.0)
parser.add_argument('--mean_alpha', help='Store mean_alpha', type = float, default = 0.0)
parser.add_argument('--std_alpha', help='Store std_alpha', type = float, default = 1.0)
parser.add_argument('--mean_beta', help='Store mean_beta', type = float, default = 0.0)
parser.add_argument('--std_beta', help='Store std_beta', type = float, default = 1.0)
parser.add_argument('--mean_gamma', help='Store mean_gamma', type = float, default = 1.0)
parser.add_argument('--std_gamma', help='Store std_gamma', type = float, default = 1.0)
args = parser.parse_args()
num_bins = args.num_bins
num_steps = args.num_steps
num_sectors = args.num_sectors
num_stocks = args.num_stocks
std_market = args.std_market
std_sectors = args.std_sectors
std_stocks = args.std_stocks
mean_alpha = args.mean_alpha
std_alpha = args.std_alpha
mean_beta = args.mean_beta
std_beta = args.std_beta
mean_gamma = args.mean_gamma
std_gamma = args.std_gamma

print("num_steps", num_steps)
print("num_stocks", num_stocks)
print("num_sectors", num_sectors)

#
range_max = 20.0
range_min = -20.0

#
dir_output_main = "./z_output_mst_factor" + "_bins_" + str(num_bins).zfill(8) + "_stocks_" + str(num_stocks).zfill(6) + "_steps_" + str(num_steps).zfill(8)
if not os.path.exists(dir_output_main):
    os.mkdir(dir_output_main)

def truncate(number, decimals=0):

    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals

    return math.trunc(number * factor) / factor

def compute_rho():

    lists_interest_rates = []
    for i in range(num_stocks):
        path_output_main_all = "z_output_factor" + "_stocks_" + str(num_stocks).zfill(6) + "_sectors_" + str(num_sectors).zfill(6) + "_steps_" + str(num_steps).zfill(8)
        path_output_main = path_output_main_all + "/z_output" + "_m_alpha_" + '{:0=8.4f}'.format(mean_alpha) + "_std_alpha_" + '{:0=8.4f}'.format(std_alpha) + "_m_beta_" + '{:0=8.4f}'.format(mean_beta) + "_std_beta_" + '{:0=8.4f}'.format(std_beta) + "_m_gamma_" + '{:0=8.4f}'.format(mean_gamma) + "_std_gamma_" + '{:0=8.4f}'.format(std_gamma) + "_std_market_" + '{:0=8.4f}'.format(std_market) + "_std_sectors_" + '{:0=8.4f}'.format(std_sectors) + "_std_stocks_" + '{:0=28.14f}'.format(std_stocks)
        lists_interest_rates.append(pd.read_csv(path_output_main + "/interest_rate/" + "A_" + str(i).zfill(6) + ".csv", names = ["price"])["price"].values.tolist())

    rho_full = np.corrcoef(lists_interest_rates)

    mode_market = np.array(lists_interest_rates).mean(axis = 0)
    correlations_market = []
    for i in range(num_stocks):
        correlations_market.append(np.corrcoef(lists_interest_rates[i], mode_market)[0, 1])

    rho_market = np.zeros((num_stocks, num_stocks))
    for i in range(num_stocks):
        for j in range(num_stocks):
            rho_market[i, j] = correlations_market[i] * correlations_market[j]

    return rho_full, rho_market

def draw_graph(rho_full):

    d_connection = [[np.sqrt(2.0 * (1.0 - rho_full[i, j])) for j in range(num_stocks)] for i in range(num_stocks)]

    G_full = nx.from_numpy_matrix(np.around(np.array(d_connection),1))
    G_mst = nx.minimum_spanning_tree(G_full)

    options = {'node_size': 20, 'width': 0.5}
    
    c_map = {0: "blue", 1: "red", 2: "green"}
    color_map = []
    for c in range(num_stocks):
        color_map.append(c_map[int(c%num_sectors)])

    nx.draw_kamada_kawai(G_mst, with_labels = False, node_color = color_map, **options, edge_color = "silver")
    name_output_file = "_m_alpha_" + '{:0=10.5f}'.format(mean_alpha) + "_std_alpha_" + '{:0=10.5f}'.format(std_alpha) + "_m_beta_" + '{:0=10.5f}'.format(mean_beta) + "_std_beta_" + '{:0=10.5f}'.format(std_beta) + "_m_gamma_" + '{:0=10.5f}'.format(mean_gamma) + "_std_gamma_" + '{:0=10.5f}'.format(std_gamma) + "_std_market_" + '{:0=10.5f}'.format(std_market) + "_std_sectors_" + '{:0=10.5f}'.format(std_sectors) + "_std_stocks_" + '{:0=28.14f}'.format(std_stocks) 
    plt.savefig(dir_output_main + "/" "pic_graph" + name_output_file + ".pdf", format = "pdf", bbox_inches='tight')
    plt.close()

def compute_array(rho_correlation):

    array_rho = []
    for i in range(len(rho_correlation)):
        for j in range(len(rho_correlation[i])):
            array_rho.append(truncate(rho_correlation[i][j], 4))

    return array_rho

def histogram_c_ij(array_rho):

    mean_c_ij = np.mean(array_rho)
    var_c_ij = np.var(array_rho)
    sigma_cij = np.sqrt(var_c_ij)

    hist_array = np.histogram(array_rho, bins = num_bins, range = (range_min, range_max))
    prob_array = num_bins / (range_max - range_min) / sum(hist_array[0]) * hist_array[0]

    x_original_cij = []
    y_original_cij = []
    for i in range(len(hist_array[0])):
        x_original_cij.append(hist_array[1][i])
        y_original_cij.append(prob_array[i])

    name_output_file = "_m_alpha_" + '{:0=10.5f}'.format(mean_alpha) + "_std_alpha_" + '{:0=10.5f}'.format(std_alpha) + "_m_beta_" + '{:0=10.5f}'.format(mean_beta) + "_std_beta_" + '{:0=10.5f}'.format(std_beta) + "_m_gamma_" + '{:0=10.5f}'.format(mean_gamma) + "_std_gamma_" + '{:0=10.5f}'.format(std_gamma) + "_std_market_" + '{:0=10.5f}'.format(std_market) + "_std_sectors_" + '{:0=10.5f}'.format(std_sectors) + "_std_stocks_" + '{:0=28.14f}'.format(std_stocks) 
    f_output = open(dir_output_main + '/' + "z_output_hist_original" + name_output_file + ".txt", "w")
    for i in range(len(hist_array[0])):
        f_output.write(str(x_original_cij[i]))
        f_output.write(" ")
        f_output.write(str(y_original_cij[i]))
        f_output.write("\n")
    f_output.close()

    x_scaled_cij = []
    y_scaled_cij = []
    for i in range(len(hist_array[0])):
        x_scaled_cij.append(hist_array[1][i] / sigma_cij - mean_c_ij / sigma_cij)
        y_scaled_cij.append(prob_array[i] * sigma_cij)

    name_output_file = "_m_alpha_" + '{:0=10.5f}'.format(mean_alpha) + "_std_alpha_" + '{:0=10.5f}'.format(std_alpha) + "_m_beta_" + '{:0=10.5f}'.format(mean_beta) + "_std_beta_" + '{:0=10.5f}'.format(std_beta) + "_m_gamma_" + '{:0=10.5f}'.format(mean_gamma) + "_std_gamma_" + '{:0=10.5f}'.format(std_gamma) + "_std_market_" + '{:0=10.5f}'.format(std_market) + "_std_sectors_" + '{:0=10.5f}'.format(std_sectors) + "_std_stocks_" + '{:0=28.14f}'.format(std_stocks) 
    f_output = open(dir_output_main + '/' + "z_output_hist_scaled" + name_output_file + ".txt", "w")
    for i in range(len(hist_array[0])):
        f_output.write(str(x_scaled_cij[i]))
        f_output.write(" ")
        f_output.write(str(y_scaled_cij[i]))
        f_output.write("\n")
    f_output.close()

def main_function():

    rho_full_main, rho_market_main = compute_rho()
    draw_graph(rho_full_main)

    rho_diff_main = rho_full_main - rho_market_main
    array_rho_diff_main = compute_array(rho_diff_main)

    histogram_c_ij(array_rho_diff_main)

if __name__ == "__main__":

    main_function()

    # IPython.embed()

#

