
import os, sys
import IPython
import numpy as np
import math
import datetime
from datetime import timedelta
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import networkx as nx

from scipy import interpolate
import argparse

from calendar import monthrange
import hdbscan
from networkx.algorithms.community.centrality import girvan_newman

#
np.random.seed(0)

#
parser = argparse.ArgumentParser()
parser.add_argument('--num_stocks', help='Store num_stocks', type = int, default = 500)
parser.add_argument('--num_steps', help='Store num_steps', type = int, default = 10000)
parser.add_argument('--num_bins', help='Store num_bins', type = int, default = 10000)
parser.add_argument('--delta', help='Store delta', type = float, default = 0.1)
args = parser.parse_args()


std_stockses = [100.0, 1.0, 0.01]
labels = None

for std_stocks in std_stockses:
    num_stocks = args.num_stocks
    num_steps = args.num_steps
    num_bins = args.num_bins
    delta = args.delta

    #
    range_max = 20.0
    range_min = -20.0

    print("num_bins", num_bins)
    print("num_stocks", num_stocks)
    print("std_stocks", std_stocks)
    print("num_steps", num_steps)
    print("delta", delta)
    print("range_max", range_max)
    print("range_min", range_min)

    #
    dir_output_main = "./z_output_mst_vicsek_bins_" + str(num_bins).zfill(8) + "_stocks_" + str(num_stocks).zfill(6) + "_steps_" + str(num_steps).zfill(8) 
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
            lists_interest_rates.append(pd.read_csv("z_output_vicsek" + "_stocks_" + str(num_stocks).zfill(6) + "_steps_" + str(num_steps).zfill(8) + "/z_output" + "_std_stocks_" + '{:0=28.14f}'.format(std_stocks) + "_delta_" + '{:0=28.14f}'.format(delta) + "/interest_rate/" + "A_" + str(i).zfill(6) + ".csv", names = ["interest_rate"])["interest_rate"].values.tolist())

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
        global labels
        d_connection = np.array([[np.sqrt(2.0 * (1.0 - rho_full[i, j])) for j in range(num_stocks)] for i in range(num_stocks)])
        G_full = nx.from_numpy_matrix(np.around(np.array(d_connection), 1))
        G_mst = nx.minimum_spanning_tree(G_full)

        if labels == None:
            d_conn_renew = girvan_newman(G_mst)
            limited = itertools.takewhile(lambda c: len(c) <= 11, d_conn_renew)
            for communities in limited:
                sets = tuple(sorted(c) for c in communities)
            color_map = [0]*num_stocks
            for i,s in enumerate(sets):
                for s_ in s:
                    color_map[s_] = i
            labels = color_map
        else:
            color_map = labels

        options = {'node_size': 20, 'width': 0.5}
        
        #color_map = ["blue"]
        nx.draw_kamada_kawai(G_mst, with_labels = False, node_color = color_map, **options, edge_color = "silver", cmap="gist_rainbow")
        plt.savefig(dir_output_main + "/" "pic_graph" + "_std_stocks_" + '{:0=28.14f}'.format(std_stocks) + "_delta_" + '{:0=28.14f}'.format(delta) + ".pdf", format = "pdf", bbox_inches='tight')
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
        sigma_c_ij = np.sqrt(var_c_ij)

        f_output = open(dir_output_main + '/' + "z_output_statistics" + "_std_stocks_" + '{:0=28.14f}'.format(std_stocks) + "_delta_" + '{:0=28.14f}'.format(delta) + ".txt", "w")
        f_output.write(str(num_stocks))
        f_output.write(" ")
        f_output.write(str(num_steps))
        f_output.write(" ")
        f_output.write(str(std_stocks))
        f_output.write(" ")
        f_output.write(str(delta))
        f_output.write(" ")
        f_output.write(str(mean_c_ij))
        f_output.write(" ")
        f_output.write(str(var_c_ij))
        f_output.write(" ")
        f_output.write(str(sigma_c_ij))
        f_output.write("\n")
        f_output.close()

        hist_array = np.histogram(array_rho, bins = num_bins, range = (range_min, range_max))
        hist_array = np.histogram(array_rho, bins = num_bins, range = (range_min, range_max))
        prob_array = num_bins / (range_max - range_min) / sum(hist_array[0]) * hist_array[0]

        x_original_cij = []
        y_original_cij = []
        for i in range(len(hist_array[0])):
            x_original_cij.append(hist_array[1][i])
            y_original_cij.append(prob_array[i])

        f_output = open(dir_output_main + '/' + "z_output_hist_original" + "_std_stocks_" + '{:0=28.14f}'.format(std_stocks) + "_delta_" + '{:0=28.14f}'.format(delta) + ".txt", "w")
        for i in range(len(hist_array[0])):
            f_output.write(str(x_original_cij[i]))
            f_output.write(" ")
            f_output.write(str(y_original_cij[i]))
            f_output.write("\n")
        f_output.close()

        x_scaled_cij = []
        y_scaled_cij = []
        for i in range(len(hist_array[0])):
            x_scaled_cij.append(hist_array[1][i] / sigma_c_ij - mean_c_ij / sigma_c_ij)
            y_scaled_cij.append(prob_array[i] * sigma_c_ij)

        f_output = open(dir_output_main + '/' + "z_output_hist_scaled" + "_std_stocks_" + '{:0=28.14f}'.format(std_stocks) + "_delta_" + '{:0=28.14f}'.format(delta) + ".txt", "w")
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

    main_function()

