import os, sys
import numpy as np
import argparse
from tqdm import tqdm
np.random.seed(0)

#
parser = argparse.ArgumentParser()
parser.add_argument('--num_stocks', help='Store num_stocks', type = int, default = 500)
parser.add_argument('--num_steps', help='Store num_steps', type = int,  default = 10000)
parser.add_argument('--delta', help='Store delta', type = float, default = 0.1)
args = parser.parse_args()
num_stocks = args.num_stocks
num_steps = args.num_steps
delta = args.delta
std_stockses = [0.01, 1.0, 100.0]

for std_stocks in std_stockses:
    print("num_stocks", num_stocks)
    print("std_stocks", std_stocks)
    print("num_steps", num_steps)
    print("delta", delta)

    #
    noise_stocks = np.random.normal(0.0, std_stocks, (num_stocks, num_steps))

    alpha = np.array([1.0 for i in range(num_stocks)])
    # alpha = np.random.normal(1.0, 0.00, num_stocks)

    beta = np.array([0.05 for i in range(num_stocks)])

    gamma = np.array([1.0 for i in range(num_stocks)])
    # gamma = np.random.normal(1.0, 0.00, num_stocks)

    Delta = 1.0

    interest_rates = np.array([[0.0 for i in range(num_steps)] for j in range(num_stocks)])

    for k in tqdm(range(num_steps)):
        if k == 0:
            interest_rates[:, 0] = noise_stocks[:, 0]/alpha[:]
        elif k == 1:
            interest_rates[:, 1] = interest_rates[:, 0] + noise_stocks[:, 1]/alpha[:] 
        elif k >= 2:
            interaction_sum = 0.0
            distance_matrix = (np.abs(interest_rates[:, k - 1].reshape(-1,1) - interest_rates[:, k - 1].reshape(1,-1)) < delta) + 0.0
            N = np.sum(distance_matrix, axis=1)
            diff = interest_rates[:,k-1] - interest_rates[:,k-2]
            interaction_sum = np.matmul(distance_matrix, diff)
            interest_rates[:, k] = (alpha[:] - Delta * beta[:])/alpha[:] * interest_rates[:, k - 1] + (1 / alpha[:]) * gamma[:] * 1.0 / (N) * interaction_sum + (1 / alpha[:]) * noise_stocks[:, k] 


    path_output_main_all = "z_output_vicsek" + "_stocks_" + str(num_stocks).zfill(6) + "_steps_" + str(num_steps).zfill(8) 
    if not os.path.exists(path_output_main_all):
        os.mkdir(path_output_main_all)

    path_output_main = path_output_main_all + "/z_output" + "_std_stocks_" + '{:0=28.14f}'.format(std_stocks) + "_delta_" + '{:0=28.14f}'.format(delta)
    if not os.path.exists(path_output_main):
        os.mkdir(path_output_main)

    path_output_sub = path_output_main + "/interest_rate/"
    if not os.path.exists(path_output_sub):
        os.mkdir(path_output_sub)

    f_output_001 = []
    for i in range(num_stocks):
        f_output_001.append(open(path_output_sub + "/A_" + str(i).zfill(6) + ".csv", "w"))
        for j in range(num_steps):
            f_output_001[i].write(str(interest_rates[i, j]))
            f_output_001[i].write("\n")
        f_output_001[i].close()

# import IPython
# IPython.embed()

