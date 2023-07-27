import os
import numpy as np
import argparse

np.random.seed(0)

#
parser = argparse.ArgumentParser()
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

print("num_stocks", num_stocks)
print("num_steps", num_steps)
print("num_sectors", num_sectors)

alpha = np.random.normal(mean_alpha, std_alpha, num_stocks)
beta = np.random.normal(mean_beta, std_beta, num_stocks)
gamma = np.random.normal(mean_gamma, std_gamma, num_stocks)

noise_market = np.random.normal(0.0, std_market, num_steps)
noise_sectors = np.random.normal(0.0, std_sectors, (num_sectors, num_steps))
noise_stocks = np.random.normal(0.0, std_stocks, (num_stocks, num_steps))

interest_rates = np.array([[0.0 for i in range(num_steps)] for j in range(num_stocks)])

interest_rate_market = [0.0 for i in range(num_steps)]
interest_rate_sectors = [[0.0 for i in range(num_steps)] for j in range(num_sectors)]
for j in range(num_steps):
    interest_rate_market[j] = np.sum(noise_market[0:j])
    for k in range(num_sectors):
        interest_rate_sectors[k][j] = np.sum(noise_sectors[k][0:j])

for i in range(num_stocks):
    k = i % num_sectors
    for j in range(num_steps):
        interest_rates[i, j] = alpha[i] + beta[i] * interest_rate_market[j] + noise_stocks[i, j] + gamma[k] * interest_rate_sectors[k][j]

path_output_main_all = "z_output_factor" + "_stocks_" + str(num_stocks).zfill(6) + "_sectors_" + str(num_sectors).zfill(6) + "_steps_" + str(num_steps).zfill(8)
if not os.path.exists(path_output_main_all):
    os.mkdir(path_output_main_all)

path_output_main = path_output_main_all + "/z_output" + "_m_alpha_" + '{:0=8.4f}'.format(mean_alpha) + "_std_alpha_" + '{:0=8.4f}'.format(std_alpha) + "_m_beta_" + '{:0=8.4f}'.format(mean_beta) + "_std_beta_" + '{:0=8.4f}'.format(std_beta) + "_m_gamma_" + '{:0=8.4f}'.format(mean_gamma) + "_std_gamma_" + '{:0=8.4f}'.format(std_gamma) + "_std_market_" + '{:0=8.4f}'.format(std_market) + "_std_sectors_" + '{:0=8.4f}'.format(std_sectors) + "_std_stocks_" + '{:0=28.14f}'.format(std_stocks) + "/"
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

