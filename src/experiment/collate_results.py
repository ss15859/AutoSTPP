import yaml
import sys
import os
import pandas as pd
from dotwiz import DotWiz
from shapely.geometry import Polygon
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import torch
import re

# Load configuration
with open(sys.argv[1], 'r') as file:
	config_dict = yaml.safe_load(file)
config = DotWiz(config_dict)

# Define results directory
results_dir = f"output_data/{config.data.init_args.name}"

# Collect data from .pt files
lls, slls, tlls, int_lambds, lambd_stars = [], [], [], [], []
if 'deep' in config.data.init_args.name:
	w_is, b_is, gamma_is_1, gamma_is_2 = [], [], [], []

def extract_number(filename):
	match = re.search(r'\d+', filename)  # Find the first sequence of digits in the filename
	return int(match.group()) if match else float('inf')  # Return the number or infinity if no number is found

# Sort files numerically based on extracted numbers
for file in sorted(os.listdir(results_dir), key=extract_number):
	if file.endswith(".pt"):
		print(f"Loading {file}")
		if 'deep' in config.data.init_args.name:
			nll_vec_scaled, sll_vec_scaled, tll_vec_scaled, int_lambd, lambd_star, w_i, b_i, inv_var = torch.load(os.path.join(results_dir, file))
			w_is.append(w_i[:,:20])
			b_is.append(b_i[:,:20])
			gamma_is_1.append(inv_var[:,:20,0])
			gamma_is_2.append(inv_var[:,:20,1])
		else:            
			nll_vec_scaled, sll_vec_scaled, tll_vec_scaled, int_lambd, lambd_star = torch.load(os.path.join(results_dir, file))
		
		lls.append(nll_vec_scaled)
		slls.append(sll_vec_scaled)
		tlls.append(tll_vec_scaled)
		int_lambds.append(int_lambd)
		lambd_stars.append(lambd_star)

# Concatenate tensors
lls = torch.cat(lls)
slls = torch.cat(slls)
tlls = torch.cat(tlls)
int_lambds = torch.cat(int_lambds)
lambd_stars = torch.cat(lambd_stars)

if 'deep' in config.data.init_args.name:
	w_is = torch.cat(w_is)
	b_is = torch.cat(b_is)
	gamma_is_1 = torch.cat(gamma_is_1)
	gamma_is_2 = torch.cat(gamma_is_2)



# Load and filter catalog data
df = pd.read_csv(
	config.catalog.path,
	parse_dates=["time"],
	dtype={"url": str, "alert": str},
)

df = df.sort_values(by='time')
df = df[df['magnitude'] >= config.catalog.Mcut]

# Add new columns for results
df['LL'] = np.nan
df['SLL'] = np.nan
df['TLL'] = np.nan
df['int_lambd'] = np.nan
df['lambd_star'] = np.nan



# Fill in values for the specified time range
df.loc[df['time'] >= config.catalog.test_nll_start, 'LL'] = -lls.cpu().numpy()
df.loc[df['time'] >= config.catalog.test_nll_start, 'SLL'] = slls.cpu().numpy()
df.loc[df['time'] >= config.catalog.test_nll_start, 'TLL'] = tlls.cpu().numpy()
df.loc[df['time'] >= config.catalog.test_nll_start, 'int_lambd'] = -int_lambds.cpu().numpy()
df.loc[df['time'] >= config.catalog.test_nll_start, 'lambd_star'] = np.exp(lambd_stars.cpu().numpy())

if 'deep' in config.data.init_args.name:
	for i in range(20):
		df.loc[df['time'] >= config.catalog.test_nll_start, f'w_i_{i+1}'] = w_is[:, i].cpu().numpy()
		df.loc[df['time'] >= config.catalog.test_nll_start, f'b_i_{i+1}'] = b_is[:, i].cpu().numpy()
		df.loc[df['time'] >= config.catalog.test_nll_start, f'gamma_i_1_{i+1}'] = gamma_is_1[:, i].cpu().numpy()
		df.loc[df['time'] >= config.catalog.test_nll_start, f'gamma_i_2_{i+1}'] = gamma_is_2[:, i].cpu().numpy()

# Write augmented catalog to file
output_path = f"output_data/{config.data.init_args.name}/augmented_catalog.csv"
df.to_csv(output_path, index=False)
print(f"Wrote {output_path}")
print("Done")

if "deep" in config.data.init_args.name:

	log10_k0 = -2.6755424481822057
	k0 = 10 ** log10_k0
	print(f"k0: {k0}")
	a = 1.5391618199834523

	df = df[df['time'] >= config.catalog.test_nll_start]
	df['time'] = pd.to_datetime(df['time'])

	

	df['ETAS_prod'] = k0 * np.exp(a * (df['magnitude']-2.5))

	# Set window size for moving average
	window = 50  # Adjust this based on your needs

	# Compute moving averages
	df['w_i_1_ma'] = df['gamma_i_1_1'].rolling(window=window).mean()
	df['b_i_1_ma'] = df['gamma_i_2_1'].rolling(window=window).mean()
	# Compute moving average of ETAS_prod
	df['ETAS_prod_ma'] = df['ETAS_prod'].rolling(window=window).mean()

	# Plotting
	fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 9), sharex=True,
									gridspec_kw={'height_ratios': [1, 1, 1]})

	# Plot moving averages instead of scatter
	ax1.plot(df['time'], df['w_i_1_ma'], label=r'Moving Avg $k_i$', color="#84A07C")
	ax1.fill_between(df['time'], df['w_i_1_ma'].min(), df['w_i_1_ma'], color="#84A07C", alpha=0.6)

	ax2.plot(df['time'], df['b_i_1_ma'], label=r'Moving Avg $\beta_i$', color="#5F0F40")
	ax2.fill_between(df['time'], df['b_i_1_ma'].min(), df['b_i_1_ma'], color="#5F0F40", alpha=0.6)

	# ax1.set_yscale('log')
	# ax2.set_yscale('log')
	ax1.set_ylabel(r'$\gamma_i(x)$',fontsize=16, rotation=0, labelpad=20)
	ax2.set_ylabel(r'$\gamma_i(y)$',fontsize=16, rotation=0, labelpad=20)
	ax3.set_ylabel(r'$M_w$',fontsize=16,rotation=0, labelpad=20)	
	ax3.set_xlabel(r'time',fontsize=16)



	# Still using scatter for magnitude, optionally keep it
	z = (9.5 ** df['magnitude']) * 0.0001
	ax3.scatter(df['time'], df['magnitude'], s=z, alpha=0.7,color = "#E36414")
	ax3.set_yticks([2, 4, 6, 8])

	# Optional: Add legends
	# ax1.legend()
	# ax2.legend()
	plt.tight_layout()
	plt.savefig(f"output_data/{config.data.init_args.name}/space_moving_averages.png", dpi=300)


if "deep" in config.data.init_args.name:

	log10_k0 = -2.6755424481822057
	k0 = 10 ** log10_k0
	print(f"k0: {k0}")
	a = 1.5391618199834523

	df = df[df['time'] >= config.catalog.test_nll_start]
	df['time'] = pd.to_datetime(df['time'])

	df['ETAS_prod'] = k0 * np.exp(a * (df['magnitude']-2.5))

	# Set window size for moving average
	window = 50  # Adjust this based on your needs

	# Compute moving averages
	df['w_i_1_ma'] = df['w_i_10'].rolling(window=window).mean()
	df['b_i_1_ma'] = df['b_i_1'].rolling(window=window).mean()
	# Compute moving average of ETAS_prod
	df['ETAS_prod_ma'] = df['ETAS_prod'].rolling(window=window).mean()

	# Plotting
	fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
									gridspec_kw={'height_ratios': [1, 1, 1]})

	# Plot moving averages instead of scatter
	ax1.plot(df['time'], df['w_i_1_ma'], label=r'Moving Avg $k_i$', color="#84A07C")
	ax1.fill_between(df['time'], df['w_i_1_ma'].min(), df['w_i_1_ma'], color="#84A07C", alpha=0.6)

	ax2.plot(df['time'], 1 / df['b_i_1_ma'], label=r'Moving Avg $\beta_i$', color="#5F0F40")
	ax2.fill_between(df['time'], (1 / df['b_i_1_ma']).max(), 1 / df['b_i_1_ma'], color="#5F0F40", alpha=0.6)

	ax1.set_yscale('log')
	ax2.set_yscale('log')
	ax1.set_ylabel(r'$k_i$',fontsize=16, rotation=0, labelpad=20)
	ax2.set_ylabel(r'$1/\beta_i$',fontsize=16, rotation=0, labelpad=20)
	ax3.set_ylabel(r'$M_w$',fontsize=16,rotation=0, labelpad=20)	
	ax3.set_xlabel(r'time',fontsize=16)



	# Still using scatter for magnitude, optionally keep it
	z = (9.5 ** df['magnitude']) * 0.0001
	ax3.scatter(df['time'], df['magnitude'], s=z, alpha=0.7,color = "#E36414")
	ax3.set_yticks([2, 4, 6, 8])

	# Optional: Add legends
	# ax1.legend()
	# ax2.legend()
	plt.tight_layout()
	plt.savefig(f"output_data/{config.data.init_args.name}/moving_averages.png", dpi=300)

