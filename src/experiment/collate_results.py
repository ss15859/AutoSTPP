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

mask = (df['time'] >= config.catalog.test_nll_start) & (df['time'] <= config.catalog.test_nll_end)

# Fill in values for the specified time range
df.loc[mask, 'LL'] = -lls.cpu().numpy()
df.loc[mask, 'SLL'] = slls.cpu().numpy()
df.loc[mask, 'TLL'] = tlls.cpu().numpy()
df.loc[mask, 'int_lambd'] = -int_lambds.cpu().numpy()
df.loc[mask, 'lambd_star'] = np.exp(lambd_stars.cpu().numpy())

