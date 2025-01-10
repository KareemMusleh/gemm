#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

with open(Path(__file__).parent.parent/'config.json') as config_file:
    config = json.load(config_file)

def show_and_save_results():
    
    GHz_of_processor = config['GHz_of_processor']
    nprocessors = config['nprocessors']
    nflops_per_cycle = config['nflops_per_cycle']
    # Extract max_gflops from proc_parameters
    max_gflops = nflops_per_cycle * nprocessors * GHz_of_processor

    # Load the first dataset (output_old equivalent)
    # Assuming output_old is a NumPy array (or similar structure)
    # Replace this with actual code to load the dataset, e.g. np.loadtxt() or np.genfromtxt()
    old_data = pd.read_csv('output_old.csv')  # Assuming output_old.txt contains the data
    num_columns = old_data.shape[1]

    # Store version from the old dataset
    old_version = old_data.columns[0]
    old_data.columns = range(num_columns)

    # Plot the old dataset
    plt.plot(old_data[0], old_data[1], 'bo-.', label=f'OLD ({old_version})')
    last = old_data.shape[0]

    plt.axis([0, old_data.iloc[-1][0], 0, max_gflops])

    plt.xlabel('m = n = k')
    plt.ylabel('GFLOPS/sec.')

    new_data = pd.read_csv('output_new.csv')

    new_version = new_data.columns[0]
    new_data.columns = range(new_data.shape[1])

    # Plot the new dataset
    plt.plot(new_data[0], new_data[1], 'r-*', label=f'NEW ({new_version})')

    # Title and formatting
    title_string = f"OLD = {old_version}, NEW = {new_version}"
    plt.title(title_string)

    # Show the legend
    plt.legend()

    # Save the plot as a PNG image
    filename = f"compare_{old_version}_{new_version}.png"
    plt.savefig(filename)
    plt.show()

if __name__ == '__main__':
    show_and_save_results()