#!/bin/bash
#SBATCH --job-name="coherence largescale tests"
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1


module load apps/python3/2020.02

source activate ny_env

# python coherence_test.py <file> <averaging_window_length> <sub_window_length> <overlap> <first_channel> <num_channels> <samples_per_sec> <channel_offset> <method> <batch> <batch_size>

python %~dp0../makeplots.py 2201050 svd

python %~dp0../makeplots_clusters.py 2201050 svd