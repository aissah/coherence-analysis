#!/bin/bash
#SBATCH --job-name="coherence largescale tests"
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1

module load apps/python3/2020.02

source activate ny_env

# python large_scale_test.py <file> <averaging_window_length> <sub_window_length> <overlap> <first_channel> <channel_offset> <num_channels> <samples_per_sec> <method> <batch> <batch_size>

python large_scale_test.py "/beegfs/projects/martin/BradyHotspring" 60 2 0 3100 2000 200 1000 qr 1 0

echo "First Job finished"

python large_scale_test.py "/beegfs/projects/martin/BradyHotspring" 60 2 0 3100 2000 200 1000 exact 1 0

echo "Second Job finished"
