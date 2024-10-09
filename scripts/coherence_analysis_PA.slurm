#!/bin/bash
#SBATCH --job-name="coherence largescale tests"
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1

module load apps/python3

source activate research_env

# python <method> ../coherence_analysis/coherence_analysis.py <data_directory> <averaging_window_length> <sub_window_length> <overlap> <channel_range> <channel_offset> <samples_per_sec>

python ../coherence_analysis/coherence_analysis.py qr "/u/st/by/aissah/scratch/Cascadia-2023-04-12T23_06_38Z" 60 2 -ch "('50', '980')" -ds 3

echo "QR Job finished"

python ../coherence_analysis/coherence_analysis.py exact "/u/st/by/aissah/scratch/Cascadia-2023-04-12T23_06_38Z" 60 2 -ch "('50', '980')" -ds 3

echo "Exact Job finished"

python ../coherence_analysis/coherence_analysis.py svd "/u/st/by/aissah/scratch/Cascadia-2023-04-12T23_06_38Z" 60 2 -ch "('50', '980')" -ds 3

echo "SVD Job finished"
