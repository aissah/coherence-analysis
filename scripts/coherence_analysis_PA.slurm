#!/bin/bash
#SBATCH --job-name="coherence largescale tests"
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1

module load apps/python3

source activate research_env

# python large_scale_test.py <file> <averaging_window_length> <sub_window_length> <overlap> <first_channel> <channel_offset> <num_channels> <samples_per_sec> <method> <batch> <batch_size>

python coherence_analysis/coherence_analysis.py qr "/data2/portAngelesWA/Cascadia-2023-04-12T23_06_38Z" 60 2 -ch "('50', '980')" -ds 3

echo "QR Job finished"

python coherence_analysis/coherence_analysis.py exact "/data2/portAngelesWA/Cascadia-2023-04-12T23_06_38Z" 60 2 -ch "('50', '980')" -ds 3

echo "Exact Job finished"

python coherence_analysis/coherence_analysis.py svd "/data2/portAngelesWA/Cascadia-2023-04-12T23_06_38Z" 60 2 -ch "('50', '980')" -ds 3

echo "SVD Job finished"

# python large_scale_tests.py "/beegfs/projects/martin/BradyHotspring" 60 2 0 3100 2000 200 1000 rsvd 1 0

# echo "rSVD Job finished"
