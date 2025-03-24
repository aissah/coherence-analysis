#!/bin/bash
#SBATCH --job-name="coherence largescale tests"
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1

module load apps/python3

source activate coherence

# python large_scale_test.py <file> <averaging_window_length> <sub_window_length> <overlap> <first_channel> <channel_offset> <num_channels> <samples_per_sec> <method> <batch> <batch_size>
# python ../coherence_analysis/coherence_analysis.py <method> <data_directory> <averaging_window_length> <sub_window_length> <overlap> <channel_range> <channel_offset> <samples_per_sec>

python ../coherence_analysis/coherence_analysis.py qr "/beegfs/projects/martin/BradyHotspring" 120 2 -ch "(3100, 5100)" -ds 10 -dt 0.001

echo "QR Job finished"

# python large_scale_tests.py "/beegfs/projects/martin/BradyHotspring" 120 2 0 3100 2000 200 1000 exact 1 0

# echo "Exact Job finished"

python ../coherence_analysis/coherence_analysis.py svd "/beegfs/projects/martin/BradyHotspring" 120 2 -ch "(3100, 5100)" -ds 10 -dt 0.001

echo "SVD Job finished"

# python large_scale_tests.py "/beegfs/projects/martin/BradyHotspring" 60 2 0 3100 2000 200 1000 rsvd 1 0

# echo "rSVD Job finished"
