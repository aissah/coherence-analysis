#!/bin/bash
#SBATCH --job-name="coherence largescale tests"
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1

module load apps/python3

source activate coherence

# python large_scale_test.py <file> <averaging_window_length> <sub_window_length> <overlap> <first_channel> <channel_offset> <num_channels> <samples_per_sec> <method> <batch> <batch_size>
# python ../coherence_analysis/coherence_analysis.py <method> <data_directory> <averaging_window_length> <sub_window_length> <overlap> <channel_range> <channel_offset> <samples_per_sec>

python ../coherence_analysis/coherence_analysis.py "qr" "/u/st/by/aissah/scratch/data/BradysHotspring" 60 1 -ch "(4500, 6500)" -ds 5 -dt 0.001

echo "QR Job finished"

# python large_scale_tests.py "/beegfs/projects/martin/BradyHotspring" 120 2 0 3100 2000 200 1000 exact 1 0

# echo "Exact Job finished"

python ../coherence_analysis/coherence_analysis.py "svd" "/u/st/by/aissah/scratch/data/BradysHotspring" 60 1 -ch "(4500, 6500)" -ds 5 -dt 0.001
# python ../coherence_analysis/large_scale_test.py "/u/st/by/aissah/scratch/data/BradysHotspring" 120 2 0 3100 2000 200 1000 qr 1 0
echo "SVD Job finished"

# python large_scale_tests.py "/beegfs/projects/martin/BradyHotspring" 60 2 0 3100 2000 200 1000 rsvd 1 0

# echo "rSVD Job finished"
