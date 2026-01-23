#!/bin/bash
#SBATCH --job-name="coherence largescale tests"
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1

module load apps/python3

source activate coherence

# python ../coherence_analysis/coherence_analysis.py <method> <data_path> <averaging_window_length> <sub_window_length> [-o <overlap>] [-t <time_range>] [-ch <channel_range>] [-ds <channel_offset>] [-dt <time_step>] [-r <result_path>] [-p <parallel>]
# python ../coherence_analysis/coherence_analysis_no_dascore.py <method> <data_path> <averaging_window_length> <sub_window_length> [-o <overlap>] [-ch <channel_range>] [-ds <channel_offset>] [-dt <time_step>] [-r <result_path>] [-b <batch>] [-bs <batch_size>]

python ../coherence_analysis/coherence_analysis.py "qr" "/u/st/by/aissah/scratch/data/BradysHotspring" 60 1 -ch "(4500, 6500)" -ds 5 -dt 0.001
echo "QR Job finished"

# srun --exclusive --ntasks=1 python ../coherence_analysis/coherence_analysis_no_dascore.py "svd" "/u/st/by/aissah/scratch/data/BradysHotspring" 120 1 -o 0 -ch "(5000, 6500)" -ds 3 -dt 0.001 -b 1 -bs 0 -r "/u/st/by/aissah/scratch/coherence/coherence_test_results"
python ../coherence_analysis/coherence_analysis.py "svd" "/u/st/by/aissah/scratch/data/BradysHotspring" 60 1 -ch "(4500, 6500)" -ds 5 -dt 0.001
echo "SVD Job finished"

# python large_scale_tests.py "/beegfs/projects/martin/BradyHotspring" 60 2 0 3100 2000 200 1000 rsvd 1 0
# echo "rSVD Job finished"
