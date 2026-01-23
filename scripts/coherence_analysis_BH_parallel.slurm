#!/bin/bash
#SBATCH --job-name="coherence largescale tests in parallel"
#SBATCH --export=ALL
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4

module load apps/python3

source activate coherence

## declare array of data directories. This is to allow for parallel processing of multiple datasets.
## Example: declare -a arr=("/path/to/data1" "/path/to/data2" "/path/to/data3")
# declare -a arr=("/u/st/by/aissah/scratch/data/BradysHotspring/03_13_2016" "/u/st/by/aissah/scratch/data/BradysHotspring/03_14_2016")
data_dir="/u/st/by/aissah/scratch/data/BradysHotspring"
## declare array of methods to be used.
declare -a arr=("qr" "svd")

# python ../coherence_analysis/coherence_analysis.py <method> <data_path> <averaging_window_length> <sub_window_length> [-o <overlap>] [-t <time_range>] [-ch <channel_range>] [-ds <channel_offset>] [-dt <time_step>] [-r <result_path>] [-p <parallel>]
# python ../coherence_analysis/coherence_analysis_no_dascore.py <method> <data_path> <averaging_window_length> <sub_window_length> [-o <overlap>] [-ch <channel_range>] [-ds <channel_offset>] [-dt <time_step>] [-r <result_path>] [-b <batch>] [-bs <batch_size>]
for method in "${arr[@]}";
do
   echo "Working on directory: $data_dir"
   srun --exclusive --ntasks=1 python ../coherence_analysis/coherence_analysis_no_dascore.py $method "$data_dir" 120 1 -o 0 -ch "(5000, 6500)" -ds 3 -dt 0.001 -b 1 -bs 0 -r "/u/st/by/aissah/scratch/coherence/coherence_test_results" &
   # srun --exclusive --ntasks=1 python ../coherence_analysis/coherence_analysis.py $method "$data_dir" 60 1 -ch "(4500, 6500)" -ds 5 -dt 0.001 &
done
wait

echo "All jobs finished"
