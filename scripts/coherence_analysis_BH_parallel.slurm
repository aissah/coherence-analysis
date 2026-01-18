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
# python ../coherence_analysis/coherence_analysis.py <method> <data_directory> <averaging_window_length> <sub_window_length> <overlap> <channel_range> <channel_offset> <samples_per_sec>
for method in "${arr[@]}";
do
   echo "Working on directory: $data_dir"
   srun --exclusive --ntasks=1 python ../coherence_analysis/wo_dascore_large_scale_tests.py "$data_dir" 120 1 0 5000 1500 500 1000 $method 1 0 &
   # srun --exclusive --ntasks=1 python ../coherence_analysis/coherence_analysis.py "$method" "$data_dir" 60 1 -ch "(4500, 6500)" -ds 3 -dt 0.001 -p &
done
wait

echo "All jobs finished"
