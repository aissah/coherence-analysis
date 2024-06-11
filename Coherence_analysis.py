# This python file is to test coherence analyses for a larger dataset. 
# The data is from Brady Hotspring and is in hdf5 format.
# This is not complete and is a work in progress.
# python coherence_test.py <file> <averaging_window_length> <sub_window_length> <overlap> <first_channel> <num_channels> <samples_per_sec> <channel_offset> <method> <batch> <batch_size>

import os
import pickle
import sys
from datetime import datetime
import numpy as np

import functions as func

METHODS = ['exact', 'qr', 'svd', 'rsvd', 'power', 'qr iteration']

# Take inputs from the command line
file = sys.argv[1]
averaging_window_length = int(sys.argv[2]) # Averaging window length in seconds
sub_window_length = int(sys.argv[3]) # sub-window length in seconds
overlap = int(sys.argv[4]) # overlap in seconds
first_channel = int(sys.argv[5]) # first channel
num_channels = int(sys.argv[6]) # number of sensors
samples_per_sec = int(sys.argv[7]) # samples per second
channel_offset = int(sys.argv[8]) # channel offset
method = sys.argv[9] # method to use for coherence analysis
batch = int(sys.argv[10]) # Batch of files assuming jobs are run in parallel for files in batches. Should be one if that is not the case.
batch_size = int(sys.argv[11]) # Number of files in batch. Should be number of files being considered if job is not done in batches

# batch = int(
#     sys.argv[4]
# )  # Batch of files assuming jobs are run in parallel for files in batches. Should be one if that is not the case.
# batch_size = int(
#     sys.argv[5]
# )  # Number of files in batch. Should be number of files being considered if job is not done in batches
# compression_flag = int(
#     sys.argv[6]
# )  # 1 if compressed data is used otherwise uncompressed data is used

# Path to the directory containing the data files
data_basepath = "/beegfs/projects/martin/BradyHotspring"  # "D:/CSM/Mines_Research/Test_data/Brady Hotspring"
# files = os.listdir(data_basepath)

# Path to the directory where the results will be saved
save_location = "/u/st/by/aissah/scratch/event_detection/template_matching"  # "D:/CSM/Mines_Research/Test_data/"
# samples_per_sec = 1000

start_time = datetime.now()

# Get the file names of the data files by going through the folders contained
# in the base path and putting together the paths to files ending in .h5
data_files = []
for dir_path, dir_names, file_names in os.walk(data_basepath):
    dir_names.sort()
    file_names.sort()
    data_files.extend(
        [
            os.path.join(dir_path, file_name)
            for file_name in file_names
            if ".h5" in file_name
        ]
    )

metadata = {}
if batch == 1:
    first_file_time = data_files[0][-15:-3]

    data_files = data_files[:batch_size]
    data, _ = func.loadBradyHShdf5(data_files[0], normalize="no")
    # data = data[first_channel:last_channel]
    data = data[first_channel:channel_offset+first_channel:int(channel_offset/num_channels)]
else:  # with more batches, append end of previous file for continuity
    try:
        data_files = data_files[(batch - 1) * batch_size - 1 : batch * batch_size]
        metadata["files"] = [a[-15:-3] for a in data_files]
    except IndexError:
        data_files = data_files[(batch - 1) * batch_size - 1 :]
        metadata["files"] = [a[-15:-3] for a in data_files]
    data, _ = func.loadBradyHShdf5(data_files[1], normalize="no")
    data = data[first_channel:channel_offset+first_channel:int(channel_offset/num_channels)]
    # preceding_data, _ = func.loadBradyHShdf5(data_files[0], normalize="no")
    # data = np.append(
    #     preceding_data[first_channel:last_channel, -len(template[1]) + 1 :],
    #     data[first_channel:last_channel],
    #     axis=1,
    # )

# work on files after first file in batch. This works exactly as we handled the
# beginning of later batches. Then we keep appending to the variables set up for
# first file of the batch above

if method == 'qr':
    detection_significances, eig_estimatess = func.coherence(data[first_channel:channel_offset+first_channel:int(channel_offset/num_channels)], sub_window_length, overlap, sample_interval=1/samples_per_sec, method=method)
elif method in METHODS:
    detection_significances = func.coherence(data[first_channel:channel_offset+first_channel:int(channel_offset/num_channels)], sub_window_length, overlap, sample_interval=1/samples_per_sec, method=method)

# norm_win_spectra, frequencies = func.normalised_windowed_spectra(data[first_channel:num_channels+first_channel:int(num_channels/nsensors)], sub_window_length, overlap, sample_interval=1/samples_per_sec)
# welch_coherence_mat = np.matmul(norm_win_spectra, np.conjugate(norm_win_spectra.transpose(0,2,1)))
# coherence = np.absolute(welch_coherence_mat)**2

# num_freqs = coherence.shape[0]
# eig_ratios = np.empty(num_freqs)
# eig_ratios_qr = np.empty(num_freqs)
# for d in range(num_freqs):
#     eigenvals, _ = np.linalg.eig(coherence[d])
#     eigenvals = np.sort(eigenvals)[::-1]
#     eig_ratios[d] = eigenvals[0]/np.sum(eigenvals)

#     Q,R = np.linalg.qr(norm_win_spectra[d])
#     qr_approx = np.sort(np.diag(np.absolute(R@R.transpose()))**2)[::-1]
#     eig_ratios_qr[d] = qr_approx[0]/np.sum(np.absolute(qr_approx))

end_time = datetime.now()
print(f"First file completed in: {end_time - start_time}", flush=True)

for a in data_files[1:]:
    # preceding_data = data[:, -len(template[1]) + 1 :]
    data, _ = func.loadBradyHShdf5(a, normalize="no")
    # data = np.append(preceding_data, data[first_channel:last_channel], axis=1)
    data = data[first_channel:channel_offset+first_channel:int(channel_offset/num_channels)]
    if method == 'qr':
        detection_significance, eig_estimates = func.coherence(data[first_channel:channel_offset+first_channel:int(channel_offset/num_channels)], sub_window_length, overlap, sample_interval=1/samples_per_sec, method=method)
    elif method in METHODS:
        detection_significance = func.coherence(data[first_channel:channel_offset+first_channel:int(channel_offset/num_channels)], sub_window_length, overlap, sample_interval=1/samples_per_sec, method=method)

    if detection_significance.shape == detection_significances.shape:
        detection_significances = np.append(detection_significances[np.newaxis], detection_significance[np.newaxis], axis=0)
    else:
        detection_significances = np.append(detection_significances, detection_significance[np.newaxis], axis=0)
    # if method == 'qr':
    #     eig_estimatess = np.append(eig_estimatess[np.newaxis], eig_estimates[np.newaxis], axis=0)
    save_data = {'detection_significance': detection_significances, 'metadata': metadata}

savename = save_location / f"{method}_detection_significance_{metadata["files"][0]}_{metadata["files"][-1]}.pkl" # need to modify to save with correct file nameS
with open(savename, 'wb') as f:
    pickle.dump(save_data, f)

end_time = datetime.now()
print(f"Total duration: {end_time - start_time}", flush=True)
