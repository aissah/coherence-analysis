import os
import pickle
import sys
from datetime import datetime
import numpy as np


event_id = int(sys.argv[1])  # Used to select the event template. Used: 2201050
first_channel = int(sys.argv[2])  # First channel in range of channels used. Used: 1000
last_channel = int(sys.argv[3])  # Last channel in range of channels used: Used:5000
batch = int(
    sys.argv[4]
)  # Batch of files assuming jobs are run in parallel for files in batches. Should be one if that is not the case.
batch_size = int(
    sys.argv[5]
)  # Number of files in batch. Should be number of files being considered if job is not done in batches
compression_flag = int(
    sys.argv[6]
)  # 1 if compressed data is used otherwise uncompressed data is used


data_basepath = "/beegfs/projects/martin/BradyHotspring"  # "D:/CSM/Mines_Research/Test_data/Brady Hotspring"
# files = os.listdir(data_basepath)
save_location = "/u/st/by/aissah/scratch/event_detection/template_matching"  # "D:/CSM/Mines_Research/Test_data/"


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


if batch == 1:
    first_file_time = data_files[0][-15:-3]

    data_files = data_files[:batch_size]
    data, _ = general_funcs.loadBradyHShdf5(data_files[0], normalize="no")
    data = data[first_channel:last_channel]

else:  # with more batches, append end of previous file for continuity
    try:
        data_files = data_files[(batch - 1) * batch_size - 1 : batch * batch_size]
        metadata["files"] = [a[-15:-3] for a in data_files]
    except IndexError:
        data_files = data_files[(batch - 1) * batch_size - 1 :]
        metadata["files"] = [a[-15:-3] for a in data_files]
    data, _ = general_funcs.loadBradyHShdf5(data_files[1], normalize="no")
    metadata["start_lag"] = (
        (batch - 1) * batch_size * len(data[0]) - len(template[0]) + 1
    )

    preceding_data, _ = general_funcs.loadBradyHShdf5(data_files[0], normalize="no")
    data = np.append(
        preceding_data[first_channel:last_channel, -len(template[1]) + 1 :],
        data[first_channel:last_channel],
        axis=1,
    )

# work on files after first file in batch. This works exactly as we handled the
# beginning of later batches. Then we keep appending to the variables set up for
# first file of the batch above

end_time = datetime.now()
print(f"Duration: {end_time - start_time}", flush=True)

for a in data_files[1:]:
    preceding_data = data[:, -len(template[1]) + 1 :]
    data, _ = general_funcs.loadBradyHShdf5(a, normalize="no")
    data = np.append(preceding_data, data[first_channel:last_channel], axis=1)

if compression_flag == 0:
    savefile_name = (
        save_location
        + "/"
        + str(event_id)
        + "uncompressed_batch"
        + str(batch)
        + "_"
        + metadata["files"][0]
        + "_"
        + metadata["files"][-1]
    )
else:
    savefile_name = (
        save_location
        + "/"
        + str(event_id)
        + "_"
        + compression_type
        + "_"
        + "_".join([str(int(a)) for a in metadata["compression_rates"][0]])
        + "_batch"
        + str(batch)
        + "_"
        + metadata["files"][0]
        + "_"
        + metadata["files"][-1]
    )

with open(savefile_name, "wb") as f:
    pickle.dump([mean_ccs_acrossfiles, metadata], f)

end_time = datetime.now()
print(f"Total duration: {end_time - start_time}", flush=True)
