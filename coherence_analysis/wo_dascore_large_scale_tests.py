"""
This python file is to test coherence analyses for a larger dataset. The was
written for some data from Brady Geothermal DAS experiment and is in hdf5
format. Can be ran as:
TO DO: Add output directory as an argument
python large_scale_test.py <data_location> <averaging_window_length>
    <sub_window_length> <overlap> <first_channel> <channel_offset>
        <num_channels> <samples_per_sec> <method> <batch> <batch_size>
- data_location: path to the directory containing the data files
- averaging_window_length: Averaging window length in seconds
- sub_window_length: sub-window length in seconds
- overlap: overlap in seconds
- first_channel: first channel
- num_channels: number of sensors
- samples_per_sec: samples per second
- channel_offset: channel offset
- method: method to use for coherence analysis
- batch: Batch of files assuming jobs are run in parallel for files in batches.
    Should be one (1) if that is not the case.
- batch_size: Number of files in batch. Should be number of files being
    considered if job is not done in batches.
The script will then go through the files in the batch and perform coherence
analysis on the data. The results are saved to a file for later analysis.
Example:
- python large_scale_test.py "/beegfs/projects/martin/BradyHotspring" 60 2 0
    3100 2000 200 1000 exact 1 0

"""

import os
import pickle
import sys
from datetime import datetime

import numpy as np
import utils as func


def _next_data_window(
    data_files: list[str],
    next_index: int,
    averaging_window_length: int,
    samples_per_sec: int,
    start_sample_index: int = 0,
):
    """
    Load the next data window from the data files. This function is used to
    load the next window of data from the list of data files. It continues
    to read data from the files until the window length is reached. The
    function returns the data, the index of the next file to read data from,
    and the index with the file at which we stopped reading.

    Parameters
    ----------
    data_file : list[str]
        list of the data files to read data from
    next_index : int
        index of the next file to read data from
    averaging_window_length : int
        length of the averaging window in seconds
    samples_per_sec : int
        number of samples per second in the data
    start_sample_index : int
        index of the first sample to read from the next data file

    Returns
    -------
    data : np array
        data read from the data files
    next_index : int
        index of the next file to read data from
    stop_sample_index : int
        index we stopped reading data from file "next_index"

    """
    num_files = len(data_files)
    total_window_length = averaging_window_length * samples_per_sec
    data, _ = func.load_brady_hdf5(data_files[next_index], normalize="no")
    data_len = data.shape[1]
    data = data[
        first_channel : channel_offset + first_channel : int(
            channel_offset / num_channels
        ),
        start_sample_index : start_sample_index + total_window_length,
    ]

    stop_sample_index = (
        start_sample_index + total_window_length
    )  # index we stopped reading data from file "next_index"

    # number of samples to add to the data to make up the window length
    window_deficit = total_window_length - data.shape[1]

    if window_deficit == 0 and stop_sample_index == data_len:
        next_index += 1
        stop_sample_index = 0

    while window_deficit > 0 and next_index < num_files - 1:
        next_index += 1  # index of the next file to read data from
        next_data, _ = func.loadBradyHShdf5(
            data_files[next_index],
            normalize="no",
        )
        next_data = func.rm_laser_drift(next_data)
        next_data = next_data[
            first_channel : channel_offset + first_channel : int(
                channel_offset / num_channels
            )
        ]
        data = np.append(data, next_data[:, :window_deficit], axis=1)

        if window_deficit < next_data.shape[1]:
            stop_sample_index = window_deficit
        elif (
            window_deficit == next_data.shape[1] or next_index == num_files - 1
        ):
            next_index += 1
            stop_sample_index = 0

        window_deficit = total_window_length - data.shape[1]

    return data, next_index, stop_sample_index


if __name__ == "__main__":
    # record start time
    start_time = datetime.now()

    # list of methods to use for coherence analysis
    METHODS = ["exact", "qr", "svd", "rsvd", "power", "qr iteration"]

    # Take inputs from the command line
    # Path to the directory containing the data files
    data_basepath = sys.argv[1]
    # Path to the directory where the results will be saved
    # save_location = sys.argv[12]
    # Averaging window length in seconds
    averaging_window_length = int(sys.argv[2])
    sub_window_length = int(sys.argv[3])  # sub-window length in seconds
    overlap = int(sys.argv[4])  # overlap in seconds
    first_channel = int(sys.argv[5])  # first channel
    channel_offset = int(sys.argv[6])  # Number of channels to choose from
    num_channels = int(
        sys.argv[7]
    )  # Number of channels to subselect from the range of channels
    samples_per_sec = int(sys.argv[8])  # samples per second
    method = sys.argv[9]  # method to use for coherence analysis
    # Batch of files assuming jobs are run in parallel for files in batches.
    # Should be one if that is not the case.
    batch = int(sys.argv[10])
    # Number of files in batch. Should be 0 or number of files being
    # considered if job is not done in batches.
    batch_size = int(sys.argv[11])

    # Path to the directory containing the data files
    # data_basepath = "/beegfs/projects/martin/BradyHotspring"
    # "D:/CSM/Mines_Research/Test_data/Brady Hotspring"

    # Path to the directory where the results will be saved
    save_location = "/u/st/by/aissah/scratch/coherence/coherence_test_results"
    # "D:/CSM/Mines_Research/Test_data/"

    # Get the file names of the data files by going through the folders
    # contained in the base path and putting together the paths to files
    # ending in .h5
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

    # use all the files if batch size is specified as 0
    batch_size = len(data_files) if batch_size == 0 else batch_size

    # create a dictionary to store the metadata of the files
    metadata = {}
    metadata["sampling_rate"] = samples_per_sec
    metadata["averaging_window_length"] = averaging_window_length
    metadata["sub_window_length"] = sub_window_length
    metadata["overlap"] = overlap
    metadata["first_channel"] = first_channel
    metadata["num_channels"] = num_channels
    metadata["channel_offset"] = channel_offset
    metadata["method"] = method

    # load the first file in the batch
    if batch == 1:
        first_file_time = data_files[0][-15:-3]
        data_files = data_files[:batch_size]
        metadata["files"] = [a[-15:-3] for a in data_files]
    else:  # with more batches, append end of previous file for continuity
        try:
            data_files = data_files[
                (batch - 1) * batch_size - 1 : batch * batch_size
            ]
            metadata["files"] = [a[-15:-3] for a in data_files]
        except IndexError:
            data_files = data_files[(batch - 1) * batch_size - 1 :]
            metadata["files"] = [a[-15:-3] for a in data_files]

    next_index = 0
    data, next_index, stop_sample_index = _next_data_window(
        data_files, next_index, averaging_window_length, samples_per_sec
    )

    # work on files after first file in batch. This works exactly as we
    # handled the beginning of later batches. Then we keep appending to
    # the variables set up for first file of the batch above
    if method in METHODS:
        detection_significances, eig_estimatess = func.coherence(
            data,
            sub_window_length,
            overlap,
            sample_interval=1 / samples_per_sec,
            method=method,
        )
    else:
        error_msg = f"Method {method} not available for coherence analysis"
        raise ValueError(error_msg)

    end_time = datetime.now()
    print(f"First file completed in: {end_time - start_time}", flush=True)

    # for a in data_files[1:]:
    while next_index < len(data_files) - 1:
        data, next_index, stop_sample_index = _next_data_window(
            data_files,
            next_index,
            averaging_window_length,
            samples_per_sec,
            stop_sample_index,
        )

        if data.shape[1] == averaging_window_length * samples_per_sec:
            detection_significance, eig_estimates = func.coherence(
                data,
                sub_window_length,
                overlap,
                sample_interval=1 / samples_per_sec,
                method=method,
            )

            if detection_significance.shape == detection_significances.shape:
                detection_significances = np.append(
                    detection_significances[np.newaxis],
                    detection_significance[np.newaxis],
                    axis=0,
                )
            else:
                detection_significances = np.append(
                    detection_significances,
                    detection_significance[np.newaxis],
                    axis=0,
                )

            eig_estimatess = np.append(eig_estimatess, eig_estimates, axis=1)
        else:
            print(
                f"Data length of {data.shape[1]} not the expected"
                " {averaging_window_length * samples_per_sec} for analysis."
                " {len(data_files) - next_index} files still remaining ",
                flush=True,
            )

    print(
        f"Finished in: {datetime.now()-start_time} for {method} method."
        " Saving to file...",
        flush=True,
    )

    # save the results of detection significance, eigenvalues, and metadata to
    # different files
    savename = os.path.join(
        save_location,
        f"{method}_detection_significance_{metadata['files'][0]}_{metadata['files'][-1]}.pkl",
    )
    with open(savename, "wb") as f:
        pickle.dump(detection_significances, f)

    savename = os.path.join(
        save_location,
        f"{method}_eig_estimatess_{metadata['files'][0]}_{metadata['files'][-1]}.pkl",
    )
    with open(savename, "wb") as f:
        pickle.dump(eig_estimatess, f)

    savename = os.path.join(
        save_location,
        f"{method}_metadata_{metadata['files'][0]}_{metadata['files'][-1]}.pkl",
    )
    with open(savename, "wb") as f:
        pickle.dump(metadata, f)

    end_time = datetime.now()
    print(f"Total duration: {end_time - start_time}", flush=True)
