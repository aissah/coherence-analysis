"""
Test coherence analyses for a larger dataset.

The was written for some data from Brady Geothermal DAS experiment
and is in hdf5 format. Can be ran as:
python coherence_analysis_no_dascore.py <method> <data_location>
    <averaging_window_length> <sub_window_length> <overlap: optional, flag:-o>
    <channel_range(optional): flag:-ch> <channel_offset(optional): flag:-ds>
    <time_step(optional): flag:-dt> <result_path(optional): flag:-r>
    <batch(optional): flag:-b> <batch_size(optional): flag:-bs>
- data_location: path to the directory containing the data files
- averaging_window_length: Averaging window length in seconds
- sub_window_length: sub-window length in seconds
- overlap: overlap in seconds
- channel_range: range of channels to use for coherence analysis
- channel_offset: channels to skip in between selected channels
- time_step: seconds per sample
- method: method to use for coherence analysis
- result_path: directory to save results
- batch: Batch of files assuming jobs are run in parallel for files in batches.
    Should be one (1) if that is not the case.
- batch_size: Number of files in batch. Should be number of files being
    considered if job is not done in batches.
The script will then go through the files in the batch and perform coherence
analysis on the data. The results are saved to a file for later analysis.
Example:
- python coherence_analysis_no_dascore.py exact
    "/beegfs/projects/martin/BradyHotspring" 60 2 0 -o 0 -ch "(0, ...)"
    -ds 2 -dt 0.001
    -r "/u/st/by/aissah/scratch/coherence/coherence_test_results" -b 1 -bs 0

"""

import argparse
import os
import pickle
from ast import literal_eval
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import utils.utils as func


def _next_data_window(
    data_files: list[str],
    next_index: int,
    averaging_window_length: int,
    samples_per_sec: int,
    start_sample_index: int = 0,
):
    """
    Load the next data window from the data files.

    This function is used to load the next window of data from the list of
    data files. It continues to read data from the files until the window
    length is reached. The function returns the data, the index of the next
    file to read data from, and the index with the file at which we stopped
    reading.

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

    window_start_time = datetime.strptime(
        data_files[next_index][-15:-3], "%y%m%d%H%M%S"
    )
    window_start_time += timedelta(
        seconds=start_sample_index / samples_per_sec
    )

    data, _ = func.load_brady_hdf5(data_files[next_index], normalize="no")
    data = func.rm_laser_drift(data)
    data_len = data.shape[1]

    stop_sample_index = (
        start_sample_index + total_window_length
    )  # index we stopped reading data from file "next_index"
    # data = data[
    #     first_channel : channel_offset + first_channel : int(
    #         channel_offset / num_channels
    #     ),
    #     start_sample_index:stop_sample_index,
    # ]
    data = data[
        first_channel:last_channel:channel_offset,
        start_sample_index:stop_sample_index,
    ]

    # number of samples to add to the data to make up the window length
    window_deficit = total_window_length - data.shape[1]

    if window_deficit == 0 and stop_sample_index == data_len:
        next_index += 1
        stop_sample_index = 0

    ignored_files = []

    while window_deficit > 0 and next_index < num_files - 1:
        next_index += 1  # index of the next file to read data from
        file_start_time = datetime.strptime(
            data_files[next_index][-15:-3], "%y%m%d%H%M%S"
        )
        if file_start_time - window_start_time > timedelta(
            seconds=int(data.shape[1] / samples_per_sec) + 1
        ):
            ignored_files.append(data_files[next_index - 1])

            window_start_time = file_start_time
            data, _ = func.load_brady_hdf5(
                data_files[next_index],
                normalize="no",
            )
            data = func.rm_laser_drift(data)
            # data = data[
            #     first_channel : channel_offset + first_channel : int(
            #         channel_offset / num_channels
            #     ),
            #     :total_window_length,
            # ]
            data = data[
                first_channel:last_channel:channel_offset,
                :total_window_length,
            ]
            window_deficit = total_window_length - data.shape[1]
            if window_deficit == 0:
                next_index += 1
                stop_sample_index = 0
        else:
            next_data, _ = func.load_brady_hdf5(
                data_files[next_index],
                normalize="no",
            )
            next_data = func.rm_laser_drift(next_data)
            # next_data = next_data[
            #     first_channel : channel_offset + first_channel : int(
            #         channel_offset / num_channels
            #     )
            # ]
            next_data = next_data[first_channel:last_channel:channel_offset]
            data = np.append(data, next_data[:, :window_deficit], axis=1)

            if window_deficit < next_data.shape[1]:
                stop_sample_index = window_deficit
            elif (
                window_deficit == next_data.shape[1]
                or next_index == num_files - 1
            ):
                next_index += 1
                stop_sample_index = 0

            window_deficit = total_window_length - data.shape[1]

    window_end_time = window_start_time + timedelta(
        seconds=total_window_length / samples_per_sec
    )

    return (
        data,
        next_index,
        stop_sample_index,
        window_start_time,
        window_end_time,
        ignored_files,
    )


def parse_args():
    """Parse command line arguments.

    Raises
    ------
    ValueError
        Raise error if the method selected is not available.
    """
    methods = ["exact", "qr", "svd", "rsvd"]
    # Initialize the parser
    parser = argparse.ArgumentParser(
        description="Coherence Analysis Configuration"
    )

    # Add arguments
    parser.add_argument(
        "method",
        type=str,
        choices=methods,
        help="Method to use for coherence analysis",
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the directory containing the data files",
    )
    parser.add_argument(
        "averaging_window_length",
        type=int,
        help="Averaging window length in seconds",
    )
    parser.add_argument(
        "sub_window_length", type=int, help="Sub-window length in seconds"
    )
    parser.add_argument(
        "-o", "--overlap", type=int, help="Overlap in seconds", default=0
    )
    parser.add_argument(
        "-t",
        "--time_range",
        type=str,
        help="Range of time to use for coherence analysis "
        "(in Python list format)",
        default="(..., ...)",
    )
    parser.add_argument(
        "-ch",
        "--channel_range",
        type=str,
        help="Range of channels to use for coherence analysis "
        " (in Python list format)",
        default="(0, ...)",
    )
    parser.add_argument(
        "-ds",
        "--channel_offset",
        type=int,
        help="Channels to skip in between",
        default=1,
    )
    parser.add_argument(
        "-dt",
        "--time_step",
        type=float,
        help="Seconds per sample",
        default=None,
    )
    parser.add_argument(
        "-r",
        "--result_path",
        type=str,
        help="Directory to save results",
        default=os.path.join(
            os.path.dirname(__file__), os.pardir, "data/results"
        ),
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        help="Which batch of files to process",
        default=1,
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        help="Number of files in each batch",
        type=int,
        default=0,
    )

    return parser.parse_args()


if __name__ == "__main__":
    # record start time
    start_time = datetime.now()

    # list of methods to use for coherence analysis
    METHODS = ["exact", "qr", "svd", "rsvd", "power", "qr iteration"]

    # Take inputs from the command line
    args = parse_args()

    # Path to the directory containing the data files
    data_basepath = args.data_path
    # Path to the directory where the results will be saved
    save_location = args.result_path
    # Averaging window length in seconds
    averaging_window_length = args.averaging_window_length
    # sub-window length in seconds
    sub_window_length = args.sub_window_length
    # overlap in seconds
    overlap = args.overlap
    channel_range = literal_eval(args.channel_range)
    # first channel
    first_channel = 0 if channel_range[0] == ... else channel_range[0]
    # channel offset
    # Number of channels to choose from
    channel_offset = args.channel_offset
    # Number of channels to subselect from the range of channels
    # num_channels = int(sys.argv[7])
    last_channel = -1 if channel_range[1] == ... else channel_range[1]
    # seconds per sample
    samples_per_sec = 1 / args.time_step
    # method to use for coherence analysis
    method = args.method
    # Batch of files assuming jobs are run in parallel for files in batches.
    # Should be one if that is not the case.
    batch = args.batch
    # Number of files in batch. Should be 0 or number of files being
    # considered if job is not done in batches.
    batch_size = args.batch_size

    # Path to the directory containing the data files
    # data_basepath = "/beegfs/projects/martin/BradyHotspring"
    # "D:/CSM/Mines_Research/Test_data/Brady Hotspring"

    # Path to the directory where the results will be saved
    # save_location="/u/st/by/aissah/scratch/coherence/coherence_test_results"

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
                if ".h5" in file_name and file_name[0] != "."
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
    # metadata["first_channel"] = first_channel
    metadata["channel_range"] = channel_range
    # metadata["num_channels"] = num_channels
    metadata["channel_offset"] = channel_offset
    metadata["method"] = method

    print(f"Starting {method} method with {metadata}", flush=True)

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
    (
        data,
        next_index,
        stop_sample_index,
        window_start_time,
        window_end_time,
        ignored_files,
    ) = _next_data_window(
        data_files, next_index, averaging_window_length, samples_per_sec
    )
    window_start_times = [window_start_time]
    window_end_times = [window_end_time]
    ignored_files = ignored_files
    # work on files after first file in batch. This works exactly as we
    # handled the beginning of later batches. Then we keep appending to
    # the variables set up for first file of the batch above
    if method in METHODS:
        detection_significances, eig_estimatess, _ = func.coherence(
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
        (
            data,
            next_index,
            stop_sample_index,
            window_start_time,
            window_end_time,
            ignored_files,
        ) = _next_data_window(
            data_files,
            next_index,
            averaging_window_length,
            samples_per_sec,
            stop_sample_index,
        )
        window_start_times.append(window_start_time)
        window_end_times.append(window_end_time)
        ignored_files.extend(ignored_files)

        if data.shape[1] == averaging_window_length * samples_per_sec:
            detection_significance, eig_estimates, _ = func.coherence(
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
                f" {averaging_window_length * samples_per_sec} for analysis."
                f" {len(data_files) - next_index} files still remaining ",
                flush=True,
            )

        if next_index % 500 == 0:
            print(
                f"Processed {next_index} files. "
                f"in {datetime.now() - start_time} for {method} method. "
                f"{len(data_files) - next_index} files remaining.",
                flush=True,
            )

    metadata["ignored_files"] = ignored_files
    metadata["window_start_times"] = window_start_times
    metadata["window_end_times"] = window_end_times

    print(
        f"Finished in: {datetime.now() - start_time} for {method} method."
        " Saving to file...",
        flush=True,
    )

    # Create the result directory if it does not exist
    Path(save_location).mkdir(parents=True, exist_ok=True)
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
