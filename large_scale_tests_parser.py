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
import argparse
from ast import literal_eval
from datetime import datetime
import os
import pickle

import dascore as dc
import numpy as np
        

if __name__ == "__main__":
    # record start time
    start_time = datetime.now()

    # Define a list of methods to use for coherence analysis
    METHODS = ["exact", "qr", "svd", "rsvd", "power", "qr iteration"]

    # Initialize the parser
    parser = argparse.ArgumentParser(description="Coherence Analysis Configuration")

    # Add arguments
    parser.add_argument('data_path', type=str, help='Path to the directory containing the data files')
    parser.add_argument('result_path', type=str, help='Directory to save results')
    parser.add_argument('time_range', type=str, help='Range of time to use for coherence analysis (in Python list format)')
    parser.add_argument('channel_range', type=str, help='Range of channels to use for coherence analysis (in Python list format)')
    parser.add_argument('channel_offset', type=int, help='Channels to skip in between')
    parser.add_argument('averaging_window_length', type=int, help='Averaging window length in seconds')
    parser.add_argument('sub_window_length', type=int, help='Sub-window length in seconds')
    parser.add_argument('overlap', type=int, help='Overlap in seconds')
    parser.add_argument('time_step', type=int, help='Seconds per sample', default=0.002)
    parser.add_argument('method', type=str, choices=METHODS, help='Method to use for coherence analysis')

    # Parse arguments
    args = parser.parse_args()

    # Convert time_range and channel_range from strings to lists using literal_eval
    time_range = [datetime.strptime(a, '%m/%d/%y %H:%M:%S') if a != ... else ... for a in literal_eval(args.time_range)]
    channel_range = literal_eval(args.channel_range)

    # Access the parsed arguments
    data_path = args.data_path
    save_location = args.result_path
    channel_offset = args.channel_offset
    averaging_window_length = args.averaging_window_length
    sub_window_length = args.sub_window_length
    overlap = args.overlap
    time_step = args.time_step
    method = args.method

    # Example of accessing the parsed arguments
    print(f"Data Path: {data_path}")
    print(f"Time Range: {time_range}")
    print(f"Channel Range: {channel_range}")
    print(f"Method: {method}")

    # Path to the directory containing the data files
    # data_basepath = "/beegfs/projects/martin/BradyHotspring"
    # "D:/CSM/Mines_Research/Test_data/Brady Hotspring"

    # Path to the directory where the results will be saved
    save_location = "/u/st/by/aissah/scratch/coherence/coherence_test_results"
    # "D:/CSM/Mines_Research/Test_data/"

    # Get the file names of the data files by going through the folders
    # contained in the base path and putting together the paths to files
    # ending in .h5
    spool = dc.spool(data_path)

    # chunk the spool into averaging_window length
    spool = spool.chunk(time=averaging_window_length)

    # subselect n_channels number of channels starting from start_channel
    channels = np.arange(
        channel_range[0],
        channel_range[1],
        channel_offset,
        dtype=int,
    )

    spool = spool.select(distance=(channels), samples=True)
    spool = spool.select(time=(channels), samples=True)

    # another way to subselect channels
    # sub_patch = patch.select(distance=np.array([0, 12, 10, 9]), samples=True)

    end_time = datetime.now()
    print(f"Data read in: {end_time - start_time}", flush=True)

    contents = spool.contents
    time_step = contents["time_step"][0].total_seconds()
    # sample_interval = 1 / samples_per_sec

    # use all the files if batch size is specified as 0
    # batch_size = len(data_files) if batch_size == 0 else batch_size

    # create a dictionary to store the metadata of the files
    metadata = {}
    metadata["time_step"] = time_step
    metadata["averaging_window_length"] = averaging_window_length
    metadata["sub_window_length"] = sub_window_length
    metadata["overlap"] = overlap
    metadata["channel_range"] = channel_range
    metadata["channel_offset"] = channel_offset
    metadata["method"] = method
    metadata["times"] = contents[["time_min", "time_max"]]

    # work on files after first file in batch. This works exactly as we
    # handled the beginning of later batches. Then we keep appending to
    # the variables set up for first file of the batch above
    if method in METHODS:
        # perform coherence calculation on each patch
        map_out = spool.map(
            lambda x: f.coherence(
                x.data.T,
                sub_window_length,
                overlap,
                sample_interval=time_step,
                method=method,
            )
        )

        detection_significance = np.stack([a[0] for a in map_out], axis=-1)
        eig_estimates = np.stack([a[1] for a in map_out], axis=-1)
    else:
        error_msg = f"Method {method} not available for coherence analysis"
        raise ValueError(error_msg)

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
        pickle.dump(detection_significance, f)

    savename = os.path.join(
        save_location,
        f"{method}_eig_estimatess_{metadata['files'][0]}_{metadata['files'][-1]}.pkl",
    )
    with open(savename, "wb") as f:
        pickle.dump(eig_estimates, f)

    savename = os.path.join(
        save_location,
        f"{method}_metadata_{metadata['files'][0]}_{metadata['files'][-1]}.pkl",
    )
    with open(savename, "wb") as f:
        pickle.dump(metadata, f)

    end_time = datetime.now()
    print(f"Total duration: {end_time - start_time}", flush=True)
