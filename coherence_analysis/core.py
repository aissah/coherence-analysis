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
- python large_scale_tests_parser.py exact "D:\CSM\Mines_Research\Test_data\Port_Angeles"
    "('06/01/23 07:32:09', ...)" "(..., ...)" 1 60 5 0 0.002

"""

import argparse
import os
import pickle
from ast import literal_eval
from datetime import datetime

import dascore as dc
import numpy as np
from utils import coherence


class coherence_analysis:

    def __init__(self):
        pass

    def _parse_args(self):
        # Define a list of methods to use for coherence analysis
        METHODS = ["exact", "qr", "svd", "rsvd", "power", "qr iteration"]

        # Initialize the parser
        parser = argparse.ArgumentParser(description="Coherence Analysis Configuration")

        # Add arguments
        parser.add_argument(
            "method",
            type=str,
            choices=METHODS,
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
            help="Range of time to use for coherence analysis (in Python list format)",
            default="(..., ...)",
        )
        parser.add_argument(
            "-ch",
            "--channel_range",
            type=str,
            help="Range of channels to use for coherence analysis (in Python list format)",
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
            "-dt", "--time_step", type=float, help="Sampling rate", default=0.002
        )
        parser.add_argument(
            "-r",
            "--result_path",
            type=str,
            help="Directory to save results",
            default=os.path.join( os.path.dirname( __file__ ), '../data/results' ),
        )

        # Parse arguments
        args = parser.parse_args()

        # Convert time_range and channel_range from strings to lists using literal_eval
        self.time_range = [
            datetime.strptime(a, "%m/%d/%y %H:%M:%S") if a != ... else ...
            for a in literal_eval(args.time_range)
        ]
        self.channel_range = literal_eval(args.channel_range)

        # Access the parsed arguments
        self.data_path = args.data_path
        self.save_location = args.result_path
        self.channel_offset = args.channel_offset
        self.averaging_window_length = args.averaging_window_length
        self.sub_window_length = args.sub_window_length
        self.overlap = args.overlap
        self.time_step = args.time_step
        self.method = args.method

        if self.method not in METHODS:
            error_msg = f"Method {self.method} not available for coherence analysis"
            raise ValueError(error_msg)

    def read_data(self):

        # read the data files using the spool function from dascore
        self.spool = dc.spool(self.data_path)

        # chunk the spool into averaging_window length
        self.spool = self.spool.chunk(time=self.averaging_window_length)

        # subselect n_channels number of channels starting from start_channel
        channels = np.arange(
            self.channel_range[0] if self.channel_range[0] is not ... else 0,
            (
                self.channel_range[1]
                if self.channel_range[1] is not ...
                else self.spool[0].data.shape[1]
            ),
            self.channel_offset,
            dtype=int,
        )

        # subsample the spool to select the channels and time range
        self.spool = self.spool.select(distance=(channels), samples=True)
        self.spool = self.spool.select(time=self.time_range, samples=True)

        self.contents = self.spool.get_contents()
        self.time_step = self.contents["time_step"][0].total_seconds()

    def run(self):
        # perform coherence calculation on each patch
        map_out = coherence_instance.spool.map(
            lambda x: coherence(
                x.data.T,
                coherence_instance.sub_window_length,
                coherence_instance.overlap,
                sample_interval=coherence_instance.time_step,
                method=coherence_instance.method,
            )
        )

        self.detection_significance = np.stack([a[0] for a in map_out], axis=-1)
        self.eig_estimates = np.stack([a[1] for a in map_out], axis=-1)

    def save_results(self):
        # create a dictionary to store metadata
        metadata = {}
        metadata["time_step"] = self.time_step
        metadata["averaging_window_length"] = self.averaging_window_length
        metadata["sub_window_length"] = self.sub_window_length
        metadata["overlap"] = self.overlap
        metadata["channel_range"] = self.channel_range
        metadata["channel_offset"] = self.channel_offset
        metadata["method"] = self.method
        metadata["times"] = self.contents[["time_min", "time_max"]]

        print(self.contents[["time_min", "time_max"]])
        print(self.contents["time_min"][0], self.contents["time_max"][self.contents.index[-1]])

        start_time = str(self.contents["time_min"][0]).replace(" ", "_").replace(":", "_").replace(".", "_")
        end_time = str(self.contents["time_max"][self.contents.index[-1]]).replace(" ", "_").replace(":", "_").replace(".", "_")

        # save the results of detection significance, eigenvalues,
        # and metadata to different files
        savename = os.path.join(
            self.save_location,
            f"{self.method}_detection_significance_"
            f"{start_time}_"
            f"{end_time}.pkl"
        )
        with open(savename, "wb") as f:
            pickle.dump(self.detection_significance, f)

        savename = os.path.join(
            self.save_location,
            f"{self.method}_eig_estimatess_"
            f"{start_time}_"
            f"{end_time}.pkl"
        )
        with open(savename, "wb") as f:
            pickle.dump(self.eig_estimates, f)

        savename = os.path.join(
            self.save_location,
            f"{self.method}_metadata_"
            f"{start_time}_"
            f"{end_time}.pkl"
        )
        with open(savename, "wb") as f:
            pickle.dump(metadata, f)


if __name__ == "__main__":
    # record start time
    start_time = datetime.now()

    # Initialize the coherence_analysis instance
    coherence_instance = coherence_analysis()

    # Parse the command line arguments
    coherence_instance._parse_args()

    # Print the parsed arguments
    print(f"Data Path: {coherence_instance.data_path}")
    print(f"Time Range: {coherence_instance.time_range}")
    print(f"Channel Range: {coherence_instance.channel_range}")
    print(f"Method: {coherence_instance.method}")
    print(f"Result Path: {coherence_instance.save_location}")

    # Read the data
    print("Reading data...", flush=True)
    coherence_instance.read_data()

    end_time = datetime.now()
    print(f"Data read in: {end_time - start_time}", flush=True)

    # run the coherence analysis
    print("Running coherence analysis...", flush=True)
    coherence_instance.run()

    # save the results
    print(
        f"Finished in: {datetime.now()-start_time} for {coherence_instance.method} method."
        " Saving results...",
        flush=True,
    )
    coherence_instance.save_results()

    end_time = datetime.now()
    print(f"Total duration: {end_time - start_time}", flush=True)
