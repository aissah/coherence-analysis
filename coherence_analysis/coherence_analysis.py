r"""
Test coherence analyses a directory of DAS data.

This version uses dascore to read files and hence requires to given data to be
readable by dascore. Can be ran as:
python coherence_analysis.py <method> <data_location> <averaging_window_length>
    <sub_window_length> <overlap: optional, flag:-o> <time_range(optional): flag -t>
    <channel_range(optional): flag:-ch> <channel_offset(optional): flag:-ds>
    <time_step(optional): flag:-dt> <result_path(optional): flag:-r>

- method: method to use for coherence analysis
- data_location: path to the directory containing the data files
- averaging_window_length: Averaging window length in seconds
- sub_window_length: sub-window length in seconds
- overlap: overlap between sub-windows in seconds
Optional arguments:
- time_range(flags: "-t", "--time_range"): Range of time to use for coherence
analysis (in Python list format). Each time should have the format
"%m/%d/%y %H:%M:%S".
- channel_range(flags: "-ch", "--channel_range"): Range of channels to use for
coherence analysis (in Python list format).
- channel_offset(flags: "-ds", "--channel_offset"): Channels to skip
in-between
- time_step(flags: "-dt", "--time_step"): Sampling rate
- result_path(flags: "-r", "--result_path"): Directory to save results

The script will then go through the files in the directory provided that fall
within the ranges specified and perform coherence analysis on the data. The
results are saved to a file for later analysis.
Example:
- python coherence_analysis.py exact "D:\CSM\Mines_Research\Test_data\Port_Angeles"
    60 5 -o 0 -t "('06/01/23 07:32:09', ...)" -ch "(..., ...)"
    -ds 1 -dt 0.002 -r "D:\CSM\Mines_Research\Test_data\Port_Angeles\results"

"""

import argparse
import os
import pickle
from ast import literal_eval
from datetime import datetime

import dascore as dc
import numpy as np
from utils import coherence


class CoherenceAnalysis:
    """Class to perform coherence analysis on DAS data."""

    def __init__(self):
        # Define a list of methods to use for coherence analysis
        self.methods = ["exact", "qr", "svd", "rsvd"]

    def _parse_args(self):
        """Parse command line arguments.

        Raises
        ------
        ValueError
            Raise error if the method selected is not available.
        """
        # Initialize the parser
        parser = argparse.ArgumentParser(
            description="Coherence Analysis Configuration"
        )

        # Add arguments
        parser.add_argument(
            "method",
            type=str,
            choices=self.methods,
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
            "-dt",
            "--time_step",
            type=float,
            help="Sampling rate",
            default=0.002,
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

        if self.method not in self.methods:
            error_msg = (
                f"Method {self.method} not available for coherence analysis"
            )
            raise ValueError(error_msg)

    def read_data(self):
        """Read the data files and subselect according to input parameters using dascore."""
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
        # Using the distance array to select the channels with samples = False
        # This is a temporary solution to the issue of selecting channels
        # Only works for PRODML format
        # TODO: need to find a more general solution
        distance_coords = self.spool[0].coords.get_array("distance")
        distance_array = distance_coords[channels]
        # subsample the spool to select the channels and time range
        self.spool = self.spool.select(distance=(distance_array), samples=False)

        # self.spool = self.spool.select(time=self.time_range, samples=True)
        self.spool = self.spool.select(time=self.time_range)

        self.contents = self.spool.get_contents()
        self.time_step = self.contents["time_step"][0].total_seconds()

    def run(self):
        """Implement the coherence analysis using initialized parameters."""
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

        self.detection_significance = np.stack(
            [a[0] for a in map_out], axis=-1
        )
        self.eig_estimates = np.stack([a[1] for a in map_out], axis=-1)

    def save_results(self):
        """Save results to a file."""
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
        print(
            self.contents["time_min"][0],
            self.contents["time_max"][self.contents.index[-1]],
        )

        start_time = (
            str(self.contents["time_min"][0])
            .replace(" ", "_")
            .replace(":", "_")
            .replace(".", "_")
        )
        end_time = (
            str(self.contents["time_max"][self.contents.index[-1]])
            .replace(" ", "_")
            .replace(":", "_")
            .replace(".", "_")
        )

        # save the results of detection significance, eigenvalues,
        # and metadata to different files
        savename = os.path.join(
            self.save_location,
            f"{self.method}_detection_significance_"
            f"{start_time}_"
            f"{end_time}.pkl",
        )
        with open(savename, "wb") as f:
            pickle.dump(self.detection_significance, f)

        savename = os.path.join(
            self.save_location,
            f"{self.method}_eig_estimatess_"
            f"{start_time}_"
            f"{end_time}.pkl",
        )
        with open(savename, "wb") as f:
            pickle.dump(self.eig_estimates, f)

        savename = os.path.join(
            self.save_location,
            f"{self.method}_metadata_" f"{start_time}_" f"{end_time}.pkl",
        )
        with open(savename, "wb") as f:
            pickle.dump(metadata, f)


if __name__ == "__main__":
    # record start time
    start_time = datetime.now()

    # Initialize the coherence_analysis instance
    coherence_instance = CoherenceAnalysis()

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
