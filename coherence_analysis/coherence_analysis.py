r"""
Test coherence analyses a directory of DAS data.

This version uses dascore to read files and hence requires to given data to be
readable by dascore. Can be ran as:
python coherence_analysis.py <method> <data_location> <averaging_window_length>
    <sub_window_length> <overlap: optional, flag:-o>
    <time_range(optional): flag -t>
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
- python coherence_analysis.py exact
    "D:\CSM\Mines_Research\Test_data\Port_Angeles"
    60 5 -o 0 -t "('06/01/23 07:32:09', ...)" -ch "(..., ...)"
    -ds 1 -dt 0.002 -r "D:\CSM\Mines_Research\Test_data\Port_Angeles\results"

"""

import argparse
import os
import pickle
import sys
from ast import literal_eval
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import dascore as dc
import numpy as np

sys.path.append(os.path.join(os.path.dirname(""), os.pardir))
# from coherence_analysis.utils.utils import coherence
from utils.utils import coherence


class CoherenceAnalysis:
    """Class to perform coherence analysis on DAS data."""

    def __init__(self, args: dict = None):
        """
        Initialize the coherence analysis with input parameters.

        Parameters
        ----------
        args : dict, optional
            Dictionary of input parameters, by default None
            Expected keys are:
            - data_path: Path to the directory containing the data files
            - time_range: Range of time to use for coherence analysis
            (in list format)
            - channel_range: Range of channels to use for coherence analysis
            (in tuple format)
            - channel_offset: Channels to skip in between
            - averaging_window_length: Averaging window length in seconds
            - sub_window_length: Sub-window length in seconds
            - overlap: Overlap in seconds
            - time_step: Seconds per sample
            - method: Method to use for coherence analysis
            (one of "exact", "qr", "svd", "rsvd")
            - result_path: Directory to save results

        """
        self.methods = ["exact", "qr", "svd", "rsvd"]
        if args is not None:
            # Convert time_range and channel_range from strings to lists
            self.time_range = [
                datetime.strptime(a, "%m/%d/%y %H:%M:%S") if a != ... else ...
                for a in literal_eval(args["time_range"])
            ]
            self.channel_range = literal_eval(args["channel_range"])

            # Access the parsed arguments
            self.data_path = args["data_path"]
            self.save_location = args["result_path"]
            self.channel_offset = args["channel_offset"]
            self.averaging_window_length = args["averaging_window_length"]
            self.sub_window_length = args["sub_window_length"]
            self.overlap = args["overlap"]
            self.time_step = args["time_step"]
            self.method = args["method"]

            if self.method not in self.methods:
                error_msg = f"""
                Method {self.method} not available for coherence analysis.
                Available methods are: {self.methods}
                """
                raise ValueError(error_msg)

            print(f"""Initialized with the following parameters:
            data_path: {self.data_path}
            time_range: {self.time_range}
            channel_range: {self.channel_range}
            channel_offset: {self.channel_offset}
            averaging_window_length: {self.averaging_window_length}
            sub_window_length: {self.sub_window_length}
            overlap: {self.overlap}
            time_step: {self.time_step}
            method: {self.method}
            save_location: {self.save_location}
            """)
        else:
            # Initialize parameters to None
            self.data_path = os.path.join(
                os.path.dirname(__file__), os.pardir, "data", "rawdata"
            )
            self.time_range = [..., ...]
            self.channel_range = (..., ...)
            self.channel_offset = None
            self.averaging_window_length = None
            self.sub_window_length = None
            self.overlap = None
            self.time_step = None
            self.method = None
            self.save_location = os.path.join(
                os.path.dirname(__file__), os.pardir, "data", "results"
            )
            print(f"""No arguments provided. These attributes are set to
                default values:

                data_path: {self.data_path}
                time_range: {self.time_range}
                channel_range: {self.channel_range}
                channel_offset: {self.channel_offset}
                averaging_window_length: {self.averaging_window_length}
                sub_window_length: {self.sub_window_length}
                overlap: {self.overlap}
                time_step: {self.time_step}
                method: {self.method}
                save_location: {self.save_location}

                These can be set manually to desired values.
                """)

    def _set_channel_dim(self, channel_dim: str = None):
        """Set the channel dimension to 'channel' if not already set."""
        first_patch = self.spool[0]
        if channel_dim is None:
            dims = first_patch.dims
            print(f"The data has the following dimensions: {dims}")
            print(f"""Channels will be grouped based on the '{dims[1]}'
                  dimension. If another dimension is desired, use the
                  method, '_set_channel_dim()' to set it.""")
            channel_dim = dims[1]
        self.channel_dim = channel_dim

        try:
            start_ch = (
                0 if self.channel_range[0] == ... else self.channel_range[0]
            )
            end_ch = (
                first_patch.coords.get_array(self.channel_dim).shape[0]
                if self.channel_range[1] == ...
                else self.channel_range[1]
            )
            print(f"Channels will be selected from {start_ch} to {end_ch}.")
            self.channel_range = (start_ch, end_ch)
        except AttributeError:
            print("Error: ")
            print(
                f"""
                The dimension '{self.channel_dim}' does not exist in the data.
                """
            )
            print("Available dimensions are:")
            for dim in dims:
                print(f"- {dim}")
            raise AttributeError(
                f"The dimension '{self.channel_dim}' does not exist in data."
            )

        channels = np.arange(
            self.channel_range[0],
            self.channel_range[1],
            self.channel_offset,
            dtype=int,
        )
        distance_coords = first_patch.coords.get_array(self.channel_dim)
        self.distance_array = distance_coords[channels]

    def read_data(self):
        """Read the files and subselect using dascore."""
        # read the data files using the spool function from dascore
        self.spool = dc.spool(self.data_path)
        # get the time step from the spool
        print(self.spool)
        try:
            self.time_step = self.spool.get_contents()["time_step"].iloc[0]
        except (KeyError, IndexError):
            if self.time_step is None:
                raise ValueError(
                    "Time step not found in data or input parameters"
                )
        # chunk the spool into averaging_window length
        self.spool = self.spool.chunk(
            time=self.averaging_window_length, keep_partial=True
        )

        self.spool = self.spool.select(time=self.time_range, samples=True)

        # set the channel dimension
        self._set_channel_dim()

        # subselect n_channels number of channels starting from start_channel
        # channels = np.arange(
        #     self.channel_range[0],
        #     self.channel_range[1],
        #     self.channel_offset,
        #     dtype=int,
        # )
        # Using the distance array to select the channels with samples = False
        # This is a temporary solution to the issue of selecting channels
        # Only works for PRODML format
        # TODO: need to find a more general solution
        # distance_coords = self.spool[0].coords.get_array("distance")
        # distance_array = distance_coords[channels]

        # subsample the spool to select the channels and time range
        # self.spool = self.spool.select(
        #     distance=(distance_array), samples=False
        # )

        # A more general solution. Yet to be tested

        # patch_list = []
        # for patch in self.spool:
        #     patch_list.append(patch.select(distance=(distance_array)))

        # self.spool = dc.spool(patch_list)

        # self.spool = self.spool.select(time=self.time_range, samples=True)
        # self.spool = self.spool.select(time=self.time_range)

        self.contents = self.spool.get_contents()
        self.time_step = self.contents["time_step"][0].total_seconds()

    def single_patch_coherence(self, patch):
        """Compute coherence for a single patch of data."""
        return coherence(
            patch.select(**{self.channel_dim: self.distance_array}).data.T,
            self.sub_window_length,
            self.overlap,
            sample_interval=self.time_step,
            method=self.method,
        )

    def run(self, client=None):
        """Implement the coherence analysis using initialized parameters."""
        # perform coherence calculation on each patch
        if client is not None:
            map_out = client.map(self.single_patch_coherence, self.spool)
        else:
            map_out = self.spool.map(
                lambda x: coherence(
                    x.select(**{self.channel_dim: self.distance_array}).data.T,
                    self.sub_window_length,
                    self.overlap,
                    sample_interval=self.time_step,
                    method=self.method,
                )
            )
        self.detection_significance = np.stack(
            [a[0] for a in map_out if a is not None], axis=-1
        )
        self.eig_estimates = np.stack(
            [a[1] for a in map_out if a is not None], axis=-1
        )

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

        # print(self.contents[["time_min", "time_max"]])
        # print(
        #     self.contents["time_min"][0],
        #     self.contents["time_max"][self.contents.index[-1]],
        # )

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

        # Create the result directory if it does not exist
        Path(self.save_location).mkdir(parents=True, exist_ok=True)
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
            f"{self.method}_eig_estimatess_{start_time}_{end_time}.pkl",
        )
        with open(savename, "wb") as f:
            pickle.dump(self.eig_estimates, f)

        savename = os.path.join(
            self.save_location,
            f"{self.method}_metadata_{start_time}_{end_time}.pkl",
        )
        with open(savename, "wb") as f:
            pickle.dump(metadata, f)


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
        "-p",
        "--parallel",
        help="Whether to parallelize the computation",
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # record start time
    start_time = datetime.now()

    # Parse arguments
    args = parse_args()
    print(f"Arguments read from command line: {args}", flush=True)

    if args.parallel:
        available_threads = os.environ.get("SLURM_CPUS_PER_TASK")
        if available_threads is None:
            available_threads = os.cpu_count()

        print(f"Available threads: {available_threads}", flush=True)
        max_threads = max(int(int(available_threads) // 2), 1)
        print(
            f"Using {max_threads} threads for parallel processing.", flush=True
        )
        client = ProcessPoolExecutor(max_workers=max_threads)
    else:
        client = None

    # Initialize the coherence_analysis instance
    coherence_instance = CoherenceAnalysis(vars(args))

    # Read the data
    print("Reading data...", flush=True)
    coherence_instance.read_data()

    end_time = datetime.now()
    print(f"Data read in: {end_time - start_time}", flush=True)

    # run the coherence analysis
    print("Running coherence analysis...", flush=True)
    coherence_instance.run(client=client)

    # save the results
    print(
        f"""Finished in: {datetime.now() - start_time} for"
         {coherence_instance.method} method. Saving results...""",
        flush=True,
    )
    coherence_instance.save_results()

    end_time = datetime.now()
    print(f"Total duration: {end_time - start_time}", flush=True)
