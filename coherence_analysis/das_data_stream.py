"""DAS data loader module."""

import os
from ast import literal_eval
from datetime import datetime, timedelta

import compression_rate_prediction.utils as utils
import numpy as np


class DasDataStream:
    """Class to load DAS data using various loader functions."""

    def __init__(self, filepath, loader_id: str = None, args: dict = None):
        """Initialize DasDataStream with file path, loader, and parameters."""
        self.available_loaders = ["dascore", "stanford_segy", "brady_hdf5"]
        self.data_path = filepath
        self.loader_id = loader_id
        self.loader_function = None
        self.sample_rate = None
        self.in_memory_data = None
        self.current_file_index = 0
        self.in_memory_data_start_time = None
        self.ignored_files = []

        # load parameters in args
        self.parse_attributes(args)
        print(f"""Initialized with the following parameters:
            data_path: {self.data_path}
            time_range: {self.time_range}
            channel_range: {self.channel_range}
            channel_step: {self.channel_step}
            window_length: {self.window_length}
            overlap: {self.overlap}
            time_step: {self.time_step}
            """)

        # Set default loader if none specified
        if self.loader_id is None or self.loader_id == "dascore":
            print("No loader specified, defaulting to use dascore.")
            try:
                import dascore as dc

                # self.loader_function = dc.spool
                self.loader_id = "dascore"
                self.spool = dc.spool(self.data_path)
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "dascore is not installed. Please install dascore or"
                    " provide a custom loader. Available loaders are: "
                    f"{self.available_loaders}"
                )
        else:
            self._custom_loader_setup()

    def reset(self):
        """Reset the data loader to the beginning of the data files."""
        self.current_file_index = 0
        self.in_memory_data = None
        self.in_memory_data_start_time = None

    def parse_attributes(self, args: dict):
        """Set or reset the attributes of the data loader.

        Parameters
        ----------
        args : dict
            Dictionary of arguments to set for the data loader
        """
        if args is not None:
            # Convert time_range and channel_range from strings to lists
            args["time_range"] = (
                literal_eval(args["time_range"])
                if isinstance(args["time_range"], str)
                else args["time_range"]
            )

            self.time_range = [
                datetime.strptime(a, "%m/%d/%y %H:%M:%S") if a != ... else ...
                for a in args["time_range"]
            ]

            self.channel_range = (
                literal_eval(args["channel_range"])
                if isinstance(args["channel_range"], str)
                else args["channel_range"]
            )

            self.channel_range = args["channel_range"]

        if args is None:
            args = {}
            print("No arguments provided, using default values.")

        if self.loader_id is None:
            self.loader_id = args.get("loader_id", self.loader_id)

        self.channel_range = args.get("channel_range", (..., ...))
        self.time_range = args.get("time_range", [..., ...])
        self.data_path = args.get("data_path", self.data_path)
        self.window_length = args.get("window_length", None)
        self.channel_step = args.get("channel_step", 1)
        self.overlap = args.get("overlap", 0)
        self.time_step = args.get("time_step", None)

    def file_extension(self):
        """Get the file extension based on the loader function."""
        if self.loader_id == "stanford_segy":
            self.loader_function = self.load_stanford_segy
            return ".sgy"
        elif self.loader_id == "brady_hdf5":
            self.loader_function = self.load_brady_hdf5
            return ".h5"

    def populate_data_files(self):
        """Get list of data files in the specified directory.

        This is only used if for a custom loader function.

        Parameters
        ----------
        data_path : str
            Path to the directory containing the data files

        Returns
        -------
        file_list : list of str
            List of data file paths
        """
        file_extension = self.file_extension()

        if self.data_path.endswith(file_extension):
            self.file_list = [self.data_path]
        else:
            file_list = []
            for dir_path, dir_names, file_names in os.walk(self.data_path):
                dir_names.sort()
                file_names.sort()
                file_list.extend(
                    [
                        os.path.join(dir_path, file_name)
                        for file_name in file_names
                        if file_name.endswith(file_extension)
                        and file_name[0] != "."
                    ]
                )
            self.file_list = file_list
        if len(self.file_list) == 0:
            raise FileNotFoundError(
                f"No files with extension '{file_extension}' found in "
                f"directory '{self.data_path}'."
            )
        if self.time_range == [..., ...]:
            self.num_files = len(self.file_list)
        else:
            start_times = [self.get_file_start_time(f) for f in self.file_list]
            if self.time_range[0] is not ...:
                self.file_list = [
                    f
                    for f, t in zip(self.file_list, start_times)
                    if t >= self.time_range[0]
                ]
            if self.time_range[1] is not ...:
                self.file_list = [
                    f
                    for f, t in zip(self.file_list, start_times)
                    if t < self.time_range[1]
                ]
            self.num_files = len(self.file_list)
        # return file_list

    # Functions to handle custom loaders
    def _custom_loader_setup(self):
        """Set up the custom loader function based on the loader name."""
        self.channel_range = list(self.channel_range)
        if self.channel_range[0] is Ellipsis:
            self.channel_range[0] = 0
        if self.channel_range[1] is Ellipsis:
            self.channel_range[1] = -1
        if self.loader_id in self.available_loaders:
            self.populate_data_files()
        else:
            raise ValueError(
                f"Loader '{self.loader_id}' not recognized. Available loaders"
                f" are: {self.available_loaders}."
            )

    # def stanford_segy_data(self, filename):
    #     """Import SEGY data from a file."""
    #     from obspy.io.segy.segy import _read_segy

    #     data_stream = _read_segy(filename)
    #     num_of_traces = len(data_stream.traces)
    #     data = np.asarray(
    #         [data_stream.traces[i].data for i in range(num_of_traces)]
    #     )
    #     nchannels, nsamples = data.shape
    #     metadata = data_stream.traces[0].to_obspy_trace().stats
    #     dt = metadata["delta"]
    #     starttime = metadata.starttime.datetime

    #     if self.sample_rate is None:
    #         self.sample_rate = 1 / dt
    #     elif self.sample_rate != 1 / dt:
    #         raise ValueError(
    #             f"Sample rate has changed: {self.sample_rate} != {1 / dt}"
    #         )

    #     return data, nchannels, nsamples, dt, starttime

    def load_stanford_segy(self, filename):
        """Import SEGY data from a file."""
        data, nchannels, nsamples, dt, starttime = utils.import_segy_data(
            filename
        )
        if self.sample_rate is None:
            self.sample_rate = 1 / dt
        elif self.sample_rate != 1 / dt:
            raise ValueError(
                f"Sample rate has changed: {self.sample_rate} != {1 / dt}"
            )

        self.in_memory_data_start_time = starttime

        if self.window_length is None:
            self.window_length = nsamples / self.sample_rate
        self.total_window_length = int(self.window_length * self.sample_rate)

        return data

    def load_brady_hdf5(self, file: str):
        """Load brady hotspring h5py data file."""
        data, timestamp_arr = utils.load_brady_hdf5(file, normalize="no")
        dt = timestamp_arr[1] - timestamp_arr[0]

        if self.sample_rate is None:
            self.sample_rate = 1 / dt
        elif self.sample_rate != 1 / dt:
            raise ValueError(
                f"Sample rate has changed: {self.sample_rate} != {1 / dt}"
            )

        self.in_memory_data_start_time = datetime.fromtimestamp(
            timestamp_arr[0], datetime.timezone.utc
        )

        if self.window_length is None:
            self.window_length = data.shape[1] / self.sample_rate

        self.total_window_length = int(self.window_length * self.sample_rate)

        return data

    def get_file_start_time(self, file_path: str):
        """Get the start time of a data file from its filename.

        Parameters
        ----------
        file_path : str
            Path to the data file

        Returns
        -------
        start_time : datetime
            Start time of the data file
        """
        if self.loader_id == "stanford_segy":
            # import obspy
            # segy_file = obspy.io.segy.segy._read_segy(file_path)
            # start_time = (
            #     segy_file.traces[0].to_obspy_trace().stats.starttime.datetime
            # )
            file_name = os.path.basename(file_path)
            start_time = datetime.strptime(
                file_name[-26:-20] + file_name[-19:-13], "%y%m%d%H%M%S"
            )
        elif self.loader_id == "brady_hdf5":
            file_name = os.path.basename(file_path)
            start_time = datetime.strptime(file_name[-15:-3], "%y%m%d%H%M%S")

        return start_time

    def _partition_data_samples(
        self, data: np.array = None, window_deficit: int = None
    ):
        """Partition data into data and self.in_memory_data.

        Parameters
        ----------
        data : np array
            Input data to partition
        window_deficit : int, optional
            Number of samples to load into data variable, by default None

        Returns
        -------
        data : np array
            Partitioned data
        """
        if window_deficit is None:
            window_deficit = self.total_window_length

        if data is None:
            data = self.in_memory_data

        data, self.in_memory_data = (
            data[:, :window_deficit],
            data[:, window_deficit:],
        )
        if self.in_memory_data.size == 0:
            self.in_memory_data = None

        if self.in_memory_data is not None:
            self.in_memory_data_start_time += timedelta(
                seconds=int(window_deficit / self.sample_rate)
            )
        else:
            self.current_file_index += 1
            self.in_memory_data_start_time = None

        return data

    def _load_and_subselect_data(self, window_deficit: int = None):
        """Load data from a file and subselect channels.

        Parameters
        ----------
        window_deficit : int, optional
            Number of samples to load into data variable, by default None

        Returns
        -------
        data : np array
            Loaded and subselected data
        """
        data = self.loader_function(self.file_list[self.current_file_index])
        data = data[
            self.channel_range[0] : +self.channel_range[1] : self.channel_step,
            :,
        ]
        data = self._partition_data_samples(data, window_deficit)
        # data, self.in_memory_data = (
        #     data[:, : window_deficit],
        #     data[:, window_deficit :],
        # )

        # if len(self.in_memory_data) == 0:
        #         self.in_memory_data = None

        # if self.in_memory_data is not None:
        #     self.in_memory_data_start_time += timedelta(
        #         seconds=int(window_deficit / self.sample_rate)
        #     )
        # else:
        #     self.current_file_index += 1
        #     self.in_memory_data_start_time = None

        return data

    def _next_data_window(self):
        """
        Load the next data window from the data files.

        This function is used to load the next window of data from the list of
        data files. It continues to read data from the files until the window
        length is reached.

        Returns
        -------
        data : np array
            data read from the data files
        window_start_time : datetime
            Start time of the data window
        window_end_time : datetime
            End time of the data window

        """
        # num_files = len(data_files)
        # total_window_length = window_length * samples_per_sec

        if self.in_memory_data is None:
            if self.current_file_index >= self.num_files:
                return None
            # window_start_time = self.get_file_start_time(
            #     self.file_list[self.current_file_index]
            # )
            # window_start_time += timedelta(
            #     seconds=start_sample_index / samples_per_sec
            # )
            data = self._load_and_subselect_data()
            window_start_time = self.in_memory_data_start_time
            # data = self.loader_function(
            #     self.file_list[self.current_file_index]
            # )
            # # data = func.rm_laser_drift(data)
            # data = data[
            #     self.channel_range[0] : self.channel_step
            #     + self.channel_range[0] : int(
            #         self.channel_step / self.num_channels
            #     ),
            #     :,
            # ]
            # data, self.in_memory_data = (
            #         data[:, : self.total_window_length],
            #         data[:, self.total_window_length :],
            #     )
        else:
            window_start_time = self.in_memory_data_start_time
            # data, self.in_memory_data = (
            #     self.in_memory_data[:, : self.total_window_length],
            #     self.in_memory_data[:, self.total_window_length :],
            # )

            data = self._partition_data_samples()

        # if len(self.in_memory_data) == 0:
        #         self.in_memory_data = None

        # data_len = data.shape[1]
        # number of samples to add to the data to make up the window length
        window_deficit = self.total_window_length - data.shape[1]

        # if self.in_memory_data is not None:
        #     self.in_memory_data_start_time += timedelta(
        #         seconds=int(data.shape[1] / samples_per_sec)
        #     )
        # else:
        #     self.current_file_index += 1
        #     self.in_memory_data_start_time = None

        # ignored_files = []

        while window_deficit > 0:
            if self.current_file_index >= self.num_files:
                return None
            # self.current_file_index += (
            #     1  # index of the next file to read data from
            # )
            self.in_memory_data_start_time = self.get_file_start_time(
                self.file_list[self.current_file_index]
            )

            if self.in_memory_data_start_time - window_start_time > timedelta(
                seconds=int(data.shape[1] / self.sample_rate) + 1
            ):
                self.ignored_files.append(
                    self.file_list[self.current_file_index - 1]
                )

                window_start_time = self.in_memory_data_start_time
                data = self._load_and_subselect_data()
                # data = self.loader_function(
                #     self.file_list[self.current_file_index],
                # )
                # # data = func.rm_laser_drift(data)
                # data = data[
                #     self.channel_range[0] : self.channel_step
                #     + self.channel_range[0] : int(
                #         self.channel_step / self.num_channels
                #     ),
                #     :,
                # ]
                # data, self.in_memory_data = (
                #     (
                #         data[:, : self.total_window_length],
                #         data[:, self.total_window_length :],
                #     )
                #     if self.total_window_length < data_len
                #     else (data, None)
                # )

                # if self.in_memory_data is not None:
                #     self.in_memory_data_start_time += timedelta(
                #         seconds=int(data.shape[1] / samples_per_sec)
                #     )
                # else:
                #     self.current_file_index += 1
                #     self.in_memory_data_start_time = None

                window_deficit = self.total_window_length - data.shape[1]
            else:
                next_data = self._load_and_subselect_data(window_deficit)
                # next_data = self.loader_function(
                #     self.file_list[self.current_file_index],
                # )
                # # next_data = func.rm_laser_drift(next_data)
                # next_data = next_data[
                #     self.channel_range[0] : self.channel_step
                #     + self.channel_range[0] : int(
                #         self.channel_step / self.num_channels
                #     )
                # ]
                # next_data, self.in_memory_data = (
                #     data[:, :window_deficit],
                #     data[:, window_deficit:],
                # )
                data = np.append(data, next_data, axis=1)
                next_data = None

                # if self.in_memory_data is not None:
                #     self.in_memory_data_start_time += timedelta(
                #         seconds=int(window_deficit / samples_per_sec)
                #     )
                # else:
                #     self.current_file_index += 1
                #     self.in_memory_data_start_time = None

                window_deficit = self.total_window_length - data.shape[1]

        window_end_time = window_start_time + timedelta(
            seconds=self.total_window_length / self.sample_rate
        )

        return (
            data,
            window_start_time,
            window_end_time,
        )

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
            self.channel_step,
            dtype=int,
        )
        distance_coords = first_patch.coords.get_array(self.channel_dim)
        self.distance_array = distance_coords[channels]

    def read_data(self):
        """Read the files and subselect using dascore."""
        # read the data files using the spool function from dascore
        self.spool = self.loader(self.data_path)
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
            time=self.window_length, keep_partial=True
        )

        self.spool = self.spool.select(time=self.time_range, samples=True)

        # set the channel dimension
        self._set_channel_dim()

        self.contents = self.spool.get_contents()
        self.time_step = self.contents["time_step"][0].total_seconds()

    def run(self, client=None):
        """Implement the coherence analysis using initialized parameters."""
        # perform coherence calculation on each patch
        # if client is not None:
        #     map_out = client.map(self.single_patch_coherence, self.spool)
        # else:
        #     map_out = self.spool.map(
        #         lambda x: coherence(
        #          x.select(**{self.channel_dim: self.distance_array}).data.T,
        #             self.sub_window_length,
        #             self.overlap,
        #             sample_interval=self.time_step,
        #             method=self.method,
        #         )
        #     )
        # self.detection_significance = np.stack(
        #     [a[0] for a in map_out if a is not None], axis=-1
        # )
        # self.eig_estimates = np.stack(
        #     [a[1] for a in map_out if a is not None], axis=-1
        # )
        pass

    def __next__(self):
        """Get the next data window."""
        if self.loader_id == "dascore":
            raise NotImplementedError(
                "Dascore loader does not support __next__ method."
            )
        else:
            next_ = self._next_data_window()
            if next_ is None:
                raise StopIteration
            return next_

    def __str__(self):
        """Return a string representation of the DasDataStream object."""
        return (
            f"DasDataStream(loader_id={self.loader_id}, "
            f"data_path={self.data_path}, "
            f"time_range={self.time_range}, "
            f"channel_range={self.channel_range}, "
            f"channel_step={self.channel_step}, "
            f"window_length={self.window_length}, "
            f"overlap={self.overlap}, "
            f"time_step={self.time_step})"
        )

    def __iter__(self):
        """Return the iterator object itself."""
        return self
