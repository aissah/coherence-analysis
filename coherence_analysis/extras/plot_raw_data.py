"""Script to plot raw data.

This script reads raw data from a specified directory, selects a time range and
channel range, and generates a plot of the raw data. The plot is saved to a
designated results directory.

Usage:
    python plot_raw_data.py <data_path> <time_range> [-ch <channel_range>]
                            [-ds <channel_offset>] [-r <result_path>]
Arguments:
    data_path (str): Path to the directory containing the data files.
    time_range (str): Range of time (in Python list format,
                    e.g., "[start_time, end_time]").
    -ch, --channel_range (str, optional): Range of channels to plot.
                    (in Python list format, e.g., "[start_channel,
                    end_channel]"). Default is "(0, ...)".
    -ds, --channel_offset (int, optional): Channels to skip in between.
                    Default is 1.
    -r, --result_path (str, optional): Directory to save results. Default is
                    "../data/results".
Example:
    python plot_raw_data.py ./data "[2023-01-01 00:00:00, 2023-01-02 00:00:00]"
    -ch "[0, 100]" -ds 2 -r ./results
"""

import argparse
import os
from ast import literal_eval
from datetime import datetime

import dascore as dc
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def parse_args():
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
        "data_path",
        type=str,
        help="Path to the directory containing the data files",
    )
    parser.add_argument(
        "time_range",
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
        "-r",
        "--result_path",
        type=str,
        help="Directory to save results",
        default=os.path.join(
            os.path.dirname(__file__), os.pardir, "data/results"
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    # record start time
    start_time = datetime.now()

    # Parse arguments
    args = parse_args()
    print(f"Arguments read from command line: {args}", flush=True)

    time_range = [
        datetime.strptime(a, "%m/%d/%y %H:%M:%S") if a != ... else ...
        for a in literal_eval(args.time_range)
    ]
    channel_range = literal_eval(args.channel_range)

    # Read the data
    print("Reading data...", flush=True)
    spool = dc.spool(args.data_path)
    spool = spool.select(time=time_range, samples=True)
    # spool = spool.concatenate(time=None)

    dims = spool[0].dims
    print(f"The data has the following dimensions: {dims}")
    print(
        f"""Channels will be grouped based on the '{dims[1]}'
            dimension. If another dimension is desired, use the
            method, '_set_channel_dim()' to set it.""",
        flush=True,
    )
    channel_dim = dims[1]

    channels = np.arange(
        channel_range[0],
        channel_range[1],
        args.channel_offset,
        dtype=int,
    )
    distance_coords = spool[0].coords.get_array(channel_dim)
    distance_array = distance_coords[channels]

    data_array = np.concatenate(
        [d.select(**{channel_dim: distance_array}).data for d in spool], axis=0
    )

    print("Data read successfully.", flush=True)

    dpi = 600
    label_size = 16
    tick_size = 14
    legend_size = 12
    sns.set_theme(style="ticks", context="paper", font_scale=2)
    # plt.figure(figsize=(10, 6), dpi=dpi)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    ax.imshow(
        data_array.T,
        aspect="auto",
        extent=[
            time_range[0],
            time_range[1],
            distance_array[-1],
            distance_array[0],
        ],
        cmap="RdBu_r",
        vmin=-0.005,
        vmax=0.005,
    )
    cbar = fig.colorbar(ax.images[0], ax=ax, pad=0.02)
    cbar.set_label("Strain Rate")
    # cbar.set_label("Strain Rate", size=legend_size, weight="bold")
    # cbar.ax.tick_params(labelsize=legend_size)
    # ax.xaxis_date()

    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_formatter(formatter)

    # fig.autofmt_xdate()
    # ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%H:%M:%S"))
    plt.xlabel("Time")
    plt.ylabel("Channels")
    # plt.xticks(fontsize=tick_size)
    # plt.yticks(fontsize=tick_size)
    fig.tight_layout()

    print("Saving raw data plot...", flush=True)
    os.makedirs(args.result_path, exist_ok=True)
    plt.savefig(
        os.path.join(args.result_path, f"raw_data_plot_{time_range[0]}.png")
    )
