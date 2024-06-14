r"""
Computes the detection significance through coherence of a single file using various methods for coherence analysis
This was written with the intention of comparing the performance of various ways of doing coherence analysis
The results are saved to a file for later analysis
The data intended for use here is from Brady Hotspring and is in hdf5 format.

Can be run from the command line as follows:
# python single_file_coerence.py <file> <averaging_window_length> <sub_window_length> <overlap> <first_channel> <num_channels> <samples_per_sec> <channel_offset> <method>
# Example:
# python single_file_coerence.py "D:\CSM\Mines_Research\Test_data\Brady Hotspring\PoroTomo_iDAS16043_160312000048.h5" 0 2 0 3100 200 1000 2000 exact
"""
import pickle
import sys
from datetime import datetime
from pathlib import Path

import functions as func

METHODS = ["exact", "qr", "svd", "rsvd", "power", "qr iteration"]
save_location = Path(
    "D:/CSM/Mines_Research/Repositories/Coherence_Analyses/test_results"
)

start_time = datetime.now()
if __name__ == "__main__":
    file = sys.argv[1]
    averaging_window_length = int(sys.argv[2])  # Averaging window length in seconds
    sub_window_length = int(sys.argv[3])  # sub-window length in seconds
    overlap = int(sys.argv[4])  # overlap in seconds
    first_channel = int(sys.argv[5])  # first channel
    num_channels = int(sys.argv[6])  # number of sensors
    samples_per_sec = int(sys.argv[7])  # samples per second
    channel_offset = int(sys.argv[8])  # channel offset
    method = sys.argv[9]  # method to use for coherence analysis

    file = Path(
        file
    )  # r"D:\CSM\Mines_Research\Test_data\Brady Hotspring\PoroTomo_iDAS16043_160312000048.h5"
    data, _ = func.loadBradyHShdf5(file, normalize="no")

    if True:  # method == "qr":
        detection_significance, eig_estimates = func.coherence(
            data[
                first_channel : channel_offset
                + first_channel : int(channel_offset / num_channels)
            ],
            sub_window_length,
            overlap,
            sample_interval=1 / samples_per_sec,
            method=method,
        )
        save_data = {
            "detection_significance": detection_significance,
            "eig_estimates": eig_estimates,
        }
    elif method in METHODS:
        detection_significance = func.coherence(
            data[
                first_channel : channel_offset
                + first_channel : int(channel_offset / num_channels)
            ],
            sub_window_length,
            overlap,
            sample_interval=1 / samples_per_sec,
            method=method,
        )
        save_data = {"detection_significance": detection_significance}
    # print(detection_significance.shape, flush=True)
    print(
        f"Finished in: {datetime.now()-start_time} for {method} method. Saving to file...",
        flush=True,
    )
    savename = (
        save_location / f"{method}_detection_significance_{str(file)[-15:-3]}.pkl"
    )
    with open(savename, "wb") as f:
        pickle.dump(detection_significance, f)
    savename = save_location / f"{method}_eig_estimates_{str(file)[-15:-3]}.pkl"
    with open(savename, "wb") as f:
        pickle.dump(eig_estimates, f)
