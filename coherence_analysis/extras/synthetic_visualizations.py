"""Make noise test simulations for synthetic experiments."""

import argparse
import os
import sys
from ast import literal_eval
from datetime import datetime

import torch

sys.path.append(os.path.join(os.path.dirname(""), os.pardir))
import coherence_analysis.utils.synthetic_tests_utils as f


def parse_args():
    """Parse command line arguments.

    Raises
    ------
    ValueError
        Raise error if the method selected is not available.
    """
    # Initialize the parser
    parser = argparse.ArgumentParser(
        description="Run noise test on pre-computed synthetic data"
    )

    # Add arguments
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the synthetic data file",
    )
    parser.add_argument(
        "event_freq_range",
        type=str,
        help="str of 1 float or tuple of two comma-separated frequencies"
        " defining the event frequency range, e.g., '(5,15)' or '10'",
    )
    parser.add_argument(
        "signal_to_noise_list",
        type=str,
        help="List of signal to noise ratios as a string",
        default="[2, 1.75, 1.5, 1.25, 1, 0.75, 0.5, 0.25]",
    )
    parser.add_argument(
        "cov_len_list",
        type=str,
        help="List of covariance lengths as a string",
        default="[10, 50, 100, 200]",
    )
    parser.add_argument(
        "sub_window_length",
        type=int,
        help="Sub-window length in seconds",
        default=1,
    )
    parser.add_argument(
        "-o", "--overlap", type=int, help="Overlap in seconds", default=0
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
        "-s",
        "--nsims",
        type=int,
        help="Number of simulations to run",
        default=50,
    )
    return parser.parse_args()


if __name__ == "__main__":
    # record start time
    start_time = datetime.now()

    # Parse arguments
    args = parse_args()
    print(f"Arguments read from command line: {args}", flush=True)

    print("Reading data...", flush=True)

    # save_directory = os.path.join(
    # os.path.dirname(""), os.pardir, os.pardir, "data", "simulated_data"
    # )

    if os.path.exists(args.file_path):
        receiver_amplitudes = torch.load(args.file_path)
    else:
        raise FileNotFoundError(f"File not found: {args.file_path}")

    end_time = datetime.now()
    print(f"Data read in: {end_time - start_time}", flush=True)

    # Running noise test
    print("running noise test...", flush=True)
    coherence_data = receiver_amplitudes[0].cpu().numpy()
    df = f.noise_test(
        receiver_amplitudes[0].cpu().numpy(),
        win_len=args.sub_window_length,
        overlap=args.overlap,
        sample_interval=args.time_step,
        signal_to_noise_list=literal_eval(args.signal_to_noise_list),
        cov_len_list=literal_eval(args.cov_len_list),
        event_freq_range=literal_eval(args.event_freq_range),
        num_of_sims=args.nsims,
    )

    # save the results
    print(
        f"Finished in: {datetime.now() - start_time}. Saving results...",
        flush=True,
    )
    df.to_csv(os.path.join(args.result_path, "noise_test_results.csv"))

    end_time = datetime.now()
    print(f"Total duration: {end_time - start_time}", flush=True)
