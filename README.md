
# Coherence_Analyses

## Overview

This repository contains research code for performing coherence analyses on Distributed Acoustic Sensing (DAS) data.

## Features

- Coherence analysis using exact computation and multiple methods of approximation: QR, SVD, randomized SVD
- Data reading and preprocessing with [dascore](https://dascore.org/)
- Batch processing and result saving for large datasets
- Example scripts and Jupyter notebooks for exploration and visualization

## Directory Structure

```bash
coherence_analyses/
│
├── coherence_analysis/
│   ├── extras/                  # Additional scripts and resources
│   ├── coherence_analysis.py     # Main analysis script
│   ├── single_file_coherence.py  # Single file analysis
│   └── utils/                   # Utility functions
├── data/                        # This an untracked directory for data
│   ├── images/                  # Figures and plots
│   └── results/                 # Output results
├── notebooks/                   # Jupyter notebooks for exploration
├── scripts/                     # SLURM and batch scripts
├── tests.py                     # Unit and integration tests
├── requirements.txt             # Python dependencies
└── pyproject.toml               # Project metadata and optional dependencies
└── README.md                    # Project documentation
```

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/aissah/coherence-analysis.git
    cd coherence_analyses
    ```

2. Install dependencies (Python >=3.11 required): All the details about the project and its dependencies are in `pyproject.toml`. This contains details which dependencies are required for the core functionality, as well as optional dependencies needed to run the notebooks. You can install the core dependencies using:

   ```sh
   pip install -r requirements_basic.txt
   ```

requirements_notebooks.txt contains additional dependencies for running the notebooks, and requirements.txt includes all optional dependencies except development dependencies.

## Usage

This repo can be used through command line scripts or importing the utility function in coherence_analysis/utils/.

### Command Line

The command line interface takes care of I/O prodived the data can be read by dascore. Run coherence analysis on a directory of DAS data:

```sh
python coherence_analysis/coherence_analysis.py <method> <data_path> <averaging_window_length> <sub_window_length> [-o <overlap>] [-t <time_range>] [-ch <channel_range>] [-ds <channel_offset>] [-dt <time_step>] [-r <result_path>] [-p <parallel>]
```

Example:

```sh
python coherence_analysis/coherence_analysis.py exact "data/Port_Angeles" 60 5 -o 0 -t "('06/01/23 07:32:09', '06/01/23 07:42:09')" -ch "(0, 10)" -ds 1 -dt 0.002 -r "data/results"
```

For a directory containing large amounts of data, the current implementation runs into memory issues. Hence, there is a more premitive command line option that uses a custom reader. This can be ran as follows:

```sh
python coherence_analysis/coherence_analysis_no_dascore.py  <method> <data_path> <averaging_window_length> <sub_window_length> [-o <overlap>] [-ch <channel_range>] [-ds <channel_offset>] [-dt <time_step>] [-r <result_path>] [-b <batch>] [-bs <batch_size>]
```

More details of the arguments can be found in the docstring of the script. One notable difference is that this script does not have an option for time range selection. This option is nicely handled by dascore but it turns out is not that straightforward to implement in a custom reader.

The output results are saved in the specified result path as pickle files for later analysis. The default result_path is "../data/results" relative to the script location. There are three files saved for each run:

- Detection parameters: np array of shape
    (num_averaging_windows, num_frequencies)
- Eigenvalues of the coherence matrices: np array of shape
    (num_frequencies, min(num_channels, num_subwindows) * num_averaging_windows)
- metadata: dictionary containing the parameters used for coherence analysis
such as sampling rate, averaging window length, sub-window length, overlap,
channel range, channel offset, method, list of files used, window start and
end times, and ignored files.

The files are named as:

- `{method}_detection_significance_{first_file_date}_{last_file_date}.pkl`
- `{method}_eig_estimatess_{first_file_date}_{last_file_date}.pkl`
- `{method}_metadata_{first_file_date}_{last_file_date}.pkl`

Where {method} is the method used for coherence analysis, {first_file_date} is
the first file date in the batch, and {last_file_date} is the last file date
in the batch.

The files in coherence_analysis/extras/ can also be run from the command line for specific use cases.

### Jupyter Notebooks

Explore and visualize coherence analysis results using the notebooks in the `notebooks/` directory. This contains notebooks that cover various research directions, including:

- Exploring coherence matrices and computation
- Estimating coherence matrix eigenvalues
- Effects of noise coherence matrix analysis
- Impact of event frequency on coherence matrix analysis
- Experiments with model data

## Testing

Test suite is not fully implemented yet.

## License

This project is for research purposes. Licensing details to be determined.
