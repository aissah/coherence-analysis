
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
│   ├── coherence_analysis.py      # Main analysis script
│   ├── single_file_coherence.py  # Single file analysis
│   └── utils.py                   # Utility functions
├── data/
│   ├── images/                   # Figures and plots
│   └── results/                  # Output results
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
    git clone https://github.com/aissah/Coherence_Analyses.git
    cd Coherence_Analyses
    ```

2. Install dependencies (Python >=3.11 required): All the details about the project and its dependencies are in `pyproject.toml`. This contains details which dependencies are required for the core functionality, as well as optional dependencies needed to run the notebooks. You can install the core dependencies using:

   ```sh
   pip install -r requirements_basic.txt
   ```

requirements_notebooks.txt contains additional dependencies for running the notebooks, and requirements.txt includes all optional dependencies except development dependencies.

## Usage

### Command Line

Run coherence analysis on a directory of DAS data:

```sh
python coherence_analysis/coherence_analysis.py <method> <data_path> <averaging_window_length> <sub_window_length> [-o <overlap>] [-t <time_range>] [-ch <channel_range>] [-ds <channel_offset>] [-dt <time_step>] [-r <result_path>]
```

Example:

```sh
python coherence_analysis/coherence_analysis.py exact "data/Port_Angeles" 60 5 -o 0 -t "('06/01/23 07:32:09', '06/01/23 07:42:09')" -ch "(0, 10)" -ds 1 -dt 0.002 -r "data/results"
```

### Jupyter Notebooks

Explore and visualize coherence analysis results using the notebooks in the `notebooks/` directory. This contains notebooks that cover various research directions, including:

- Exploring coherence matrices and computation
- Estimating coherence matrix eigenvalues
- Effects of noise coherence matrix analysis
- Impact of event frequency on coherence matrix analysis
- Experiments with model data

## Testing

Run the test suite with:

```sh
pytest tests.py
```

## Contributing

Contributions, suggestions, and issues are welcome! Please open an issue or submit a pull request.

## License

This project is for research purposes. Licensing details to be determined.
