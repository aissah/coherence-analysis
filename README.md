
# Coherence_Analyses

## Overview

This repository contains research code for performing coherence analyses on Distributed Acoustic Sensing (DAS) data. It provides tools for numerical tests of matrix coherence, including methods for analyzing large-scale DAS datasets using various coherence estimation techniques.

## Features

- Coherence analysis using multiple methods: exact, QR, SVD, randomized SVD, and more
- Data reading and preprocessing with [dascore](https://dascore.org/)
- Batch processing and result saving for large datasets
- Example scripts and Jupyter notebooks for exploration and visualization
- Modular design for extension and experimentation

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/aissah/Coherence_Analyses.git
   cd Coherence_Analyses
   ```
2. Install dependencies (Python >=3.11 required):
   ```sh
   pip install -r requirements.txt
   ```
   Or use the optional dependencies in `pyproject.toml` for notebooks and modeling:
   ```sh
   pip install .[notebooks,modeling,rand-svd]
   ```

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

Explore and visualize coherence analysis results using the notebooks in the `notebooks/` directory. Example topics include:

- Exploring coherence
- Estimating coherence matrix eigenvalues
- Effects of noise and event frequency
- Model data analysis

## Directory Structure

```
coherence_analysis/
	coherence_analysis.py      # Main analysis script
	single_file_coherence.py  # Single file analysis
	utils.py                  # Utility functions
data/
	images/                   # Figures and plots
	results/                  # Output results
notebooks/                   # Jupyter notebooks for exploration
scripts/                     # SLURM and batch scripts
tests.py                     # Unit and integration tests
requirements.txt             # Python dependencies
pyproject.toml               # Project metadata and optional dependencies
README.md                    # Project documentation
```

## Testing

Run the test suite with:

```sh
pytest tests.py
```

## Contributing

Contributions, suggestions, and issues are welcome! Please open an issue or submit a pull request.

## License

This project is for research purposes. Licensing details to be determined.
