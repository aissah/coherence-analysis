"""Supporting Functions for coherence analysis of DAS data."""

import h5py
import numpy as np
import scipy.signal as ss


def load_brady_hdf5(file: str, normalize: bool = "no") -> tuple:
    """
    Load brady hotspring h5py data file and return the data and timestamps.

    Parameters
    ----------
    file : str
        path to brady hotspring h5py data file
    normalize : str, optional
        "yes" or "no". Indicates whether or not to remove laser drift and
        normalize. The default is 'yes'.

    Returns
    -------
    data : np array
        channel by samples numpy array of data
    timestamp_arr : numpy array
        array of the timestamps corresponding to the various samples in the
        data. Timestamps for brady hotspring data are with respect to the
        beginning time of the survey.

    """
    with h5py.File(file, "r") as open_file:
        dataset = open_file["das"]
        time = open_file["t"]
        data = np.array(dataset)
        timestamp_arr = np.array(time)
    data = np.transpose(data)
    if normalize == "yes":
        number_of_samples = np.shape(data)[1]
        # get rid of laser drift
        med = np.median(data, axis=0)
        for i in range(number_of_samples):
            data[:, i] = data[:, i] - med[i]

        max_of_rows = abs(data[:, :]).sum(axis=1)
        data = data / max_of_rows[:, np.newaxis]
    return data, timestamp_arr


def windowed_spectra(
    data: np.array,
    subwindow_len: int,
    overlap: float,
    freq=None,
    sample_interval: int = 1,
) -> tuple:
    """
    Calculate the frequency domain representation of data in windows.

    Parameters
    ----------
    data : numpy array
        Data in time domain
    subwindow_len : int
        Length of the subwindows in seconds
    overlap : int
        Overlap between adjacent subwindows in seconds
    freq : int, optional
        Frequency to return the spectra at. The default is None.
        If None, the spectra is returned at all frequencies
    sample_interval : float, optional
        Sample interval of the data. The default is 1.

    Returns
    -------
    win_spectra : numpy array
        Spectra of the data in windows
    frequencies : numpy array
        Frequencies at which the spectra is computed

    """
    win_start = 0
    window_samples = int(subwindow_len / sample_interval)
    total_samples = data.shape[-1]
    overlap = int(overlap / sample_interval)
    intervals = np.arange(
        window_samples, total_samples + 1, window_samples, dtype=int
    )  # break time series into windowed intervals

    win_end = intervals[0]

    spectra = np.fft.rfft(data[:, win_start:win_end])
    win_spectra = spectra[np.newaxis]

    while win_end < total_samples:
        win_start = win_end - overlap
        win_end = win_start + window_samples
        spectra = np.fft.rfft(data[:, win_start:win_end])
        win_spectra = np.append(win_spectra, spectra[np.newaxis], axis=0)
        # win_start = win_end

    frequencies = np.fft.rfftfreq(window_samples, sample_interval)

    return win_spectra, frequencies


def normalised_windowed_spectra(
    data: np.array,
    subwindow_len: int,
    overlap: float,
    freq=None,
    sample_interval: int = 1,
) -> tuple:
    """
    Compute frequency domain representation of data nomralized in windows.

    Parameters
    ----------
    data : numpy array
        Data in time domain
    subwindow_len : int
        Length of the subwindows in seconds
    overlap : int
        Overlap between adjacent subwindows in seconds
    freq : int, optional
        Frequency to return the spectra at. The default is None.
        If None, the spectra is returned at all frequencies
    sample_interval : float, optional
        Sample interval of the data. The default is 1.

    Returns
    -------
    normalized_spectra : numpy array
        Normalized spectra of the data. The normalization is done
        by dividing the spectra by the sum of the absolute values of the
        spectra squared of each channel
    frequencies : numpy array
        Frequencies at which the spectra is computed

    """
    win_spectra, frequencies = windowed_spectra(
        data, subwindow_len, overlap, freq, sample_interval
    )

    # mean_spectra = np.mean(win_spectra, axis=0)
    # win_spectra -= mean_spectra

    normalizer = np.sum(np.absolute(win_spectra) ** 2, axis=0)
    normalizer = np.tile(np.sqrt(normalizer), (win_spectra.shape[0], 1, 1))
    normalizer = normalizer.transpose(2, 1, 0)

    normalized_spectra = win_spectra.transpose(2, 1, 0) / normalizer

    return normalized_spectra, frequencies


def welch_coherence(
    data: np.array,
    subwindow_len: int,
    overlap: float,
    freq=None,
    sample_interval: int = 1,
) -> tuple:
    """
    Calculate the coherence matrix at all frequencies.

    The welch method is used for spectra density calculation.

    Parameters
    ----------
    data : numpy array
        Data in time for coherence analysis
    subwindow_len : int
        Length of the subwindows in seconds
    overlap : int
        Overlap between adjacent subwindows in seconds
    freq : int, optional
        Frequency to compute the coherence at. The default is
        None. If None, the coherence is computed at all frequencies
    sample_interval : float, optional
        Sample interval of the data. The default is 1.

    Returns
    -------
    coherence : numpy array
        Coherence matrix of the data
    frequencies : numpy array
        Frequencies at which the coherence is computed

    """
    win_spectra, frequencies = windowed_spectra(
        data, subwindow_len, overlap, freq, sample_interval
    )

    # mean_spectra = np.mean(win_spectra, axis=0)
    # win_spectra -= mean_spectra

    normalizer = np.sum(np.absolute(win_spectra) ** 2, axis=0)
    normalizer = np.tile(normalizer, (normalizer.shape[0], 1, 1))
    normalizer = normalizer * normalizer.transpose((1, 0, 2))
    normalizer = normalizer.transpose(2, 1, 0)

    welch_numerator = np.matmul(
        win_spectra.transpose(2, 1, 0),
        np.conjugate(win_spectra.transpose(2, 0, 1)),
    )
    welch_numerator = np.absolute(welch_numerator) ** 2
    coherence = np.multiply(welch_numerator, 1 / normalizer)

    return coherence, frequencies


def covariance(
    data: np.array,
    subwindow_len: int,
    overlap: float,
    freq=None,
    sample_interval: int = 1,
) -> tuple:
    """
    Calculate the covariance matrix at all frequencies.

    Parameters
    ----------
    data : numpy array
        Data in time for covariance analysis
    subwindow_len : int
        Length of the subwindows in seconds
    overlap : int
        Overlap between adjacent subwindows in seconds
    freq : int, optional
        Frequency to compute the covariance at. The default is
        None. If None, the covariance is computed at all frequencies
    sample_interval : float, optional
        Sample interval of the data. The default is 1.

    Returns
    -------
    covariance : numpy array
        Covariance matrix of the data
    frequencies : numpy array
        Frequencies at which the coherence is computed

    """
    win_spectra, frequencies = windowed_spectra(
        data, subwindow_len, overlap, freq, sample_interval
    )

    covariance = np.matmul(
        win_spectra.transpose(2, 1, 0),
        np.conjugate(win_spectra.transpose(2, 0, 1)),
    )
    # welch_numerator = np.absolute(welch_numerator) ** 2

    return covariance, frequencies


def covariance_preprocessing(
    data: np.array,
    freq_smoothing_win: float = 0.33,
    time_smoothing_win: float = 1.25,
    sample_interval: int = 1,
) -> np.array:
    """
    Calculate the covariance matrix at all frequencies.

    Parameters
    ----------
    data : 1d or 2d numpy array
        Data in time for preprocessing
    freq_smoothing_win : int
        Length of frequency range for smoothing in Hz
    time_smoothing_win : int
        Length time window for smoothing in seconds
    sample_interval : float, optional
        Sample interval of the data. The default is 1.

    Returns
    -------
    preprocessed_data : numpy array
        preprocesses data. Same shape as input data

    """
    data_dims = data.ndim
    if data_dims == 1:
        data = data[np.newaxis, :]

    data_fft = np.fft.rfft(data[:])
    delta_freq = 1 / (len(data[0]) * sample_interval)
    freq_smoothing_win_len = int(freq_smoothing_win / delta_freq)

    running_avg = ss.fftconvolve(
        np.abs(data_fft),
        np.ones((len(data_fft), freq_smoothing_win_len))
        / freq_smoothing_win_len,
        mode="same",
        axes=1,
    )

    spectral_whitened = data_fft / running_avg

    whitened_time = np.fft.irfft(spectral_whitened)

    time_smoothing_win_len = int(time_smoothing_win / sample_interval)

    running_avg = ss.fftconvolve(
        np.abs(whitened_time),
        np.ones((len(whitened_time), time_smoothing_win_len))
        / time_smoothing_win_len,
        mode="same",
        axes=1,
    )

    preprocessed_data = whitened_time / running_avg

    if data_dims == 1:
        preprocessed_data = preprocessed_data[0]

    return preprocessed_data


def exact_coherence(
    data: np.array,
    subwindow_len: int,
    overlap: int = 0,
    resolution: float = 0.1,
    sample_interval: int = 1,
    max_freq: float = None,
    min_freq: float = 0,
) -> tuple:
    """
    Compute the detection significance from coherence.

    The detection significance is the ratio of the largest eigenvalue
    to the sum of all eigenvalues. This method computes the coherence matrix
    using the Welch method, and then computes the eigenvalues and subsequent
    detection significance at all frequencies.

    Parameters
    ----------
    data : numpy array
        Data in time for coherence analysis.
    subwindow_len : int
        Length of the subwindows in seconds.
    overlap : int, optional
        Overlap between adjacent subwindows in seconds.
        The default is 0.
    resolution : float, optional
        Resolution of the detection significance from 0 to 1.
        The default is 0.1.
    sample_interval : float, optional
        Sample interval of the data. The default is 1.

    Returns
    -------
    detection_significance : numpy array
        Detection significance of the data based on coherence
        computed using the exact method
    eigenvals : numpy array
        Eigenvalues of the coherence matrix

    """
    coherence, frequencies = welch_coherence(
        data, subwindow_len, overlap, sample_interval=sample_interval
    )
    if max_freq is None:
        max_freq = frequencies[-1]
    freq_select = (frequencies >= min_freq) & (frequencies <= max_freq)
    coherence = coherence[freq_select]
    frequencies = frequencies[freq_select]

    freq_interval = int(1 / resolution)
    coherence = coherence[::freq_interval]
    frequencies = frequencies[::freq_interval]

    num_frames = coherence.shape[0]
    # num_frames = int(num_frames * resolution)

    # Custom line due to apparent lowpass in BH data:
    # only use 3/5 of the frames
    # num_frames = int(num_frames * 2 / 5)

    num_subwindows = coherence.shape[2]
    detection_significance = np.empty(num_frames)
    # store the eigenvalues
    eigenvalss = np.empty((num_frames, num_subwindows))
    # freq_interval = int(1 / resolution)

    for d in range(num_frames):
        # eigenvals, _ = np.linalg.eig(coherence[d * freq_interval])
        # eigenvals = np.linalg.eigvalsh(coherence[d * freq_interval])
        eigenvals = np.linalg.eigvalsh(coherence[d])
        eigenvalss[d] = eigenvals[:num_subwindows]
        eigenvals = np.sort(eigenvals)[::-1]
        detection_significance[d] = eigenvals[0] / np.sum(eigenvals)

    return detection_significance, eigenvalss, frequencies


def svd_coherence(norm_win_spectra: np.ndarray, remove_center: str = None):
    """
    Compute the detection significance from SVD approximation of coherence.

    The detection significance is the ratio of the largest
    eigenvalue to the sum of all eigenvalues. This method computes the
    coherence matrix from the normalised spectra matrix provided, and then
    approximates the eigenvalues and subsequent detection significance at
    all frequencies using SVD.

    Parameters
    ----------
    norm_win_spectra : numpy array
        Normalized windowed spectra
    resolution : float, optional
        Resolution of the detection significance from 0 to 1.
        The default is 1.
    remove_center : str, optional
        Central tendency to remove center from data. Options are
        'mean' or 'median'.
        The default is None.

    Returns
    -------
    detection_significance : numpy array
        Detection significance of the data based on coherence
        computed using the SVD method
    svd_approxs : numpy array
        Approximation of the eigenvalues of the data using the
        SVD method

    """
    # num_frames = norm_win_spectra.shape[0]
    # num_frames = int(num_frames * resolution)

    # Custom line due to apparent lowpass in BH data:
    # only use 3/5 of the frames
    # num_frames = int(num_frames * 2 / 5)

    # num_subwindows = norm_win_spectra.shape[2]
    num_frames, num_channels, num_subwindows = norm_win_spectra.shape
    detection_significance = np.empty(num_frames)
    svd_approxs = np.empty((num_frames, min(num_channels, num_subwindows)))

    for d in range(num_frames):
        singular_values = np.linalg.svd(
            norm_win_spectra[d],
            compute_uv=False,
            hermitian=False,
        )
        svd_approx = singular_values**2
        svd_approxs[d] = svd_approx[: min(num_channels, num_subwindows)]
        detection_significance[d] = svd_approx[0] / np.sum(svd_approx)

    if remove_center == "mean":
        detection_significance -= np.mean(detection_significance)
    elif remove_center == "median":
        detection_significance -= np.median(detection_significance)

    return detection_significance, svd_approxs


def qr_coherence(norm_win_spectra: np.ndarray, remove_center: str = None):
    """
    Compute detection significance from QR decomposition approximation.

    The detection significance is the ratio of the
    largest eigenvalue to the sum of all eigenvalues. This method computes the
    coherence matrix from the normalised spectra matrix provided, and then
    approximates the eigenvalues and subsequent detection significance at all
    frequencies using QR decomposition.

    Parameters
    ----------
    norm_win_spectra : numpy array
        Normalized windowed spectra
    remove_center : str, optional
        Central tendency to remove center from data. Options are
        'mean' or 'median'.
        The default is None.

    Returns
    -------
    detection_significance : numpy array
        Detection significance of the data based on coherence
        computed using the QR decomposition
    qr_approxs : numpy array
        Approximation of the eigenvalues of the data using the QR
        decomposition

    """
    num_frames = norm_win_spectra.shape[0]

    # Custom line due to apparent lowpass in BH data:
    # only use 3/5 of the frames
    # num_frames = int(num_frames * 2 / 5)

    detection_significance = np.empty(num_frames)
    qr_approxs = np.empty((num_frames, np.min(norm_win_spectra.shape[1:])))

    for d in range(num_frames):
        r_matrix = np.linalg.qr(norm_win_spectra[d], mode="r")
        # qr_approx = np.diag(r_matrix @ np.conjugate(r_matrix.transpose()))
        qr_approx = np.sum(
            np.multiply(r_matrix, np.conjugate(r_matrix)).real, axis=1
        )
        # sorted_qr_approx = np.sort(qr_approx)[::-1]
        # detection_significance[d] = sorted_qr_approx[0] / np.sum(
        #     sorted_qr_approx
        # )
        detection_significance[d] = np.max(qr_approx) / np.sum(qr_approx)
        # detection_significance[d] = sorted_qr_approx[0] / np.sum(
        #     np.absolute(sorted_qr_approx)
        # )
        qr_approxs[d] = qr_approx

    if remove_center == "mean":
        detection_significance -= np.mean(detection_significance)
    elif remove_center == "median":
        detection_significance -= np.median(detection_significance)

    return detection_significance, qr_approxs


def rsvd_coherence(
    norm_win_spectra: np.ndarray,
    approx_rank: int = None,
    remove_center: str = None,
):
    """
    Compute detection significance from randomized SVD approximation.

    The detection significance is the ratio
    of the largest eigenvalue to the sum of all eigenvalues. This method
    computes the coherence matrix from the normalised spectra matrix provided,
    and then approximates the eigenvalues and subsequent detection significance
    at all frequencies using randomized SVD.

    Parameters
    ----------
    norm_win_spectra : numpy array
        Normalized windowed spectra
    resolution : float, optional
        Resolution of the detection significance from 0 to 1.
        The default is 1.
    approx_rank : int, optional
        Approximate rank for the randomized SVD method.
        The default is None.
    remove_center : str, optional
        Central tendency to remove center from data. Options are
        'mean' or 'median'.
        The default is None.

    Returns
    -------
    detection_significance : numpy array
        Detection significance of the data based on coherence
        computed using the randomized SVD method
    rsvd_approxs : numpy array
        Approximation of the eigenvalues of the data using the
        randomized SVD method

    """
    from sklearn.utils.extmath import randomized_svd  # type: ignore

    num_frames = norm_win_spectra.shape[0]
    # num_frames = int(num_frames * resolution)

    # Custom line due to apparent lowpass in BH data:
    # only use 3/5 of the frames
    # num_frames = int(num_frames * 2 / 5)

    if approx_rank is None:
        approx_rank = norm_win_spectra.shape[2]
    detection_significance = np.empty(num_frames)
    rsvd_approxs = np.empty((num_frames, approx_rank))

    for d in range(num_frames):
        _, singular_values, _ = randomized_svd(
            norm_win_spectra[d], approx_rank
        )
        rsvd_approx = singular_values**2
        rsvd_approxs[d] = rsvd_approx
        detection_significance[d] = rsvd_approx[0] / np.sum(rsvd_approx)

    if remove_center == "mean":
        detection_significance -= np.mean(detection_significance)
    elif remove_center == "median":
        detection_significance -= np.median(detection_significance)

    return detection_significance, rsvd_approxs


def qr_iteration(
    matrix: np.array, tol: float = 1e-6, max_iter: int = 1000
) -> np.array:
    """
    Compute the eigenvalues of matrix using the QR iteration method.

    Parameters
    ----------
    matrix : numpy array
        Matrix to compute the eigenvalues of
    tol : float, optional
        Tolerance for convergence. The default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. The default is 1000.

    Returns
    -------
    numpy array
        Eigenvalues of matrix

    """
    n = matrix.shape[0]
    q_matrix = np.eye(n)
    for i in range(max_iter):
        q_matrix, r_matrix = np.linalg.qr(matrix)
        matrix = r_matrix @ q_matrix
        if np.linalg.norm(np.tril(matrix, -1)) < tol:
            break
    return np.diag(matrix)


def power_iteration(
    matrix: np.array, tol: float = 1e-6, max_iter: int = 1000
) -> float:
    """
    Compute first eigenvalue of matrix using the power iteration method.

    Parameters
    ----------
    matrix : numpy array
        Matrix to compute the eigenvalues of
    tol : float, optional
        Tolerance for convergence. The default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. The default is 1000.

    Returns
    -------
    float
        Largest eigenvalue of matrix

    """
    n = matrix.shape[0]
    x = np.random.rand(n)
    for i in range(max_iter):
        new_x = matrix @ x
        new_x = new_x / np.linalg.norm(new_x)
        if np.linalg.norm(new_x - x) < tol:
            x = new_x
            break
        x = new_x
    return x @ matrix @ x


def coherence(
    data: np.array,
    subwindow_len: int,
    overlap: int,
    resolution: float = 1,
    sample_interval: float = 1,
    method: str = "exact",
    approx_rank: int = 10,
    max_freq: float = None,
    min_freq: float = 0,
    remove_center: str = None,
):
    """
    Compute a detection significance using coherence.

    Parameters
    ----------
    data : numpy array
        Data for coherence analysis
    subwindow_len : int
        Length of the subwindows in seconds
    overlap : int
        Overlap between adjacent subwindows in seconds
    Freq : int, optional
        Frequency to compute the coherence at, option is not
        implemented yet. The default is None. If None, the coherence is
        computed at all frequencies.
    sample_interval : float, optional
        Sample interval of the data. The default is 1.
    method : str, optional
        Method to use for coherence analysis.
        The default is 'exact'.
        Options are: 'exact', 'qr', 'svd', 'rsvd', 'power', 'qr iteration'
    approx_rank : int, optional
        Approximate rank for the randomized SVD method.
        The default is 10.
    remove_center : str, optional
        Central tendency to remove center from data. Options are
        'mean' or 'median'.
        The default is None.

    Returns
    -------
    detection_significance : numpy array
        Detection significance of the coherence of the data
        computed using the specified method

    Example
    --------
    data = np.random.rand(100, 1000)
    detection_significance = coherence(data, 10, 5, method='exact')
    """
    methods = ["exact", "qr", "svd", "rsvd", "power", "qr iteration"]
    assert method in methods, (
        f"Invalid method: {method}; valid methods are: {methods}"
    )

    if method in ["power", "qr iteration"]:
        raise NotImplementedError(f"Method {method} not implemented yet.")

    if method == "exact":
        return exact_coherence(
            data,
            subwindow_len,
            overlap,
            sample_interval=sample_interval,
            resolution=resolution,
            max_freq=max_freq,
            min_freq=min_freq,
            remove_center=remove_center,
        )
    else:
        norm_win_spectra, frequencies = normalised_windowed_spectra(
            data, subwindow_len, overlap, sample_interval=sample_interval
        )
        if max_freq is None:
            max_freq = frequencies[-1]
        freq_select = (frequencies >= min_freq) & (frequencies <= max_freq)
        norm_win_spectra = norm_win_spectra[freq_select]
        frequencies = frequencies[freq_select]

        freq_interval = int(1 / resolution)
        norm_win_spectra = norm_win_spectra[::freq_interval]
        frequencies = frequencies[::freq_interval]

        if method == "qr":
            detection_significance, eigenvals = qr_coherence(
                norm_win_spectra, remove_center=remove_center
            )
        elif method == "svd":
            detection_significance, eigenvals = svd_coherence(
                norm_win_spectra, remove_center=remove_center
            )
        elif method == "rsvd":
            detection_significance, eigenvals = rsvd_coherence(
                norm_win_spectra,
                approx_rank=approx_rank,
                remove_center=remove_center,
            )

        return detection_significance, eigenvals, frequencies


def rm_laser_drift(data: np.array) -> np.array:
    """
    Remove laser drift from DAS data.

    We do this by subtracting the median of each time
    sample across the channels from each channel at that time
    sample. This assumes the first dimension of the data is
    along the fibre.

    Parameters
    ----------
    data : numpy array
        Data to remove laser drift from

    Returns
    -------
    data : numpy array
        Data with laser drift removed

    Example
    --------
    data = np.random.rand(100, 1000)
    data = rm_laser_drift(data)
    """
    # compute median along fibre for each time sample
    med = np.median(data, axis=0)
    # subtract median from each corresponding time sample
    data = data - med[np.newaxis, :]

    return data


def frequency_filter(
    data: np.array, frequency_range, mode: str, order, sampling_frequency
):
    """
    Butterworth filter of data.

    Parameters
    ----------
    data : array
        1d or 2d array.
    frequency_range : int/sequence
        int if mode is lowpass or high pass. Sequence of 2 frequencies if mode
        is bandpass
    mode : str
        lowpass, highpass or bandpass.
    order : int
        Order of the filter.
    sampling_frequency : int
        sampling frequency.

    Returns
    -------
    filtered_data : array
        Frequency filtered data.

    """
    from scipy.signal import butter, sosfiltfilt

    sos = butter(
        order, frequency_range, btype=mode, output="sos", fs=sampling_frequency
    )
    filtered_data = sosfiltfilt(sos, data)

    return filtered_data


def stalta_freq(data, len_lt, len_st):
    """
    Compute the STALTA of data.

    Parameters
    ----------
    data : array
        1d or 2d array.
    len_lt : int
        Length of the long time average window in samples.
    len_st : int
        Length of the short time average window in samples.

    Returns
    -------
    stalta : array
        STALTA of data.
    """
    import scipy.signal as ss

    if data.ndim == 1:
        longtime_avg = ss.correlate(
            np.absolute(data), np.ones(len_lt), mode="valid"
        )
        shorttime_avg = ss.correlate(
            np.absolute(data[(len_lt - len_st) :]),
            np.ones(len_st),
            mode="valid",
        )
        stalta = (shorttime_avg * len_lt) / (longtime_avg * len_st)
    elif data.ndim == 2:
        nch, nsamples = data.shape
        stalta = np.empty((nch, nsamples - len_lt + 1), dtype=np.float64)
        longtime_stencil = np.ones(int(len_lt))
        shorttime_stencil = np.ones(int(len_st))
        for a in range(nch):
            longtime_avg = ss.correlate(
                np.absolute(data[a]), longtime_stencil, mode="valid"
            )
            shorttime_avg = ss.correlate(
                np.absolute(data[a, int(len_lt - len_st) :]),
                shorttime_stencil,
                mode="valid",
            )
            stalta[a] = (shorttime_avg * len_lt) / (longtime_avg * len_st)

    return stalta
