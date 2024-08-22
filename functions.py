import h5py
import numpy as np


def loadBradyHShdf5(file, normalize="yes"):
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
        nSamples = np.shape(data)[1]
        # get rid of laser drift
        med = np.median(data, axis=0)
        for i in range(nSamples):
            data[:, i] = data[:, i] - med[i]

        max_of_rows = abs(data[:, :]).sum(axis=1)
        data = data / max_of_rows[:, np.newaxis]
    return data, timestamp_arr


def windowed_spectra(
    data: np.array, subwindow_len: int, overlap, freq=None, sample_interval=1
):
    """

    Calculate the frequency domain representation of data in windows.

    Parameters
    ----------
    data : numpy array
        DESCRIPTION. Data in time domain
    subwindow_len : int
        DESCRIPTION. Length of the subwindows in seconds
    overlap : int
        DESCRIPTION. Overlap between adjacent subwindows in seconds
    freq : int, optional
        DESCRIPTION. Frequency to return the spectra at. The default is None.
        If None, the spectra is returned at all frequencies
    sample_interval : float, optional
        DESCRIPTION. Sample interval of the data. The default is 1.

    Returns
    -------
    win_spectra : numpy array
        DESCRIPTION. Spectra of the data in windows
    frequencies : numpy array
        DESCRIPTION. Frequencies at which the spectra is computed

    """

    win_start = 0
    window_samples = int(subwindow_len / sample_interval)
    total_samples = data.shape[-1]
    overlap = int(overlap / sample_interval)
    intervals = np.arange(
        window_samples, total_samples + 1, window_samples, dtype=int
    )  # break time series into windowed intervals

    win_end = intervals[0]

    absolute_spectra = np.fft.rfft(data[:, win_start:win_end])
    win_spectra = absolute_spectra[np.newaxis]

    while win_end < total_samples:
        win_start = win_end - overlap
        win_end = win_start + window_samples
        absolute_spectra = np.fft.rfft(data[:, win_start:win_end])
        win_spectra = np.append(win_spectra, absolute_spectra[np.newaxis], axis=0)
        # win_start = win_end

    frequencies = np.fft.rfftfreq(window_samples, sample_interval)

    return win_spectra, frequencies


def normalised_windowed_spectra(
    data: np.array, subwindow_len: int, overlap, freq=None, sample_interval=1
):
    win_spectra, frequencies = windowed_spectra(
        data, subwindow_len, overlap, freq, sample_interval
    )
    """
    Calculate the normalized frequency domain representation of data in
    windows.

    Parameters
    ----------
    data : numpy array
        DESCRIPTION. Data in time domain
    subwindow_len : int
        DESCRIPTION. Length of the subwindows in seconds
    overlap : int
        DESCRIPTION. Overlap between adjacent subwindows in seconds
    freq : int, optional
        DESCRIPTION. Frequency to return the spectra at. The default is None.
        If None, the spectra is returned at all frequencies
    sample_interval : float, optional
        DESCRIPTION. Sample interval of the data. The default is 1.

    Returns
    -------
    normalized_spectra : numpy array
        DESCRIPTION. Normalized spectra of the data. The normalization is done
        by dividing the spectra by the sum of the absolute values of the
        spectra squared of each channel
    frequencies : numpy array
        DESCRIPTION. Frequencies at which the spectra is computed

    """

    win_spectra, frequencies = windowed_spectra(
        data, subwindow_len, overlap, freq, sample_interval
    )

    normalizer = np.sum(np.absolute(win_spectra) ** 2, axis=0)
    normalizer = np.tile(np.sqrt(normalizer), (win_spectra.shape[0], 1, 1))
    normalizer = normalizer.transpose(2, 1, 0)

    normalized_spectra = win_spectra.transpose(2, 1, 0) / normalizer

    return normalized_spectra, frequencies


def welch_coherence(
    data: np.array, subwindow_len: int, overlap, freq=None, sample_interval=1
):
    """

    Calculate the coherence matrix at all (or particular frequencies: yet to be
    implemented) using the welch method.

    Parameters
    ----------
    data : numpy array
        DESCRIPTION. Data in time for coherence analysis
    subwindow_len : int
        DESCRIPTION. Length of the subwindows in seconds
    overlap : int
        DESCRIPTION. Overlap between adjacent subwindows in seconds
    freq : int, optional
        DESCRIPTION. Frequency to compute the coherence at. The default is
        None. If None, the coherence is computed at all frequencies
    sample_interval : float, optional
        DESCRIPTION. Sample interval of the data. The default is 1.

    Returns
    -------
    coherence : numpy array
        DESCRIPTION. Coherence matrix of the data
    frequencies : numpy array
        DESCRIPTION. Frequencies at which the coherence is computed

    """
    win_spectra, frequencies = windowed_spectra(
        data, subwindow_len, overlap, freq, sample_interval
    )

    normalizer = np.sum(np.absolute(win_spectra) ** 2, axis=0)
    normalizer = np.tile(normalizer, (normalizer.shape[0], 1, 1))
    normalizer = normalizer * normalizer.transpose((1, 0, 2))
    normalizer = normalizer.transpose(2, 1, 0)

    welch_numerator = np.matmul(
        win_spectra.transpose(2, 1, 0), np.conjugate(win_spectra.transpose(2, 0, 1))
    )
    welch_numerator = np.absolute(welch_numerator) ** 2
    coherence = np.multiply(welch_numerator, 1 / normalizer)

    return coherence, frequencies


def covariance(
    data: np.array, subwindow_len: int, overlap, freq=None, sample_interval=1
):
    """

    Calculate the covariance matrix at all (or particular frequencies: yet to
    be implemented).

    Parameters
    ----------
    data : numpy array
        DESCRIPTION. Data in time for covariance analysis
    subwindow_len : int
        DESCRIPTION. Length of the subwindows in seconds
    overlap : int
        DESCRIPTION. Overlap between adjacent subwindows in seconds
    freq : int, optional
        DESCRIPTION. Frequency to compute the covariance at. The default is
        None. If None, the covariance is computed at all frequencies
    sample_interval : float, optional
        DESCRIPTION. Sample interval of the data. The default is 1.

    Returns
    -------
    covariance : numpy array
        DESCRIPTION. Covariance matrix of the data
    frequencies : numpy array
        DESCRIPTION. Frequencies at which the coherence is computed

    """
    win_spectra, frequencies = windowed_spectra(
        data, subwindow_len, overlap, freq, sample_interval
    )

    covariance = np.matmul(
        win_spectra.transpose(2, 1, 0), np.conjugate(win_spectra.transpose(2, 0, 1))
    )
    # welch_numerator = np.absolute(welch_numerator) ** 2

    return covariance, frequencies


def exact_coherence(
    data: np.array,
    subwindow_len: int,
    overlap: int = 0,
    resolution: float = 0.1,
    sample_interval=1,
):
    """

    Compute the detection significance from coherence of data using the exact
    method. The detection significance is the ratio of the largest eigenvalue
    to the sum of all eigenvalues. This method computes the coherence matrix
    using the Welch method, and then computes the eigenvalues and subsequent
    detection significance at all frequencies.

    Parameters
    ----------
    data : numpy array
        DESCRIPTION. Data in time for coherence analysis.
    subwindow_len : int
        DESCRIPTION. Length of the subwindows in seconds.
    overlap : int, optional
        DESCRIPTION. Overlap between adjacent subwindows in seconds.
        The default is 0.
    resolution : float, optional
        DESCRIPTION. Resolution of the detection significance from 0 to 1.
        The default is 0.1.
    sample_interval : float, optional
        DESCRIPTION. Sample interval of the data. The default is 1.

    Returns
    -------
    detection_significance : numpy array
        DESCRIPTION. Detection significance of the data based on coherence
        computed using the exact method
    eigenvalss : numpy array
        DESCRIPTION. Eigenvalues of the coherence matrix

    """
    coherence, _ = welch_coherence(
        data, subwindow_len, overlap, sample_interval=sample_interval
    )
    num_frames = coherence.shape[0]
    num_frames = int(num_frames * resolution)

    # Custom line due to apparent lowpass in BH data:
    # only use 3/5 of the frames
    num_frames = int(num_frames * 2 / 5)

    num_subwindows = coherence.shape[2]
    detection_significance = np.empty(num_frames)
    # store the eigenvalues
    eigenvalss = np.empty((num_frames, num_subwindows))
    freq_interval = int(1 / resolution)

    for d in range(num_frames):
        # eigenvals, _ = np.linalg.eig(coherence[d * freq_interval])
        eigenvals = np.linalg.eigvalsh(coherence[d * freq_interval])
        eigenvalss[d] = eigenvals[:num_subwindows]
        eigenvals = np.sort(eigenvals)[::-1]
        detection_significance[d] = eigenvals[0] / np.sum(eigenvals)

    return detection_significance, eigenvalss


def svd_coherence(norm_win_spectra: np.ndarray, resolution: float = 1):
    """

    Compute the detection significance from coherence of data using an SVD
    approximation. The detection significance is the ratio of the largest
    eigenvalue to the sum of all eigenvalues. This method computes the
    coherence matrix from the normalised spectra matrix provided, and then
    approximates the eigenvalues and subsequent detection significance at
    all frequencies using SVD.

    Parameters
    ----------
    norm_win_spectra : numpy array
        DESCRIPTION. Normalized windowed spectra
    resolution : float, optional
        DESCRIPTION. Resolution of the detection significance from 0 to 1.
        The default is 0.1.

    Returns
    -------
    detection_significance : numpy array
        DESCRIPTION. Detection significance of the data based on coherence
        computed using the SVD method
    svd_approxs : numpy array
        DESCRIPTION. Approximation of the eigenvalues of the data using the
        SVD method

    """
    num_frames = norm_win_spectra.shape[0]
    num_frames = int(num_frames * resolution)

    # Custom line due to apparent lowpass in BH data:
    # only use 3/5 of the frames
    num_frames = int(num_frames * 2 / 5)

    num_subwindows = norm_win_spectra.shape[2]
    detection_significance = np.empty(num_frames)
    svd_approxs = np.empty((num_frames, num_subwindows))
    freq_interval = int(1 / resolution)

    for d in range(num_frames):
        # _, S, _ = np.linalg.svd(norm_win_spectra[d * freq_interval])
        S = np.linalg.svd(
            norm_win_spectra[d * freq_interval], compute_uv=False, hermitian=False
        )
        svd_approx = S**2
        svd_approxs[d] = svd_approx[:num_subwindows]
        detection_significance[d] = svd_approx[0] / np.sum(svd_approx)

    return detection_significance, svd_approxs


def qr_coherence(norm_win_spectra: np.ndarray, resolution: float = 1):
    """

    Compute the detection significance from coherence of data using a QR
    decomposition approximation. The detection significance is the ratio of the
    largest eigenvalue to the sum of all eigenvalues. This method computes the
    coherence matrix from the normalised spectra matrix provided, and then
    approximates the eigenvalues and subsequent detection significance at all
    frequencies using QR decomposition.

    Parameters
    ----------
    norm_win_spectra : numpy array
        DESCRIPTION. Normalized windowed spectra
    resolution : float, optional
        DESCRIPTION. Resolution of the detection significance from 0 to 1.
        The default is 0.1.

    Returns
    -------
    detection_significance : numpy array
        DESCRIPTION. Detection significance of the data based on coherence
        computed using
        the QR decomposition
    qr_approxs : numpy array
        DESCRIPTION. Approximation of the eigenvalues of the data using the QR
        decomposition

    """

    num_frames = norm_win_spectra.shape[0]
    num_frames = int(num_frames * resolution)

    # Custom line due to apparent lowpass in BH data:
    # only use 3/5 of the frames
    num_frames = int(num_frames * 2 / 5)

    # num_subwindows = norm_win_spectra.shape[2]
    detection_significance = np.empty(num_frames)
    qr_approxs = np.empty((num_frames, np.min(norm_win_spectra.shape[1:])))
    freq_interval = int(1 / resolution)

    for d in range(num_frames):
        _, R = np.linalg.qr(norm_win_spectra[d * freq_interval])
        qr_approx = np.diag(R @ np.conjugate(R.transpose()))
        sorted_qr_approx = np.sort(qr_approx)[::-1]
        detection_significance[d] = sorted_qr_approx[0] / np.sum(
            np.absolute(sorted_qr_approx)
        )
        qr_approxs[d] = qr_approx

    return detection_significance, qr_approxs


def rsvd_coherence(
    norm_win_spectra: np.ndarray, resolution: int = 1, approx_rank: int = None
):
    """
    Compute the detection significance from coherence of data using a
    randomized SVD approximation. The detection significance is the ratio
    of the largest eigenvalue to the sum of all eigenvalues. This method
    computes the coherence matrix from the normalised spectra matrix provided,
    and then approximates the eigenvalues and subsequent detection significance
    at all frequencies using randomized SVD.

    Parameters
    ----------
    norm_win_spectra : numpy array
        DESCRIPTION. Normalized windowed spectra
    resolution : float, optional
        DESCRIPTION. Resolution of the detection significance from 0 to 1.
        The default is 0.1.
    approx_rank : int, optional
        DESCRIPTION. Approximate rank for the randomized SVD method.
        The default is None.

    Returns
    -------
    detection_significance : numpy array
        DESCRIPTION. Detection significance of the data based on coherence
        computed using the randomized SVD method
    rsvd_approxs : numpy array
        DESCRIPTION. Approximation of the eigenvalues of the data using the
        randomized SVD method

    """
    from sklearn.utils.extmath import randomized_svd  # type: ignore

    num_frames = norm_win_spectra.shape[0]
    num_frames = int(num_frames * resolution)

    # Custom line due to apparent lowpass in BH data:
    # only use 3/5 of the frames
    num_frames = int(num_frames * 2 / 5)

    if approx_rank is None:
        approx_rank = norm_win_spectra.shape[2]
    detection_significance = np.empty(num_frames)
    rsvd_approxs = np.empty((num_frames, approx_rank))

    for d in range(num_frames):
        _, rS, _ = randomized_svd(norm_win_spectra[d], approx_rank)
        rsvd_approx = rS**2
        rsvd_approxs[d] = rsvd_approx
        detection_significance[d] = rsvd_approx[0] / np.sum(rsvd_approx)

    return detection_significance, rsvd_approxs


def qr_iteration(A, tol=1e-6, max_iter=1000):
    """
    Compute the eigenvalues of A using the QR iteration method

    Parameters
    ----------
    A : numpy array
        DESCRIPTION. Matrix to compute the eigenvalues of
    tol : float, optional
        DESCRIPTION. Tolerance for convergence. The default is 1e-6.
    max_iter : int, optional
        DESCRIPTION. Maximum number of iterations. The default is 1000.

    Returns
    -------
    numpy array
        DESCRIPTION. Eigenvalues of A

    """
    n = A.shape[0]
    Q = np.eye(n)
    for i in range(max_iter):
        Q, R = np.linalg.qr(A)
        A = R @ Q
        if np.linalg.norm(np.tril(A, -1)) < tol:
            break
    return np.diag(A)


def power_iteration(A, tol=1e-6, max_iter=1000):
    """
    Compute the eigenvalues of A using the power iteration method

    Parameters
    ----------
    A : numpy array
        DESCRIPTION. Matrix to compute the eigenvalues of
    tol : float, optional
        DESCRIPTION. Tolerance for convergence. The default is 1e-6.
    max_iter : int, optional
        DESCRIPTION. Maximum number of iterations. The default is 1000.

    Returns
    -------
    float
        DESCRIPTION. Largest eigenvalue of A

    """
    n = A.shape[0]
    x = np.random.rand(n)
    for i in range(max_iter):
        new_x = A @ x
        new_x = new_x / np.linalg.norm(new_x)
        if np.linalg.norm(new_x - x) < tol:
            x = new_x
            break
        x = new_x
    return x @ A @ x


def coherence(
    data: np.array,
    subwindow_len: int,
    overlap: int,
    resolution: float = 1,
    sample_interval: float = 1,
    method: str = "exact",
    approx_rank: int = 10,
):
    """
    Compute the detection significance from coherence of data using the
    specified method.

    Parameters
    ----------
    data : numpy array
        DESCRIPTION. Data for coherence analysis
    subwindow_len : int
        DESCRIPTION. Length of the subwindows in seconds
    overlap : int
        DESCRIPTION. Overlap between adjacent subwindows in seconds
    Freq : int, optional
        DESCRIPTION. Frequency to compute the coherence at, option is not
        implemented yet. The default is None. If None, the coherence is
        computed at all frequencies.
    sample_interval : float, optional
        DESCRIPTION. Sample interval of the data. The default is 1.
    method : str, optional
        DESCRIPTION. Method to use for coherence analysis.
        The default is 'exact'.
        Options are: 'exact', 'qr', 'svd', 'rsvd', 'power', 'qr iteration'
    approx_rank : int, optional
        DESCRIPTION. Approximate rank for the randomized SVD method.
        The default is 10.

    Returns
    -------
    detection_significance : numpy array
        DESCRIPTION. Detection significance of the coherence of the data
        computed using the specified method

    Example
    --------
    data = np.random.rand(100, 1000)
    detection_significance = coherence(data, 10, 5, method='exact')
    """

    METHODS = ["exact", "qr", "svd", "rsvd", "power", "qr iteration"]

    if method == "exact":
        return exact_coherence(
            data,
            subwindow_len,
            overlap,
            sample_interval=sample_interval,
            resolution=resolution,
        )
    elif method == "qr":
        norm_win_spectra, _ = normalised_windowed_spectra(
            data, subwindow_len, overlap, sample_interval=sample_interval
        )
        return qr_coherence(norm_win_spectra, resolution=resolution)
    elif method == "svd":
        norm_win_spectra, _ = normalised_windowed_spectra(
            data, subwindow_len, overlap, sample_interval=sample_interval
        )
        return svd_coherence(norm_win_spectra, resolution=resolution)
    elif method == "rsvd":
        norm_win_spectra, _ = normalised_windowed_spectra(
            data, subwindow_len, overlap, sample_interval=sample_interval
        )
        return rsvd_coherence(
            norm_win_spectra, resolution=resolution, approx_rank=approx_rank
        )
    elif method == "power":
        return power_iteration(data, tol=1e-6, max_iter=1000)
    elif method == "qr iteration":
        return qr_iteration(data, tol=1e-6, max_iter=1000)
    else:
        error_msg = f"Invalid method: {method}; valid methods are: {METHODS}"
        raise ValueError(error_msg)


def rm_laser_drift(data: np.array):
    """
    remove laser drift from DAS data by subtracting the median of each time
    sample. Assumes the first dimension of the data is along the fibre.

    Parameters
    ----------
    data : numpy array
        DESCRIPTION. Data to remove laser drift from

    Returns
    -------
    data : numpy array
        DESCRIPTION. Data with laser drift removed

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
