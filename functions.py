import h5py
import numpy as np


def loadBradyHShdf5(file, normalize="yes"):
    """

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
    """
    Calculate the frequency domain representation of data in windows.
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
    Calculate the coherence matrix at all (or particular frequencies: yet to be implemented)
    using the welch method.
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


def exact_coherence(
    data: np.array, subwindow_len: int, overlap, resolution=0.1, sample_interval=1
):
    """
    Compute the k largest eigenvalues of A using the randomized SVD method
    """
    coherence, _ = welch_coherence(
        data, subwindow_len, overlap, sample_interval=sample_interval
    )
    num_frames = coherence.shape[0]
    num_frames = int(num_frames * resolution)

    # Custom line due to apparent lowpass in BH data: only use 3/5 of the frames
    num_frames = int(num_frames * 3/5)

    num_subwindows = coherence.shape[2]
    detection_significance = np.empty(num_frames)
    eigenvalss = np.empty((num_frames, num_subwindows))  # store the eigenvalues

    for d in range(num_frames):
        eigenvals, _ = np.linalg.eig(coherence[d*int(1/resolution)])
        eigenvalss[d] = eigenvals[:num_subwindows]
        eigenvals = np.sort(eigenvals)[::-1]
        detection_significance[d] = eigenvals[0] / np.sum(eigenvals)

    return detection_significance, eigenvalss


def svd_coherence(norm_win_spectra: np.ndarray):
    """
    Compute the k largest eigenvalues of A using the randomized SVD method
    """
    num_frames = norm_win_spectra.shape[0]
    num_subwindows = norm_win_spectra.shape[2]
    detection_significance = np.empty(num_frames)
    svd_approxs = np.empty((num_frames, num_subwindows))

    for d in range(num_frames):
        _, S, _ = np.linalg.svd(norm_win_spectra[d * 2])
        svd_approx = S**2
        svd_approxs[d] = svd_approx[:num_subwindows]
        detection_significance[d] = svd_approx[0] / np.sum(svd_approx)

    return detection_significance, svd_approxs


def qr_coherence(norm_win_spectra: np.ndarray):
    """
    Approximate the coherence of A using the QR decompositon
    """
    num_frames = norm_win_spectra.shape[0]
    num_subwindows = norm_win_spectra.shape[2]
    detection_significance = np.empty(num_frames)
    qr_approxs = np.empty((num_frames, num_subwindows))

    for d in range(num_frames):
        _, R = np.linalg.qr(norm_win_spectra[d])
        qr_approx = np.diag(np.absolute(R @ R.transpose()))
        sorted_qr_approx = np.sort(qr_approx)[::-1]

        detection_significance[d] = sorted_qr_approx[0] / np.sum(
            np.absolute(sorted_qr_approx)
        )
        qr_approxs[d] = qr_approx

    return detection_significance, qr_approxs


def rsvd_coherence(norm_win_spectra: np.ndarray, approx_rank: int = None):
    """
    Compute the k largest eigenvalues of A using the randomized SVD method
    """
    from sklearn.utils.extmath import randomized_svd

    num_frames = norm_win_spectra.shape[0]
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
    """
    n = A.shape[0]
    x = np.random.rand(n)
    for i in range(max_iter):
        x = A @ x
        x = x / np.linalg.norm(x)
    return x @ A @ x


def coherence(
    data: np.array,
    subwindow_len: int,
    overlap: int,
    freq=None,
    sample_interval: float = 1,
    method: str = "exact",
    approx_rank: int = 10,
):
    """
    Compute the detection significance from coherence of data using the specified method.
    Parameters
    ----------
    data : numpy array
        DESCRIPTION. Data for coherence analysis
    subwindow_len : int
        DESCRIPTION. Length of the subwindows in seconds
    overlap : int
        DESCRIPTION. Overlap between adjacent subwindows in seconds
    Freq : int, optional
        DESCRIPTION. Frequency to compute the coherence at, option is not implemented yet. The default is None. If None, the coherence is computed at all frequencies
    sample_interval : float, optional
        DESCRIPTION. Sample interval of the data. The default is 1.
    method : str, optional
        DESCRIPTION. Method to use for coherence analysis. The default is 'exact'. Options are: 'exact', 'qr', 'svd', 'rsvd', 'power', 'qr iteration'
    approx_rank : int, optional
        DESCRIPTION. Approximate rank for the randomized SVD method. The default is 10.
    Returns
    -------
    detection_significance : numpy array
        DESCRIPTION. Detection significance of the coherence of the data computed using the specified method
    Example
    --------
    data = np.random.rand(100, 1000)
    detection_significance = coherence(data, 10, 5, method='exact')
    """

    METHODS = ["exact", "qr", "svd", "rsvd", "power", "qr iteration"]
    if method == "exact":
        return exact_coherence(
            data, subwindow_len, overlap, sample_interval=sample_interval
        )
    elif method == "qr":
        norm_win_spectra, _ = normalised_windowed_spectra(
            data, subwindow_len, overlap, sample_interval=sample_interval
        )
        return qr_coherence(norm_win_spectra)
    elif method == "svd":
        norm_win_spectra, _ = normalised_windowed_spectra(
            data, subwindow_len, overlap, sample_interval=sample_interval
        )
        return svd_coherence(norm_win_spectra)
    elif method == "rsvd":
        norm_win_spectra, _ = normalised_windowed_spectra(
            data, subwindow_len, overlap, sample_interval=sample_interval
        )
        return rsvd_coherence(norm_win_spectra)
    elif method == "power":
        return power_iteration(data, tol=1e-6, max_iter=1000)
    elif method == "qr iteration":
        return qr_iteration(data, tol=1e-6, max_iter=1000)
    else:
        raise ValueError(f"Invalid method: {method}. Valid methods are: {METHODS}")
