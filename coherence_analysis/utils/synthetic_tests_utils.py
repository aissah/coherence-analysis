"""Synthetic experiment utility functions."""

from typing import Optional

import numpy as np
import scipy.signal as ss
import torch

import coherence_analysis.utils.utils as utils


def minimum_phase_wavelet(wavelet: np.ndarray) -> np.ndarray:
    """
    Convert arbitrary real wavelet to minimum-phase.

    Uses log-amplitude Hilbert transform.

    Parameters
    ----------
    wavelet : numpy array or torch tensor
        Input wavelet in time domain
    """
    wavelet_spectrum = np.fft.fft(wavelet)

    # amplitude spectrum
    spectrum_amplitude = np.abs(wavelet_spectrum)
    # avoid log(0)
    spectrum_amplitude[spectrum_amplitude == 0] = 1e-12

    # log amplitude
    log_spectrum_amp = np.log(spectrum_amplitude)

    # Hilbert transform → phase
    # phase = np.imag(hilbert(logA))
    phase = -np.imag(ss.hilbert(log_spectrum_amp))

    # minimum phase spectrum
    w_min = np.exp(log_spectrum_amp + 1j * phase)

    # invert to time
    w_min = np.real(np.fft.ifft(w_min))
    return w_min


def berlage_wavelet(
    freq: float,
    length: int,
    dt: float,
    peak_time: float,
    n: int = 2,
    alpha: float = 4.0,
    phi: float = 0.0,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Create berlage wavelet.

    Parameters
    ----------
    freq : float
        Frequency of the wavelet in Hz
    length : int
        Length of the wavelet in samples
    dt : float
        Sampling interval in seconds
    peak_time : float
        Time of the peak amplitude in seconds
    n : int, optional
        Exponent parameter (default is 2)
    alpha : float, optional
        Damping factor (default is 4.0)
    phi : float, optional
        Phase shift in radians (default is 0.0)
    dtype : torch.dtype, optional
        Desired data type of the output tensor (default is None)
    """
    t = np.arange(0, length) * dt
    t = t - peak_time
    w = np.zeros_like(t)
    tp = t[t >= 0]
    w[t >= 0] = (
        (tp**n) * np.exp(-alpha * tp) * np.cos(2 * np.pi * freq * tp + phi)
    )
    # w = w / np.max(np.abs(w))  # normalize

    # normalize in frequency domain
    spectra = np.fft.rfft(w)
    spectra /= np.max(np.abs(spectra))
    w = np.fft.irfft(spectra)

    w = np.array(w, dtype=np.float32)
    w = torch.from_numpy(w)
    return w.to(dtype)


def gabor_wavelet(t: np.ndarray, fc: float, alpha: float) -> np.ndarray:
    """
    Create Gabor wavelet.

    Parameters
    ----------
    t : numpy array
        Time array
    fc : float
        Central frequency in Hz
    alpha : float
        Gaussian width parameter

    Returns
    -------
    numpy array
        Gabor wavelet evaluated at time t
    """
    return np.exp(-alpha * t**2) * np.cos(2 * np.pi * fc * t)


def ormsby_wavelet(
    freqs: list, length: int, dt: float, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Create Ormsby wavelet via analytic time-domain expression.

    Parameters
    ----------
    freqs : list
        List of four corner frequencies [f1, f2, f3, f4] in Hz
    length : int
        Length of the wavelet in samples
    dt : float
        Sampling interval in seconds
    dtype : torch.dtype, optional
        Desired data type of the output tensor (default is None)

    Returns
    -------
    torch.Tensor
        Ormsby wavelet of specified length and dtype.
    """
    # dt = 0.005
    t = np.arange(-length // 2, length // 2) * dt
    # t = torch.arange(float(length), dtype=dtype) * dt - length//2 * dt
    f1, f2, f3, f4 = freqs
    pi_mod = np

    def sinc(x):
        # return torch.sinc(x)
        return np.sinc(x / pi_mod.pi)

    term1 = pi_mod.pi * (f4) ** 2 * sinc(pi_mod.pi * f4 * t) ** 2
    term2 = pi_mod.pi * (f3) ** 2 * sinc(pi_mod.pi * f3 * t) ** 2
    term3 = pi_mod.pi * (f2) ** 2 * sinc(pi_mod.pi * f2 * t) ** 2
    term4 = pi_mod.pi * (f1) ** 2 * sinc(pi_mod.pi * f1 * t) ** 2

    out = (term1 - term2) / (f2 - f1) - (term3 - term4) / (f4 - f3)
    # out = (term1 - term2) - (term3 - term4)
    # out = out / np.max(np.abs(out))  # normalize
    out = torch.from_numpy(out)
    return out.to(dtype)


def min_phase_ormsby_wavelet(
    freqs: list,
    length: int,
    dt: float,
    peak_time: float,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Create minimum-phase Ormsby wavelet.

    Parameters
    ----------
    freqs : list
        List of four corner frequencies [f1, f2, f3, f4] in Hz
    length : int
        Length of the wavelet in samples
    dt : float
        Sampling interval in seconds
    peak_time : float
        Time of the peak amplitude in seconds
    dtype : torch.dtype, optional
        Desired data type of the output tensor (default is None)
    """
    w = ormsby_wavelet(freqs, length, dt, dtype)
    # w /= torch.max(torch.abs(w))  # normalize
    w_min = minimum_phase_wavelet(w)
    # w_min /= np.max(np.abs(w_min))  # normalize

    # NORMALIZE IN FREQUENCY DOMAIN
    spectra = np.fft.rfft(w_min)
    spectra /= np.max(np.abs(spectra))
    w_min = np.fft.irfft(spectra)

    w_min = np.roll(w_min, int(peak_time / dt))
    w_min = np.array(w_min, dtype=np.float32)
    w_min = torch.from_numpy(w_min)
    return w_min.to(dtype)


def get_noise(data, cov_len):
    """
    Generate correlated noise based on covariance length.

    Parameters
    ----------
    data : array
        2d array.
    cov_len : int
        Length of the covariance.

    Returns
    -------
    noise : array
        Correlated noise.
    """
    nr, nc = data.shape
    noise_cov = np.eye(nr)
    for i in range(1, cov_len + 1):
        noise_cov += (
            np.diag(np.ones(nr - i), -i) + np.diag(np.ones(nr - i), i)
        ) * np.exp(-10 * i / cov_len)

    noise = np.random.multivariate_normal(
        mean=np.zeros(nr), cov=noise_cov, size=nc
    )
    noise = noise.T
    noise = noise / np.max(np.abs(noise))

    return noise


def noisy_data(data, signal_to_noise, cov_len=5):
    """
    Add correlated noise to data based on signal to noise ratio.

    Parameters
    ----------
    data : array
        2d array.
    signal_to_noise : float
        Signal to noise ratio.
    cov_len : int
        Length of the covariance.

    Returns
    -------
    noisy_data : array
        Data with added correlated noise.
    noise : array
        Correlated noise.
    """
    noise = get_noise(data, cov_len)
    noise = noise * np.max(np.abs(data)) / signal_to_noise
    return data + noise, noise


def get_event_frequency_indices(
    frequencies: np.ndarray,
    event_freq_range: int | list | tuple,
) -> np.ndarray:
    """
    Get indices of frequencies corresponding to the event frequency range.

    Parameters
    ----------
    frequencies : numpy array
        Array of frequencies.
    event_freq_range : int or list or tuple
        Frequency range of the event.

    Returns
    -------
    numpy array
        Indices of frequencies corresponding to the event frequency range.
    """
    if isinstance(event_freq_range, int):
        event_freq_inds = (
            np.abs(frequencies - event_freq_range)
            <= (frequencies[1] - frequencies[0]) / 2
        )
    else:
        event_freq_inds = (frequencies >= event_freq_range[0]) & (
            frequencies <= event_freq_range[1]
        )

    return event_freq_inds


def single_detection_test(
    coherence_data: np.ndarray,
    win_len: int,
    overlap: float,
    sample_interval: float,
    signal_to_noise: float,
    cov_len: int,
    event_freq_range: int | list | tuple = None,
):
    """
    Perform noise test on coherence data and return results as a dataframe.

    Parameters
    ----------
    coherence_data : numpy array
        Data for coherence analysis
    win_len : int
        Length of the subwindows in seconds
    overlap : float
        Overlap between adjacent subwindows in seconds
    sample_interval : float
        Sample interval of the data.
    signal_to_noise : float
        Signal to noise ratio to test.
    cov_len : int
        Covariance length to test.
    event_freq_range : int or list or tuple
        Frequency range of the event.
    num_of_sims : int
        Number of simulations to run for statistical significance.
    """
    noisy_coherence_data, noise = noisy_data(
        coherence_data, signal_to_noise, cov_len=cov_len
    )

    svd_event_detection, _, frequencies = utils.coherence(
        noisy_coherence_data,
        win_len,
        overlap,
        sample_interval=sample_interval,
        method="svd",
    )

    qr_event_detection, _, frequencies = utils.coherence(
        noisy_coherence_data,
        win_len,
        overlap,
        sample_interval=sample_interval,
        method="qr",
    )

    ratio_event_detection = svd_event_detection / qr_event_detection

    if event_freq_range is not None:
        event_freq_inds = get_event_frequency_indices(
            frequencies, event_freq_range
        )
        qr_events_list = list(qr_event_detection[event_freq_inds])
        svd_events_list = list(svd_event_detection[event_freq_inds])
        events_ratio_list = list(ratio_event_detection[event_freq_inds])
    else:
        qr_events_list = list(qr_event_detection)
        svd_events_list = list(svd_event_detection)
        events_ratio_list = list(ratio_event_detection)

    svd_noise_detection, _, _ = utils.coherence(
        noise,
        win_len,
        overlap,
        sample_interval=sample_interval,
        method="svd",
    )

    qr_noise_detection, _, _ = utils.coherence(
        noise,
        win_len,
        overlap,
        sample_interval=sample_interval,
        method="qr",
    )

    qr_noise_list = list(qr_noise_detection)
    svd_noise_list = list(svd_noise_detection)
    events_ratio_noise_list = list(svd_noise_detection / qr_noise_detection)

    return (
        svd_events_list,
        qr_events_list,
        events_ratio_list,
        svd_noise_list,
        qr_noise_list,
        events_ratio_noise_list,
    )


def noise_test_sequential_sampling(
    coherence_data: np.ndarray,
    win_len: int,
    overlap: float,
    sample_interval: float,
    signal_to_noise_list: list,
    cov_len_list: list,
    event_freq_range: int | list | tuple,
    min_sims: int = 20,
    max_sims: int = 100,
    alpha: float = 0.05,
    epsilon: float = 0.1,
):
    """
    Perform noise test on coherence data and return results as a dataframe.

    Parameters
    ----------
    coherence_data : numpy array
        Data for coherence analysis
    win_len : int
        Length of the subwindows in seconds
    overlap : float
        Overlap between adjacent subwindows in seconds
    sample_interval : float
        Sample interval of the data.
    signal_to_noise_list : list
        List of signal to noise ratios to test.
    cov_len_list : list
        List of covariance lengths to test.
    event_freq_range : int or list or tuple
        Frequency range of the event.
    min_sims : int
        Minimum number of simulations to run for statistical significance.
    max_sims : int
        Maximum number of simulations to run for statistical significance.
    alpha : float
        Significance level for confidence interval.
    epsilon : float
        Acceptable margin of error for the estimate.

    Returns
    -------
    df : pandas DataFrame
    """
    import pandas as pd
    from scipy.stats import norm

    z = norm.ppf(1 - alpha / 2)

    qr_events_list = []
    svd_events_list = []
    events_ratio_list = []
    event_labels = []
    signal_to_n_list = []
    cov_len_df_list = []
    for cov_len in cov_len_list:
        for signal_to_noise in signal_to_noise_list:
            working_svd_events_list = []
            working_qr_events_list = []
            working_events_ratio_list = []
            working_svd_noise_list = []
            working_qr_noise_list = []
            working_events_ratio_noise_list = []
            for a in range(min_sims):
                (
                    new_svd_events,
                    new_qr_events,
                    new_events_ratio,
                    new_svd_noise,
                    new_qr_noise,
                    new_events_ratio_noise,
                ) = single_detection_test(
                    coherence_data,
                    win_len,
                    overlap,
                    sample_interval,
                    signal_to_noise,
                    cov_len,
                    event_freq_range,
                )

                working_svd_events_list.extend(new_svd_events)
                working_qr_events_list.extend(new_qr_events)
                working_events_ratio_list.extend(new_events_ratio)
                working_svd_noise_list.extend(new_svd_noise)
                working_qr_noise_list.extend(new_qr_noise)
                working_events_ratio_noise_list.extend(new_events_ratio_noise)

            num_of_sims = min_sims
            # Sequential sampling logic to determine if more sims are needed
            while num_of_sims <= max_sims:
                # Compute means and variances
                qr_event_var = np.var(working_qr_events_list)
                qr_noise_var = np.var(working_qr_noise_list)
                svd_event_var = np.var(working_svd_events_list)
                svd_noise_var = np.var(working_svd_noise_list)

                var_list = [
                    qr_event_var,
                    qr_noise_var,
                    svd_event_var,
                    svd_noise_var,
                ]

                # Calculate the margin of error
                margin_of_error = z * np.sqrt(
                    (svd_event_var / num_of_sims)
                    + (svd_noise_var / num_of_sims)
                )
                margin_of_error = z * np.sqrt(max(var_list) / num_of_sims)

                # Check if the margin of error is within the acceptable range
                if (
                    margin_of_error < epsilon
                ):  # * abs(svd_event_mean - svd_noise_mean):
                    break  # Stop if within acceptable range

                # If not, perform another simulation
                (
                    new_svd_events,
                    new_qr_events,
                    new_events_ratio,
                    new_svd_noise,
                    new_qr_noise,
                    new_events_ratio_noise,
                ) = single_detection_test(
                    coherence_data,
                    win_len,
                    overlap,
                    sample_interval,
                    signal_to_noise,
                    cov_len,
                    event_freq_range,
                )

                working_svd_events_list.extend(new_svd_events)
                working_qr_events_list.extend(new_qr_events)
                working_events_ratio_list.extend(new_events_ratio)
                working_svd_noise_list.extend(new_svd_noise)
                working_qr_noise_list.extend(new_qr_noise)
                working_events_ratio_noise_list.extend(new_events_ratio_noise)

                num_of_sims += 1
            svd_events_list.extend(working_svd_events_list)
            qr_events_list.extend(working_qr_events_list)
            events_ratio_list.extend(working_events_ratio_list)
            svd_events_list.extend(working_svd_noise_list)
            qr_events_list.extend(working_qr_noise_list)
            events_ratio_list.extend(working_events_ratio_noise_list)

            signal_to_n_list.extend(
                [signal_to_noise]
                * (len(working_qr_events_list) + len(working_qr_noise_list))
            )
            cov_len_df_list.extend(
                [cov_len]
                * (len(working_qr_events_list) + len(working_qr_noise_list))
            )
            event_labels.extend(["Signal"] * len(working_svd_events_list))
            event_labels.extend(["Noise"] * len(working_svd_noise_list))

    df = pd.DataFrame(
        {
            "Signal/Noise": signal_to_n_list * 3,
            "Detection Parameter": svd_events_list
            + qr_events_list
            + events_ratio_list,
            "Method": ["svd"] * len(svd_events_list)
            + ["qr"] * len(qr_events_list)
            + ["ratio"] * len(events_ratio_list),
            "Data Label": event_labels * 3,
            "Covariance Length": cov_len_df_list * 3,
        }
    )

    return df, num_of_sims


def noise_test(
    coherence_data: np.ndarray,
    win_len: int,
    overlap: float,
    sample_interval: float,
    signal_to_noise_list: list,
    cov_len_list: list,
    event_freq_range: int | list | tuple,
    num_of_sims: int,
):
    """
    Perform noise test on coherence data and return results as a dataframe.

    Parameters
    ----------
    coherence_data : numpy array
        Data for coherence analysis
    win_len : int
        Length of the subwindows in seconds
    overlap : float
        Overlap between adjacent subwindows in seconds
    sample_interval : float
        Sample interval of the data.
    signal_to_noise_list : list
        List of signal to noise ratios to test.
    cov_len_list : list
        List of covariance lengths to test.
    event_freq_range : int or list or tuple
        Frequency range of the event.
    num_of_sims : int
        Number of simulations to run for statistical significance.
    """
    import pandas as pd

    qr_events_list = []
    svd_events_list = []
    events_ratio_list = []
    event_labels = []
    signal_to_n_list = []
    cov_len_df_list = []
    event_freq_inds = None
    for cov_len in cov_len_list:
        for signal_to_noise in signal_to_noise_list:
            for a in range(num_of_sims):
                noisy_coherence_data, noise = noisy_data(
                    coherence_data, signal_to_noise, cov_len=cov_len
                )

                svd_event_detection, _, frequencies = utils.coherence(
                    noisy_coherence_data,
                    win_len,
                    overlap,
                    sample_interval=sample_interval,
                    method="svd",
                )

                qr_event_detection, _, frequencies = utils.coherence(
                    noisy_coherence_data,
                    win_len,
                    overlap,
                    sample_interval=sample_interval,
                    method="qr",
                )

                ratio_event_detection = (
                    svd_event_detection / qr_event_detection
                )

                if event_freq_inds is None:
                    event_freq_inds = get_event_frequency_indices(
                        frequencies, event_freq_range
                    )

                # try:
                #    qr_events_list.extend(qr_event_detection[event_freq_inds])
                # except NameError:
                #    if isinstance(event_freq_range, int):
                #        event_freq_inds = (
                #            np.abs(frequencies - event_freq_range)
                #            <= (frequencies[1] - frequencies[0]) / 2
                #        )
                #    else:
                #        event_freq_inds = (
                #            frequencies >= event_freq_range[0]
                #        ) & (frequencies <= event_freq_range[1])
                #    qr_events_list.extend(qr_event_detection[event_freq_inds])

                qr_events_list.extend(qr_event_detection[event_freq_inds])
                svd_events_list.extend(svd_event_detection[event_freq_inds])
                events_ratio_list.extend(
                    ratio_event_detection[event_freq_inds]
                )

                event_labels.extend(["signal"] * np.sum(event_freq_inds))
                # signal_to_n_list.extend(
                #     [signal_to_noise] * np.sum(event_freq_inds)
                # )
                # cov_len_df_list.extend([cov_len] * np.sum(event_freq_inds))

                svd_noise_detection, _, _ = utils.coherence(
                    noise,
                    win_len,
                    overlap,
                    sample_interval=sample_interval,
                    method="svd",
                )

                qr_noise_detection, _, _ = utils.coherence(
                    noise,
                    win_len,
                    overlap,
                    sample_interval=sample_interval,
                    method="qr",
                )

                qr_events_list.extend(qr_noise_detection)
                svd_events_list.extend(svd_noise_detection)
                events_ratio_list.extend(
                    svd_noise_detection / qr_noise_detection
                )

                event_labels.extend(["noise"] * len(qr_noise_detection))
                # signal_to_n_list.extend(
                #     [signal_to_noise] * len(qr_noise_detection)
                # )
                # cov_len_df_list.extend([cov_len] * len(qr_noise_detection))

            signal_to_n_list.extend(
                [signal_to_noise]
                * (np.sum(event_freq_inds) + len(qr_noise_detection))
                * num_of_sims
            )
        cov_len_df_list.extend(
            [cov_len]
            * (np.sum(event_freq_inds) + len(qr_noise_detection))
            * num_of_sims
            * len(signal_to_noise_list)
        )

    df = pd.DataFrame(
        {
            "Signal_to_Noise": signal_to_n_list * 3,
            "Detection_Parameter": svd_events_list
            + qr_events_list
            + events_ratio_list,
            "Method": ["svd"] * len(svd_events_list)
            + ["qr"] * len(qr_events_list)
            + ["ratio"] * len(events_ratio_list),
            "Data_Label": event_labels * 3,
            "Covariance_Length": cov_len_df_list * 3,
        }
    )

    return df


def coherence_decay_test(
    coherence_data: np.ndarray,
    win_len: int,
    overlap: float,
    sample_interval: float,
    signal_to_noise: float,
    cov_len: int,
    event_freq_range: int | list | tuple,
    num_of_sims: int,
):
    """
    Perform noise test on coherence data and return results as a dataframe.

    Parameters
    ----------
    coherence_data : numpy array
        Data for coherence analysis
    win_len : int
        Length of the subwindows in seconds
    overlap : float
        Overlap between adjacent subwindows in seconds
    sample_interval : float
        Sample interval of the data.
    signal_to_noise : float
        Signal to noise ratio to test.
    cov_len : int
        Covariance length to test.
    event_freq_range : int or list or tuple
        Frequency range of the event.
    num_of_sims : int
        Number of simulations to run for statistical significance.
    """
    import pandas as pd

    nwins = None
    for a in range(num_of_sims):
        noisy_coherence_data, noise = noisy_data(
            coherence_data, signal_to_noise, cov_len=cov_len
        )

        _, svds_signal, frequencies = utils.coherence(
            noisy_coherence_data,
            win_len,
            overlap,
            sample_interval=sample_interval,
            method="svd",
        )

        _, qrs_signal, frequencies = utils.coherence(
            noisy_coherence_data,
            win_len,
            overlap,
            sample_interval=sample_interval,
            method="qr",
        )
        if nwins is None:
            nwins = qrs_signal.shape[1]
            noise_nreps = qrs_signal.shape[0]
        # ratio_event_detection = (
        #     svd_event_detection / qr_event_detection
        # )

        try:
            qr_decays = qrs_signal[event_freq_inds]
            qr_decays_df = pd.DataFrame(qr_decays.flatten())
            qr_decays_df["Method"] = "qr"
            qr_decays_df["Data_Label"] = "signal"
            qr_decays_df["Index"] = np.tile(np.arange(1, nwins + 1), nreps)
            signal_decays_df = pd.concat(
                [signal_decays_df, qr_decays_df], ignore_index=True
            )
        except NameError:
            if isinstance(event_freq_range, int):
                event_freq_inds = (
                    np.abs(frequencies - event_freq_range)
                    <= (frequencies[1] - frequencies[0]) / 2
                )
                nreps = 1
            else:
                event_freq_inds = (frequencies >= event_freq_range[0]) & (
                    frequencies <= event_freq_range[1]
                )
                nreps = np.sum(event_freq_inds)
            qr_decays = qrs_signal[event_freq_inds]
            signal_decays_df = pd.DataFrame(qr_decays.flatten())
            signal_decays_df["Method"] = "qr"
            signal_decays_df["Data_Label"] = "signal"
            signal_decays_df["Index"] = np.tile(np.arange(1, nwins + 1), nreps)

        svd_decays = svds_signal[event_freq_inds]
        svd_decays_df = pd.DataFrame(svd_decays.flatten())
        svd_decays_df["Method"] = "svd"
        svd_decays_df["Data_Label"] = "signal"
        svd_decays_df["Index"] = np.tile(np.arange(1, nwins + 1), nreps)

        signal_decays_df = pd.concat(
            [signal_decays_df, svd_decays_df], ignore_index=True
        )

        _, svds_noise, _ = utils.coherence(
            noise,
            win_len,
            overlap,
            sample_interval=sample_interval,
            method="svd",
        )

        _, qrs_noise, _ = utils.coherence(
            noise,
            win_len,
            overlap,
            sample_interval=sample_interval,
            method="qr",
        )

        svd_decays_df = pd.DataFrame(svds_noise.flatten())
        svd_decays_df["Method"] = "svd"
        svd_decays_df["Data_Label"] = "noise"
        svd_decays_df["Index"] = np.tile(np.arange(1, nwins + 1), noise_nreps)

        qr_decays_df = pd.DataFrame(qrs_noise.flatten())
        qr_decays_df["Method"] = "qr"
        qr_decays_df["Data_Label"] = "noise"
        qr_decays_df["Index"] = np.tile(np.arange(1, nwins + 1), noise_nreps)
        signal_decays_df = pd.concat(
            [signal_decays_df, svd_decays_df, qr_decays_df], ignore_index=True
        )

    return signal_decays_df
