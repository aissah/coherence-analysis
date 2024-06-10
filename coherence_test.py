'''
Test comparing the performance of various ways of doing coherence analysis
'''
import numpy as np
import sys

METHODS = ['exact', 'qr', 'svd', 'rsvd', 'power', 'qr iteration']

def windowed_spectra(data: np.array, subwindow_len: int,overlap,freq=None,sample_interval=1):
    """
    Calculate the frequency domain representation of data in windows.
    """

    win_start = 0
    window_samples = int(subwindow_len / sample_interval)
    total_samples = data.shape[-1]
    overlap = int(overlap/sample_interval)
    intervals = np.arange(window_samples, total_samples+1, window_samples, dtype=int) # break time series into windowed intervals

    win_end = intervals[0]

    absolute_spectra = np.fft.rfft(data[:,win_start:win_end])
    win_spectra = absolute_spectra[np.newaxis]

    while win_end < total_samples:
        win_start = win_end - overlap
        win_end = win_start + window_samples
        absolute_spectra = np.fft.rfft(data[:,win_start:win_end])
        win_spectra = np.append(
            win_spectra, absolute_spectra[np.newaxis], axis=0
        )
        # win_start = win_end
        
    
    frequencies = np.fft.rfftfreq(window_samples, sample_interval)

    return win_spectra, frequencies

def normalised_windowed_spectra(data: np.array, subwindow_len: int,overlap,freq=None,sample_interval=1):
    """
    Calculate the frequency domain representation of data in windows.
    """

    win_spectra, frequencies = windowed_spectra(data, subwindow_len,overlap,freq,sample_interval)

    normalizer = np.sum(np.absolute(win_spectra)**2, axis=0)
    normalizer = np.tile(np.sqrt(normalizer),(win_spectra.shape[0],1,1))
    normalizer = normalizer.transpose(2,1,0)

    normalized_spectra = win_spectra.transpose(2,1,0) / normalizer

    return normalized_spectra, frequencies

def welch_coherence(data: np.array, subwindow_len: int,overlap,freq=None,sample_interval=1):
    """
    Calculate the coherence matrix at all (or particular frequencies: yet to be implemented)
    using the welch method.
    """
    win_spectra, frequencies = windowed_spectra(data, subwindow_len,overlap,freq,sample_interval)

    normalizer = np.sum(np.absolute(win_spectra)**2, axis=0)
    normalizer = np.tile(normalizer,(normalizer.shape[0],1,1))
    normalizer = normalizer * normalizer.transpose((1,0,2))
    normalizer = normalizer.transpose(2,1,0)

    welch_numerator = np.matmul(win_spectra.transpose(2,1,0), np.conjugate(win_spectra.transpose(2,0,1)))
    welch_numerator = np.absolute(welch_numerator)**2
    coherence = np.multiply(welch_numerator,1/normalizer)

    return coherence, frequencies

def exact_coherence(data: np.array, subwindow_len: int,overlap,freq=None,sample_interval=1):
    '''
    Compute the k largest eigenvalues of A using the randomized SVD method
    '''
    coherence, _ = welch_coherence(data, subwindow_len, overlap, sample_interval=sample_interval)
    num_frames = coherence.shape[0]
    detection_significance = np.empty(num_frames)

    for d in range(num_frames):
        eigenvals, _ = np.linalg.eig(coherence[d])
        eigenvals = np.sort(eigenvals)[::-1]
        detection_significance[d] = eigenvals[0]/np.sum(eigenvals)
    
    return detection_significance

def svd_coherence(norm_win_spectra: np.ndarray):
    '''
    Compute the k largest eigenvalues of A using the randomized SVD method
    '''
    num_frames = norm_win_spectra.shape[0]
    detection_significance = np.empty(num_frames)

    for d in range(num_frames):
        _, S, _ = np.linalg.svd(norm_win_spectra[d*2]) 
        svd_approx = S**2
        detection_significance[d] = svd_approx[0]/np.sum(svd_approx)
    
    return detection_significance

def qr_coherence(norm_win_spectra: np.ndarray):
    '''
    Approximate the coherence of A using the QR decompositon
    '''
    num_frames = norm_win_spectra.shape[0]
    detection_significance = np.empty(num_frames)

    for d in range(num_frames):
        _,R = np.linalg.qr(norm_win_spectra[d])
        qr_approx = np.sort(np.diag(np.absolute(R@R.transpose())))[::-1]

        detection_significance[d] = qr_approx[0]/np.sum(np.absolute(qr_approx))
    
    return detection_significance

def rsvd_coherence(norm_win_spectra: np.ndarray, approx_rank: int =10):
    '''
    Compute the k largest eigenvalues of A using the randomized SVD method
    '''
    from sklearn.utils.extmath import randomized_svd
    num_frames = norm_win_spectra.shape[0]
    detection_significance = np.empty(num_frames)

    for d in range(num_frames):
        _, rS, _ = randomized_svd(norm_win_spectra[d], approx_rank) 
        rsvd_approx = rS**2
        detection_significance[d] = rsvd_approx[0]/np.sum(rsvd_approx)
    
    return detection_significance

def qr_iteration(A, tol=1e-6, max_iter=1000):
    '''
    Compute the eigenvalues of A using the QR iteration method
    '''
    n = A.shape[0]
    Q = np.eye(n)
    for i in range(max_iter):
        Q, R = np.linalg.qr(A)
        A = R @ Q
        if np.linalg.norm(np.tril(A, -1)) < tol:
            break
    return np.diag(A)

def power_iteration(A, tol=1e-6, max_iter=1000):
    '''
    Compute the eigenvalues of A using the power iteration method
    '''
    n = A.shape[0]
    x = np.random.rand(n)
    for i in range(max_iter):
        x = A @ x
        x = x/np.linalg.norm(x)
    return x @ A @ x

def coherence(data: np.array, subwindow_len: int,overlap: int,freq=None,sample_interval: float=1, method: str='exact', approx_rank: int=10):
    '''
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
    ''' 

    METHODS = ['exact', 'qr', 'svd', 'rsvd', 'power', 'qr iteration']
    if method == 'exact':
        return exact_coherence(data, subwindow_len, overlap, sample_interval=sample_interval)
    elif method == 'qr':
        norm_win_spectra, _ = normalised_windowed_spectra(data, subwindow_len, overlap, sample_interval=sample_interval)
        return qr_coherence(norm_win_spectra)
    elif method == 'svd':
        norm_win_spectra, _ = normalised_windowed_spectra(data, subwindow_len, overlap, sample_interval=sample_interval)
        return svd_coherence(norm_win_spectra)
    elif method == 'rsvd':
        norm_win_spectra, _ = normalised_windowed_spectra(data, subwindow_len, overlap, sample_interval=sample_interval)
        return rsvd_coherence(norm_win_spectra)
    elif method == 'power':
        return power_iteration(data, tol=1e-6, max_iter=1000)
    elif method == 'qr iteration':
        return qr_iteration(data, tol=1e-6, max_iter=1000)
    else:
        raise ValueError(f"Invalid method: {method}. Valid methods are: {METHODS}")

if __name__ == '__main__':
    file = int(sys.argv[1])
    averaging_window_length = int(sys.argv[2]) # Averaging window length in seconds
    sub_window_length = int(sys.argv[3]) # sub-window length in seconds
    overlap = int(sys.argv[4]) # overlap in seconds
    num_channels = int(sys.argv[5]) # number of sensors
    first_channel = int(sys.argv[6]) # first channel
    samples_per_sec = int(sys.argv[7]) # samples per second
    channel_offset = int(sys.argv[8]) # channel offset
    method = sys.argv[9] # method to use for coherence analysis

    file = r"D:\CSM\Mines_Research\Test_data\Brady Hotspring\PoroTomo_iDAS16043_160312000048.h5"
    data,_= loadBradyHShdf5(file,normalize='no')

    detection_significance = coherence(data[first_channel:channel_offset+first_channel:int(channel_offset/num_channels)], sub_window_length, overlap, sample_interval=1/samples_per_sec, method=method)

    
    