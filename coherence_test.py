'''
Test comparing the performance of various ways of doing coherence analysis
# python coherence_test.py <file> <averaging_window_length> <sub_window_length> <overlap> <first_channel> <num_channels> <samples_per_sec> <channel_offset> <method>
# python coherence_test.py "D:\CSM\Mines_Research\Test_data\Brady Hotspring\PoroTomo_iDAS16043_160312000048.h5" 0 2 0 3100 200 1000 2000 exact
'''
from pathlib import Path
import sys

# import numpy as np
import pickle

import functions as func

METHODS = ['exact', 'qr', 'svd', 'rsvd', 'power', 'qr iteration']
save_location = Path("D:/CSM/Mines_Research/Repositories/Coherence_Analyses/test_results")

if __name__ == '__main__':
    file = sys.argv[1]
    averaging_window_length = int(sys.argv[2]) # Averaging window length in seconds
    sub_window_length = int(sys.argv[3]) # sub-window length in seconds
    overlap = int(sys.argv[4]) # overlap in seconds
    first_channel = int(sys.argv[5]) # first channel
    num_channels = int(sys.argv[6]) # number of sensors
    samples_per_sec = int(sys.argv[7]) # samples per second
    channel_offset = int(sys.argv[8]) # channel offset
    method = sys.argv[9] # method to use for coherence analysis

    file = Path(file) # r"D:\CSM\Mines_Research\Test_data\Brady Hotspring\PoroTomo_iDAS16043_160312000048.h5"
    data,_= func.loadBradyHShdf5(file,normalize='no')

    detection_significance = func.coherence(data[first_channel:channel_offset+first_channel:int(channel_offset/num_channels)], sub_window_length, overlap, sample_interval=1/samples_per_sec, method=method)
    print(detection_significance.shape, flush=True)
    savename = save_location / f"{method}_detection_significance_{str(file)[-15:-3]}.pkl"
    with open(savename, 'wb') as f:
        pickle.dump(detection_significance, f)
    