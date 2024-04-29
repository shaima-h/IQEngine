
# Copyright (c) 2023 Marc Lichtman.
# Licensed under the MIT License.


import numpy as np
import json
from pydantic.dataclasses import dataclass
import math


def findTransmitters(input, scale, beta, jaccard_threshold, max_gap_rows, fft_size):
    '''
    Gets parameters, calculates threshold, and runs algorithm.

    return:
    list[Transmitter]: detected transmitters
    '''
    # params[0] = mean of the pairwise difference of multiscale products
    # params[1] = std of the pairwise differences of multiscale products
    # beta is a threshold-scaling parameter that determines how many standard deviations from the mean should pairwise differences be in order to be ranked as a outlier local maxima
    params, regions = computeMaxHold(input, math.log2(input.shape[0]) - scale,
				math.log2(input.shape[0]) - (scale + 1)) #TODO in java, why did he use Util.log2 intead of Math.log 
                                                         #NOTE Util.log2 may not function for arrays?
    
    threshold = params[0] + params[1]*beta # threshold = mean + stdev*beta
    
    detected = findTransmittersMultiScale(input, regions, jaccard_threshold, scale, threshold, max_gap_rows)
    
    return detected # output annotations in main run function
    

@dataclass
class Plugin:
    sample_rate: int = 0
    center_freq: int = 0
    
    # custom params
    # TODO should we calculate optimal beta and scale?
    beta: float = 1.0
    scale: int = 3

    def run(self, samples):
        print(samples[0:10])
        print(self.sample_rate)
        print(self.center_freq)
        print(self.beta)
        print(self.scale)
        
        #PF, PT = compute_maxholds(p)
        
        #adaptive_threshold_selection()
        
        
        # Your Plugin (and optionally, classification) code here

        # turn samples into 2d matrix from 1d array
        fft_size = 1024
        num_rows = int(np.floor(len(samples)/fft_size))
        spectrogram = np.zeros((num_rows, fft_size))
        for i in range(num_rows):
            spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples[i*fft_size:(i+1)*fft_size])))**2)

        # print(spectrogram.shape)
        time_for_fft = fft_size * (1/self.sample_rate) *1000 # time it takes to traverse in ms
        max_gap_rows = math.ceil(0.0/time_for_fft) # TODO max_gap_milliseconds? always 0 in java code
        jaccard_threshold = 0.5 # if they are at least halfway overlapping, considered aligned
        detected = findTransmitters(spectrogram, self.scale, self.beta, jaccard_threshold, max_gap_rows, fft_size)
        
        # When making a detector, for the return, make a list, then for each detected emission, add one of these dicts to the list:
        annotations = []
        for transmitter in detected:
            print('*** detected: ', transmitter)

            start_row = transmitter.start_row
            start_col = transmitter.start_col
            end_row = transmitter.end_row
            end_col = transmitter.end_col

            print('*** what java would output:')
            print(f"{transmitter.start_col},{num_rows - transmitter.end_row},{transmitter.end_col - transmitter.start_col},{transmitter.end_row-transmitter.start_row}\n")

            x = start_col
            y = start_row
            width = end_col - start_col
            height = end_row - start_row
            
            if height > 0:
                an = {}
                an['core:freq_lower_edge'] = int(x / fft_size * self.sample_rate - (self.sample_rate / 2) + self.center_freq) # Hz
                an['core:freq_upper_edge'] = int((x + width) / fft_size * self.sample_rate - (self.sample_rate / 2) + self.center_freq) # Hz
                an['core:sample_start'] = int(y * fft_size)
                an['core:sample_count'] = int(height * fft_size)
                an["core:label"] = "Transmitter"# NOTE should we should set this to transmitters, since that is what AirView looks for?
                annotations.append(an)

        return {
            "data_output" : [],
            "annotations" : annotations
        }
    