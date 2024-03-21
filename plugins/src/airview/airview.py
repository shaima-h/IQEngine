''' TODO Notes
- no filef and output_file (from FindTransmitters.java)
- fix and make proper function descriptions/comments
- function names convention? underscores or camel case?
'''

# Copyright (c) 2023 Marc Lichtman.
# Licensed under the MIT License.

import numpy as np
import json
from pydantic.dataclasses import dataclass
import math
import find_parameters
import multi_scale

def findTransmitters(input, scale, beta, jaccard_threshold, max_gap_rows, fft_size):
    '''
    Gets parameters, calculates threshold, and runs algorithm.

    return:
    list[Transmitter]: detected transmitters
    '''
    # params[0] = mean of the pairwise difference of multiscale products
    # params[1] = std of the pairwise differences of multiscale products
    # beta is a threshold-scaling parameter that determines how many standard deviations from the mean should pairwise differences be in order to be ranked as a outlier local maxima
    params, regions = find_parameters.findAvgAdjDiffCoarse(input, math.log2(input.shape[0]) - scale,
				math.log2(input.shape[0]) - (scale + 1)) #TODO in java, why did he use Util.log2 intead of Math.log 
                                                         #NOTE Util.log2 may not function for arrays?
    
    threshold = params[0] + params[1]*beta # threshold = mean + stdev*beta
    
    detected = multi_scale.findTransmittersMultiScale(input, regions, jaccard_threshold, scale, threshold, max_gap_rows)
    
    return detected # output annotations in main run function
    

@dataclass
class Plugin:
    sample_rate: int = 0
    center_freq: int = 0
    
    # custom params
    # TODO what values to use~no custom values since this is an automatic process. Select and go.
        #NOTE need 1 custom param or the plugin will not show up
    ignore_parameter = 0

    #NOTE what is self doing to the inputted samples?
    def run(self, samples):
        print(samples[0:10])
        print(self.sample_rate)
        print(self.center_freq)
        print(self.beta)
        print(self.scale)

        # Your Plugin (and optionally, classification) code here

        # turn samples into 2d matrix from 1d array
        fft_size = 1024
        num_rows = int(np.floor(len(samples)/fft_size))
        spectrogram = np.zeros((num_rows, fft_size))
        for i in range(num_rows):
            spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples[i*fft_size:(i+1)*fft_size])))**2)

        # print(spectrogram.shape)
        time_for_fft = fft_size * (1/sample_rate) *1000 # time it takes to traverse in ms
        max_gap_rows = math.ceil(0.0/time_for_fft) # TODO max_gap_milliseconds? always 0 in java code
        jaccard_threshold = 0.5 # if they are at least halfway overlapping, considered aligned

        detected = findTransmitters(spectrogram, self.scale, self.beta, jaccard_threshold, max_gap_rows, fft_size)

        # When making a detector, for the return, make a list, then for each detected emission, add one of these dicts to the list:
        # TODO
        #new Transmitter(r, r, curr[0].col, curr[1].col)
        #.col is a list columes we have identified a transmitter is at
        #r is row index
        annotations = []
        for detection in detected:
            #does each edge coincide with a certain column
            edgeOne, edgeTwo, columnOne, columnTwo = detection
            an = {}
            #I copied a similar from simple detector since I am currently confused about the jobs of each core, looking into this tomorrow
            #This should not give an accurate annotation
            an['core:freq_lower_edge'] = int(edgeOne / fft_size * self.sample_rate - (self.sample_rate / 2) + self.center_freq) # Hz, was 1
            an['core:freq_upper_edge'] = int((edgeOne + columnOne) / fft_size * self.sample_rate - (self.sample_rate / 2) + self.center_freq) # Hz, was 2
            an['core:sample_start'] = int(edgeTwo * fft_size)#was 3
            an['core:sample_count'] = int(columnTwo * fft_size)#was 4
            an["core:label"] = "Unknown"#NOTE should we should set this to transmitters, since that is what AirView looks for?
            annotations.append(an)

        return {
            "data_output" : [],
            "annotations" : annotations
        }

if __name__ == "__main__":
    # Example of how to test your plugin locally
    #NOTE fname changes dependent on user, as it is locally testing your plugins, set equal to where you are storing your file pairs(data & meta)
    fname = "/Users/abhome/IQEngineCapstone/DataToAnalyze"
    with open(fname + '.sigmf-meta', 'r') as f:
        meta_data = json.load(f)
    sample_rate = meta_data["global"]["core:sample_rate"]
    center_freq = meta_data["captures"][0]['core:frequency']
    samples = np.fromfile(fname + '.sigmf-data', dtype=np.complex64)
    params = {'sample_rate': sample_rate, 'center_freq': center_freq, 'param1': 1, 'param2': 'test2', 'param3': 5.67}
    plugin = Plugin(**params)
    annotations = plugin.run(samples)
    print(annotations)