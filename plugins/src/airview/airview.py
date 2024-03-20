''' TODO Notes
- confirm: we are not using training data bc it is the same as input?
- no filef and output_file (from FindTransmitters.java)
- fix and make proper function descriptions/comments
- MultiScaleDetection class? do we need classes if they just have methods and don't have any attributes in the java code?
    - functions look so messy, how do we organize?
    - multiple python files in airview folder? then import in this file? like:
    airview/
    ├── airview.py <- only has findTransmitters()
    ├── MultiScale.py
    ├── CoefficientTree.py
    ├── FindParameters.py
    └── ...
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
    params = find_parameters.findAvgAdjDiffCoarse(input, math.log2(input.shape[0]) - scale,
				math.log2(input.shape[0]) - (scale + 1)) #TODO in java, why did he use Util.log2 intead of Math.log
    
    threshold = params[0] + params[1]*beta # threshold = mean + stdev*beta
    
    # start of algorithm
    # TODO
    detected = multi_scale.findTransmittersMultiScale(input, jaccard_threshold, scale,
					params[0] + params[1] * beta, max_gap_rows)
    
    return detected # output annotations in main run function
    

@dataclass
class Plugin:
    sample_rate: int = 0
    center_freq: int = 0
    
    # custom params
    # TODO what values to use??
    beta: float = 1.0
    scale: int = 1

    def run(self, samples):
        print(samples[0:10])
        print(self.sample_rate)
        print(self.center_freq)
        print(self.param1)
        print(self.param2)
        print(self.param3)

        # TODO max_gap_milliseconds? always 0 in java code

        # Your Plugin (and optionally, classification) code here
        fft_size = samples.shape[1] # number of columns #TODO make sure it is power of 2?
        time_for_fft = fft_size * (1/sample_rate) *1000 # time it takes to traverse in ms
        max_gap_rows = math.ceil(0.0/time_for_fft) #TODO
        jaccard_threshold = 0.5 # if they are at least halfway overlapping, considered aligned

        detected = findTransmitters(samples, self.scale, self.beta, jaccard_threshold, max_gap_rows, fft_size)

        # When making a detector, for the return, make a list, then for each detected emission, add one of these dicts to the list:   
        # TODO
        annotations = []
        for detection in detected:
            an = {}
            an['core:freq_lower_edge'] = 1 # Hz
            an['core:freq_upper_edge'] = 2 # Hz
            an['core:sample_start'] = 3
            an['core:sample_count'] = 4
            an["core:label"] = "Unknown"
            annotations.append(an)

        return {
            "data_output" : [],
            "annotations" : annotations
        }

if __name__ == "__main__":
    # Example of how to test your plugin locally
    fname = "C:\\Users\\marclichtman\\Downloads\\synthetic"
    with open(fname + '.sigmf-meta', 'r') as f:
        meta_data = json.load(f)
    sample_rate = meta_data["global"]["core:sample_rate"]
    center_freq = meta_data["captures"][0]['core:frequency']
    samples = np.fromfile(fname + '.sigmf-data', dtype=np.complex64)
    params = {'sample_rate': sample_rate, 'center_freq': center_freq, 'param1': 1, 'param2': 'test2', 'param3': 5.67}
    plugin = Plugin(**params)
    annotations = plugin.run(samples)
    print(annotations)