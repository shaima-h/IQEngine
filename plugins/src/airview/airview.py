''' TODO
- we are not using training data bc it is the same as input?
- no filef and output_file
- function descriptions
'''

# Copyright (c) 2023 Marc Lichtman.
# Licensed under the MIT License.

import numpy as np
import json
from pydantic.dataclasses import dataclass
import math

def findSumabsSumsqN_row(input, scale1, scale2):
    '''
    Calculates the sum of absolute values, sum of squares, and the size of one region.
    
    return:
    list[float]: sum of absolute values, sum of squares, and size.
    '''
    sumabs_sumsq_n = [None, None, None]
    # get regions with constant power - multiscale transform
    # regions is list of lists
    # TODO
    regions = MultiScaleDetection.getDefaultRegions_training(input, scale1, scale2)

    for i in range(1, len(regions)):
        sumabs_sumsq_n[0] += abs(regions.get[i - 1][4] - regions.get[i][4])
        sumabs_sumsq_n[1] += (regions.get[i - 1][4] - regions.get[i][4]) ** 2
        sumabs_sumsq_n[2] += 1
    
    return sumabs_sumsq_n

def findSumabsSumsqN(input, scale1, scale2):
    '''
    Calculates the sum of absolute values, sum of squares, and size
    
    return:
    list[float]: sum of absolute values, sum of squares, and size
    '''
    # sumabs_sumsq_n[0] = sum of absolute values
    # sumabs_sumsq_n[1] = sum of squares
    # sumabs_sumsq_n = size
    sumabs_sumsq_n = [None, None, None]
    for i in range(input.shape[1]): # iterate through rows
        loc_sumabs_sumsq_n = findSumabsSumsqN_row(input[i], scale1, scale2) # compute the sum of absolute values, sum of squares, and size for each row
        for j in range(len(sumabs_sumsq_n)):
            sumabs_sumsq_n[j] += loc_sumabs_sumsq_n[j] # accumulate results
    return sumabs_sumsq_n


def findAvgAdjDiffCoarse(input, scale1, scale2):
    '''
    Finds the average/standard deviation of adjacent differences in coarse
	signals NOTE: This method just uses the "default" regions (those defined
	by the resolution) to construct the coarse signal
	
	return:
    list[float]: mean and standard deviation of data
    '''
    mean_stdev = [None, None]
    sumabs_sumsq_n = findSumabsSumsqN(input, scale1, scale2)
    mean_stdev[0] = sumabs_sumsq_n[0]/sumabs_sumsq_n[2]
    mean_stdev[1] = math.sqrt(sumabs_sumsq_n[1]/sumabs_sumsq_n[2] - (mean_stdev[0]**2))
    return mean_stdev


def findTransmitters(input, scale, beta, jaccard_threshold, max_gap_rows, fft_size):
    '''
    Gets parametrs, calculates threshold, and runs algorithm.

    return:
    list[Transmitter]: detected transmitters
    '''
    # params[0] = mean of the pairwise difference of multiscale products
    # params[1] = std of the pairwise differences of multiscale products
    # beta is a threshold-scaling parameter that determines how many standard deviations from the mean should pairwise differences be in order to be ranked as a outlier local maxima
    params = findAvgAdjDiffCoarse(input, math.log2(input.shape[0]) - scale,
				math.log2(input.shape[0]) - (scale + 1)) #TODO in java, why did he use Util.log2 intead of Math.log
    
    threshold = params[0] + params[1]*beta # threshold = mean + stdev*beta
    
    # start of algorithm
    # TODO
    detected = MultiScaleDetection.findTransmittersMultiScale(input, jaccard_threshold, scale,
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