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
import copy

# def findTransmittersMultiScale():

def multiscale_transform(input, scale1, scale2):
    '''
    Multiscale transform. Takes an input vector and two scales. Transforms
	the signal and builds a coefficient tree. Gets the relevant indices for
	each input level and reconstructs the signal twice, once using each set
	of indices. Finally, the element-wise product is taken between the two
	reconstructions and this is the multiscale transform.

    returns:

    '''
    # coefficientTree = new CoefficientTree TODO  
        #  ~ maybe we can skip the CoefficientTree and just use the array 
        #    implementation/storage methods below in wavelet_decomposition
        #    and wavelet_decomposition_1D ?


'''
Calculate the wavelet transform of the entire input matrix using the 1D (single-row) 
method on each row of the input matrix.
Input:  an m x n matrix of PSD values from the trace.
Output:  an m x log_2(n-1)-1.  Each inner-array (A) is the coefficients after wavelet
decomposition at every level for a single row in the input PSD trace.
Visual of the decompositions output:
dimensions: 1 x m
Visual:   (a matrix)  (each row is of length 2^(log_2(n)) - 1)

[   [All coefficients for input row 0], 
    [All coefficients for input row 1], 
                ...
    [All coefficients for input row m]  ]

'''
def wavelet_decomposition(input):
    num_rows = input.shape[0]
    num_columns = input.shape[1]
    # Total number of coefficients (i.e. averages) we need to store
    # I got this by doing:
    #    depth = math.log2(num_columns)
    #    num_coeffs = 2**(depth) - 1, since this is closed form of 2^0 + 2^1 + 2^2 + ... + 2^(n-1),
    # which just simplifies to num_coeffs = num_columns - 1
    num_coeffs = num_columns - 1  
    coefficients = np.empty(num_coeffs)
    output_matrix = np.empty([num_rows, num_coeffs])
    for i, row in enumerate(input):
        wavelet_decomposition_1D(row, coefficients, 0, 0)
        output_matrix[i] = coefficients
    return output_matrix


'''
Transform the input PSD matrix into its wavelet decomposition using the
1-D Haar wavelet decomposition as seen in the paper. Calculates the
coefficients in the same way as the paper/in the WV1D.java.transform1D code. 
Inputs:
    input_row:    an array of length n with n being a power of 2
    coefficients: storage for the wavelet coefficients (averages) calculated
                  at each level
    next_index:   the index at which we will start inserting values into the 
                  coefficients array during this function call
    level:        the current level at which we are decomposing
Output: nothing. Instead, the function manipulates the coefficients array in place.  

Details: for the final coefficients array, the last value is the average of the 
entire signal (level 0), the two values before that are the left and right averages 
(level 1),  ...  , the first 2^(log_2(n-1)) values are the "finest/most detailed" averages
(level log_2(n-1)). 
i.e., the coefficients array will store coefficients (averages) in decreasing resolution 
as you move toward the end of the array.
'''
def wavelet_decomposition_1D(input_row, coefficients, next_index, level):
    # print("LEVEL ", level, " coefficients: " , coefficients, " next index ", next_index)
    t = copy.deepcopy(input_row)  # copy original input row
    # print("LEVEL: ", level, " t at start: " , t, " coefficients at start ", coefficients)
    avg = 0
    diff = 0 
    tIndex = 0 
    endpoint = 0
    endpoint = len(input_row) // 2**(level) # where we stop manipulating the decomp vector t
    # do pairwise differencing and averaging from index 0 to the calculated endpoint
    # Loop for every pair of values up to the endpoint. Find pairwise average
    # and difference for each pair (as in paper).
    j = 0
    for i in range(0, endpoint, 2):
        # compute the average of the adjacent cells
        avg = (input_row[i] + input_row[i + 1]) / 2
        # print("avg: " , avg, " after adding values ", input_row[i], input_row[i+1])
        # compute the difference between the average and the second cell
        diff = avg - input_row[i + 1]
        # store the averages in the first half of the row in t and in the coefficients array
        t[tIndex] = avg
        coefficients[next_index+j] = avg
        # print("t after loop: ",t," coefficients after loop:",coefficients)
        # store the differences in the second half
        # I think we do end up not using the differences?
        t[tIndex + endpoint // 2] = diff
        tIndex += 1
        j += 1
    # Recurse on first half of the array by increasing the level parameter
    # next_index is where we will start inserting values into the coefficients array during 
    # the next recursive function call
    # print("DONE. t at end: ", t, " coefficients at end: ", coefficients)
    if endpoint != 2:
        level += 1
        next_index = next_index+j
        wavelet_decomposition_1D(t, coefficients, next_index, level)


def getRegionMeans(regions, input):
    '''
    Get mean in input for each region (bins with same values)
    
    return:

    '''
    # for each region
    for region in regions:
        # get start and end column
        start = region[0]
        end = region[1]
        # calculate mean
        mean = 0.0
        for j in range(start, end+1):
            # TODO: line 845 in MultiScaleDetection.java '= +', doesn't actually accumulate, but results same if I change it to += or keep it as = +, is this function even used??
            mean += input[j]

        if end - start != 0:
            mean /= end - start

        # calculate sd
        sd = 0.0
        for k in range (start, end+1):
            sd += (mean-input[k])**2
        if end - start != 0:
            sd = math.sqrt(sd/(end-start))
        else:
            sd = math.sqrt(sd)
        
        # store
        region[2] = mean
        region[3] = sd
    
    return regions

def multiscale_detection_getDefaultRegions(input, scale1, scale2):
    '''
    Get the "default" regions, meaning only the regions of constant power
	without any filtering or thresholding.

    return:
    list[float]: DESC HERE
    '''
    '''
    Multiscale transform. Takes an input vector and two scales. Transforms
	the signal and builds a coefficient tree. Gets the relevant indices for
	each input level and reconstructs the signal twice, once using each set
	of indices. Finally, the element-wise product is taken between the two
	reconstructions and this is the multiscale transform.
    '''
    transformed = multiscale_transform(input, scale1, scale2)

    # compute the regions
    regions = []
    previous_value = transformed[0]
    start = 0
    end = 0
    for i, t in enumerate(transformed, start=1):
        # if the value is remaining constant, increase the end index
        if t == previous_value and i != len(transformed)-1:
            end += 1
        
        # add the region
        regions.append([start, end, 0.0, 0.0, previous_value])
        end += 1
        start = end
        previous_value = t
    
    # calculate the mean/variance of each region
    regions = getRegionMeans(regions, input)
    return regions

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
    regions = multiscale_detection_getDefaultRegions(input, scale1, scale2)

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
    for i, row in enumerate(input): # iterate through rows
        loc_sumabs_sumsq_n = findSumabsSumsqN_row(row, scale1, scale2) # compute the sum of absolute values, sum of squares, and size for each row
        for j, s in enumerate(sumabs_sumsq_n):
            s += loc_sumabs_sumsq_n[j] # accumulate results
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
    detected = findTransmittersMultiScale(input, jaccard_threshold, scale,
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