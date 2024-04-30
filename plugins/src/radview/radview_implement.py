import numpy as np
import json
import airview
from pydantic.dataclasses import dataclass
import math
from airview import findAvgAdjDiffCoarse
from airview import multiscale_detection_getDefaultRegions
epsilon = 0.01
'''
Code for RadView Research paper. Implemented in code following algorithm
'''


def findTransmitters(input, scale, beta, fft_size):
    '''
    Take maxhold over freq and time, preprocessing, send to airview, masking where detections, and postprocessing

    return:
    list[Transmitter]: detected transmitters
    '''
    PT_maxhold, PF_maxhold = computeMaxHold(input) #Function that computes maxholds
    
    '''
    in place of learnthresh just input it into airview.
    '''
    
    #split temporal vector into chunks before sending it to airview
    #slice into size 32 chunks
    #Preprocessing - taking file and chopping it up into chunks of 32
    if fft_size > 16: 
        extraRows = PT_maxhold % 32
        perfectSizeT = PT_maxhold - extraRows
        #feed each slice of 32 into airview
        chunkOf32T = perfectSizeT/32
        detectionInFreqVec = airview(scale, beta, chunkOf32T) #send frequency vector to airview to capture any detections
    else:
        detectionInFreqVec = airview(scale, beta, PF_maxhold) #send frequency vector to airview to capture any detections

    
    #if the fft size is grerater than 16 in tthat case we have to chunk the frequency as well
    if fft_size > 16:
        extraCol = PF_maxhold % 32
        perfectSizeF = PF_maxhold - extraCol
        #feed each slice of 32 into airview
        chunkOf32F = perfectSizeF/32  
        detectionInTimeVec = airview(scale, beta, chunkOf32F) #send time vector to airview to capture any detections
    else:
        detectionInTimeVec = airview(scale, beta, PT_maxhold) #send time vector to airview to capture any detections

        
    # params[0] = mean of the pairwise difference of multiscale products
    # params[1] = std of the pairwise differences of multiscale products
    # beta is a threshold-scaling parameter that determines how many standard deviations from the mean should pairwise differences be in order to be ranked as a outlier local maxima
    params, regions = findAvgAdjDiffCoarse(input, math.log2(input.shape[0]) - scale,
				math.log2(input.shape[0]) - (scale + 1)) #TODO in java, why did he use Util.log2 intead of Math.log 
                                                         #NOTE Util.log2 may not function for arrays?
    
    threshold = params[0] + params[1]*beta # threshold = mean + stdev*beta
    
    #compute c (chunksize c here before sending to algorithm 2)
    #c is the stepsize
    # c is best computed to be 32 not too low or high
    c = 32
    maskOverTimeVec = waveMask(detectionInTimeVec, c, threshold) #Temporal characterization
    maskOverFreqVec = waveMask(detectionInFreqVec, c, threshold) #Frequency characterization

    #postProcessing for the temporal we have chunks of 32 now we need to append and put it into one big array
    #take time mask vector and frequncy mask vector and take the outer product of that and that is gonna give
    #me time over frequncy mask
    fulldetection = np.outer(maskOverTimeVec, maskOverFreqVec)  #outerproduct
    return fulldetection

@dataclass
class Plugin:
    sample_rate: int = 0
    center_freq: int = 0
    
    # custom params
    # TODO should we calculate optimal beta and scale?
    beta: float = 1.0 #beta would be around 2 for radview 
    scale: int = 3 #USing scale of 2 helps with Radview

    def run(self, samples):
        print(samples[0:10])
        print(self.sample_rate)
        print(self.center_freq)
        print(self.beta)
        print(self.scale)

        # Your Plugin (and optionally, classification) code here

        # turn samples into 2d matrix from 1d array
        fft_size = 1024  #since the fft size is grerater than 16 in tthat case we have to chunk the frequency as well
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


def computeMaxHold(input):
    #take the maxhold in each row and column 
    #input is the spectogram 1D with a whole bunch of rows we iterate through
    # Create an empty list to store individual row arrays
    individual_row_arrays = []

    # Iterate through rows and append each row to the list
    for row in input:
        individual_row_arrays.append(row)
        # p is the matrix of PSD over time and frequencies (p(t, f))
        # Compute PF = maxf∈F (p(t, f)) for each frequency f
        PF = np.max(individual_row_arrays, axis=0)  #takes the max across each row of 2d array
    
    # Transpose the array
    transposed_array = [[row[i] for row in input] for i in range(len(input[0]))]

    # Create an empty list to store individual column arrays
    individual_column_arrays = []

    # Iterate through rows of transposed array and append each row (which represents a column) to the list
    for row in transposed_array:
        individual_column_arrays.append(row)

    # Print individual column arrays
    for column_array in individual_column_arrays:
        # Compute PT = maxt∈T (p(t, f)) for each time t         
        PT = np.max(column_array, axis=1) # takes the max across each column of the 2d array
    return PT, PF


#wavelet based masking (Algorithm 2)
def waveMask(P, chunk_size, threshold, scale):
    '''
    Return a 0-1 mask of transmitter activity
    '''
    
    '''
    num_chunks = math.ceil(len(P) / chunk_size)

    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = min((i + 1) * chunk_size, len(P))
        chunk = P[start_index:end_index]
    
    '''
        
    detection_mask = np.zeros(len(P))
    
    # T is the total length of the data 
    T = len(P)   

    # Loop over chunks of size c
    for i in range(1, T, chunk_size):
        # Process each chunk starting from index i
        chunk_start = i
        chunk_end = min(i + chunk_size, T)  # Ensure the chunk doesn't exceed the length of the data
        chunk = P[chunk_start:chunk_end]
        

        # Step 2: Compute Wavelet Decomposition and lossy reconstruct at levels l and l-1
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
        Includes lossy reconstrruction and multiscale product.
        '''
        
        for r, row in enumerate(input):
            scale1 = math.log2(len(row)) - scale
            scale2 = math.log2(len(row)) - (scale + 1)

        multiscale_product = multiscale_detection_getDefaultRegions(input, scale1, scale2)
        
        # Step 4: Compute Pairwise Differences
        pairwise_differences = np.abs(np.diff(multiscale_product))

        # Step 5: Threshold Comparison
        for j in pairwise_differences:    
            if j > threshold:
                # Step 6: Produce Detection Mask for Chunk
                detection_mask[chunk_start:chunk_end - 1] = 1
            else: 
                detection_mask[chunk_start:chunk_end - 1] = 0

    return detection_mask
     
    

    
'''

#TODO: S is the number of recursive splits (How do we compute that)
    #is it a power of 2? S = 2^i per split!?
    
def learnthr(PT_maxhold, S): #2nd step (Algorithm 3) 
    
    #Input: Temporal maxhold PT , number of recursive splits S
    #Output: Transmitter detection threshold τ
    
    num_slices = 2  # Initial number of slices
    prev_threshold = 0  # threshold

    for i in range(1, S):
        Tj = []  # Store candidate thresholds for each subdivision

        # Divide the maxhold into slices
        slice_size = len(PT_maxhold) // num_slices
        for j in range(num_slices):
            
            Pj = PT_maxhold[j * slice_size : (j + 1) * slice_size]
            wavelet_decomposition = wavelet_decomposition(Pj)
            multiscale_product = multiscale_product(wavelet_decomposition)
            #absolute pairwise difference between consecutive values.
            delta_pi_Pj = np.abs(np.diff(multiscale_product))

            # Compute candidate threshold τj
            threshold_j = (np.max(delta_pi_Pj) - np.min(delta_pi_Pj)) / 2
            Tj.append(threshold_j)

        # Find the maximum threshold among all subdivisions
        max_threshold = max(Tj) #computMaxHold

        # Check for convergence
        if i > 1 and max_threshold - prev_threshold <= epsilon:
            break

        prev_threshold = max_threshold
        num_slices *= 2  # Double the number of slices for the next iteration

    return max_threshold
    


'''
    

        
    