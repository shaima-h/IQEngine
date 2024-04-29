
import numpy as np
import json
from pydantic.dataclasses import dataclass
import math

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
        
        
        PF, PT = compute_maxholds(p)
        thresh = learnthr(PT, S)
        mask = wavemask(PT , c, τ) 
        mask = wavemask(PF , c, τ) 
            


#compute the time and frequency maxholds of the input data
def compute_maxholds(p):
    #input is the spectogram 1D with a whole bunch of rows we iterate through
    for i, row in enumerate(input): # iterate through rows

        
    # p is the matrix of PSD over time and frequencies (p(t, f))
    # Compute PF = maxf∈F (p(t, f)) for each frequency f
    PF = np.max(p, axis=1) 
    
    # Compute PT = maxt∈T (p(t, f)) for each time t         
    PT = np.max(p, axis=0)
    
    return PF, PT

# p is input matrix of PSD over time and frequencies
# p = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) # Example input matrix
# PF, PT = compute_maxholds(p)
# print("PF:", PF)
# print("PT:", PT)


#Adaptive Threshold selection (Algorithm 3)
def learnthr(PT, S):
    prev_thresh = 0
    
    for i in range(1, S + 1):
        s = 2 ** i  #split should be power of 2 (increasing number of equal sized slices?)
        
        # Split PT recursively into s equal-sized slices
        PT_slices = np.array_split(PT, s)
        
        thresh_candidates = []
        
        for j in range(s):
            # Compute wavelet decomposition
            coeffs = wavelet_decomposition(PT_slices[j]) 
            #send to getvalues function and the get values function sends to reconstruct 1D
            #so getvalues function takes in t, scale and send to rreconstruct 1d this is all wavelet decomp 
            #check wavelet decmp algorithm code
            
            lossy_reconstruct = lossy_reconstruction(coeffs) #???
            
            # Compute multiscale product of lossy reconstructions
            mult_product = multiscale_product(lossy_reconstruct)
            
            #absolute pairwise Difference
            pairwise_differences = np.abs(np.diff(mult_product))

            # Compute threshold for the slice
            thresh_candidates = (max(pairwise_differences) - min(pairwise_differences))/2
            
            thresh_candidates.append(thresh_candidates)
        
        # Find the maximum of the candidate thresholds
        thresh = max(thresh_candidates)
        
        if i > 1 and abs(thresh - prev_thresh) <= epsilon:
            break
        
        prev_thresh = thresh
    
    return thresh

# PT = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])  # Example input temporal maxhold
# S = 4  # Number of recursive splits
# epsilon = 0.01  # Tolerance for convergence
# tau = adaptive_threshold_selection(PT, S, epsilon)
# print("Transmitter detection threshold τ:", tau)



#wavelet based masking (Algorithm 2)
def wavemask(P, chunk_size, threshold):
    '''
    num_chunks = math.ceil(len(P) / chunk_size)

    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = min((i + 1) * chunk_size, len(P))
        chunk = P[start_index:end_index]
    
    '''
        
    detection_mask = np.zeros(len(P))
    
    # T is the total length of the data 
    T = len(P)   #T is the lenght of the data???????????????????

    # Loop over chunks of size c
    for i in range(1, T, chunk_size):
        # Process each chunk starting from index i
        chunk_start = i
        chunk_end = min(i + chunk_size, T)  # Ensure the chunk doesn't exceed the length of the data
        chunk = P[chunk_start:chunk_end]
        

        # Step 2: Compute Wavelet Decomposition and lossy reconstruct at levels l and l-1  ?????
        #level l and l-1  ??????
        wavelet_decomposition = wavelet_decomposition(chunk)
        
        lossy_reconstruction = lossy_reconstruction()

        # Step 3: Compute Multiscale Product
        multiscale_product = multiscale_product(wavelet_decomposition) #just multiply the two

        # Step 4: Compute Pairwise Differences
        pairwise_differences = np.abs(np.diff(multiscale_product))

        # Step 5: Threshold Comparison
        for j in pairwise_differences:    
            if j > threshold:
                # Step 6: Produce Detection Mask for Chunk
                detection_mask[chunk_start:chunk_end - 1] = j.astype(int)

    return detection_mask

# P = np.random.rand(100)  # Example maxhold of PSD


    







if __name__ == "__main__":
    # Example of how to test your plugin locally
    #set fname to where you are storing your file pairs (data & meta)
    fname = "/Users/shaimahussaini/classes/icsi499/file_pairs/synthetic"
    with open(fname + '.sigmf-meta', 'r') as f:
        meta_data = json.load(f)
    sample_rate = meta_data["global"]["core:sample_rate"]
    center_freq = meta_data["captures"][0]['core:frequency']
    samples = np.fromfile(fname + '.sigmf-data', dtype=np.complex64)
    params = {'sample_rate': sample_rate, 'center_freq': center_freq}
    plugin = Plugin(**params)
    annotations = plugin.run(samples)
    print(annotations)