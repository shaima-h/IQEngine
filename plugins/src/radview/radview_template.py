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
    computeMaxhold() #1st step of the algorithm
    


class Plugin:
    def run:
        

def computeMaxhold(): #1st step of algorithm is to take maxholds
    #input is the spectogram 1D 
    #will iterate through each of the rows
    #compute the time and frequency maxholds of the input data
    #returns pT and pF
    # Will use pT (maxhold over the time) and send to adaptive threshold selection function (Algorithm 3)
    

def learnthr(Pt): #2nd step (Algorithm 3)
    #Input: Temporal maxhold PT , number of recursive splits S
    #Output: Transmitter detection threshold τ
    for i in 
    
    
    WaveletDecomposition()
    Multiscale()
    computeMaxhold(Tj)
    '''
    for loop through s
        Wavelet Decomposition of pF and pT
        Multiscale Product of Lossy Reconstruction πPj
        Compute ∆πPj - absolute pairwise difference between consecutive values 
        Compute τj
        maxhold(Tj)
    
    '''
def WaveMask(): #(Algorithm 2)
    for loop through c and T iterate 
        WaveletDecomposition() and lossyy recosntruction at a scale of l to l-1
        multiscale product()
        pairwise differnce()
        detectionMask()
        
    
