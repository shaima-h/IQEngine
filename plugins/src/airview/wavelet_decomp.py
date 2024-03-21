import math
import numpy as np
import copy

'''
This is an exact copy of the WV1D.transform1D method in the Java repo.

TODO:  add comments
'''
def transform1D(input_row, level):
    t = copy.deepcopy(input_row)
    avg = 0.0
    diff = 0.0
    tIndex = 0 
    endpoint = 0
    if level > 0:
        endpoint = (int) (len(input_row) // (2**(level)))
    else:
        endpoint = len(input_row)
    
    for i in range(0, endpoint, 2):
        avg = (input_row[i] + input_row[i + 1]) / 2
        diff = avg - input_row[i + 1]
        t[tIndex] = avg
        t[(int)(tIndex + endpoint / 2)] = diff
        tIndex += 1
    if endpoint != 2:
        level += 1
        t = transform1D(t, level)
    return t
        

'''
Find the values we need in the reconstruction process. Always keeps the first value
(the whole-signal average), and then finds the detail coefficients for the given 
scale.
This is similar to the work done in CoefficientTree.jabva.getLevelIndices() and 
MultiScale.java.getIndicesAList(), I think.
Inputs:
    t:  the complete wavelet transform
    scale:  the scale to reconstruct at
'''
def get_values(t, scale):
    output = np.zeros(len(t))
    output[0] = t[0]
    start_index = (int) (2**(scale)) # start index for coefficients to use
    end_index = (int) (start_index + 2**(scale))  # end index for coefficients to use
    for i in range(start_index, end_index):
        output[i] = t[i]
    return output


'''
This is an exact copy of the WV1D.reconstruct1D method in the Java repo.

TODO: add comments
'''
def reconstruct1D(input):
    r = np.empty(len(input))
    value = 0.0
    sign = 0
    detailCo = 0.0

    for i in range(len(input)):
        value = input[0]
        
        for k in range(1, (int)((math.log(len(input)) / math.log(2))) + 1):
            sign = (int) ((-1)**(math.floor((i * 2**(k)) / len(input))))
            detailCo = input[(int)(2**(k-1)) 
                             + (int)(math.floor((i * 2**(k-1)) / len(input)))]
            value += sign * detailCo
        
        r[i] = value
    
    return r
