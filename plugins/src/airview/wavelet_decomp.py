import numpy as np
import copy

def wavelet_decomposition(input):
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

def wavelet_decomposition_1D(input_row, coefficients, next_index, level):
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