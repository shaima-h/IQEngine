import numpy as np
import math
import wavelet_decomp

def adjacentOrOverlapping(edges, start, end):
    '''
    Check if the indices are directly adjacent to or overlapping with an
	already established region.
    '''
    for e in edges:
        if (e[1][0] + 1 == start or e[0][0] -1 == end or
        (e[0][0] <= start and e[1][0] >= end) or
        (e[1][0] >= start and e[1][0] <= end) or
        (e[0][0] >= start and e[0][0] <= end) or
        (e[0][0] >= start and e[1][0] <= end)):
            return True
    return False

#TODO make sorting more efficient??
def isSorted(regions, indexToSort):
    for i in range(1, len(regions)):
        if regions[i-1][indexToSort] + regions[i-1][indexToSort+1] < regions[i][indexToSort] + regions[i][indexToSort+1]:
            return False
    return True

def sortRegions(regions, indexToSort):
    sorted_regions = regions[:]
    
    while not isSorted(sorted_regions, indexToSort):
        for i in range(1, len(sorted_regions)):
            if sorted[i-1][indexToSort] + sorted_regions[i-1][indexToSort+1] < sorted_regions[i][indexToSort] + sorted_regions[i][indexToSort+1]:
                tmp = sorted_regions[i-1]
                sorted_regions.pop(i-1)
                sorted_regions.insert(i-1, sorted[i-1])
                sorted_regions.pop[i]
                sorted_regions.insert(i, tmp)
    
    return sorted_regions


def stats(input, start, end):
    '''
    Compute mean power stats between start and end indices.
    '''
    mean = 0.0
    sd = 0.0
    for i in range(start, end):
        mean += input[i]
    if end - start > 0:
        mean /= end - start

    for i in range(start, end):
        sd += (mean - input[i])**2
    if end - start > 0:
        sd = math.sqrt(sd/(end - start))
    else:
        sd = math.sqrt(sd)
    
    return [mean, sd]

def threshold(regions, input, alpha):
    edges = []
    for i in range(1, len(regions)):
        # compare the value of the constant power in each region (array index 4)--> multiscale value
		# if the absolute value of their difference is larger than the threshold (alpha)
        if abs(regions[i-1][4] - regions[i][4]) > alpha:
            # add the higher value to the edges list
            # looking for everything that's above threshold
            if regions[i-1][4] > regions[i][4]:
                edges.append(regions[i-1])
            else:
                edges.append(regions[i])
    
    # now we have the filtered coarse representation. Need to get averages for each adjacent region
    list = []
    for i in range(1, len(edges)):
        # start with the widest band (leftmost to right most)
        mean_sd = stats(input, edges[i-1][0], edges[i][1])
        
        # set max to these stats
        max = mean_sd
        d = [edges[i-1][0], edges[i][1], max[0], max[1]]

        # compare stats for other intervals
        # (leftmost to leftmost)
        mean_sd = stats(input, edges[i-1][0], edges[i][0])

        # if mean power is greater, this is a new max
        if mean_sd[0] > max[0] and mean_sd[0] < 0:
            max = mean_sd
            d = [edges[i-1][0], edges[i][0], max[0], max[1]]

        # repeat same process for each interval
        # (rightmost to rightmost)
        mean_sd = stats(input, edges[i-1][1], edges[i][1])
        if mean_sd[0] > max[0] and mean_sd[0] < 0:
            max = mean_sd
            d = [edges[i-1][1], edges[i][1], max[0], max[1]]
        
        # (rightmost to leftmost)
        mean_sd = stats(input, edges[i-1][1], edges[i][0])
        if mean_sd[0] > max[0] and mean_sd[0] < 0:
            max = mean_sd
            d = [edges[i-1][1], edges[i][0], max[0], max[1]]

        # add the max to the list
        list.add(d)

    #TODO write to output.csv
    return list


def coarseDetection(input, regions, scale1, scale2, alpha):
    edges = []
    
    # threshold the coarse signal
    filtered = threshold(regions, input, alpha)

    # TODO index stuff? just have word or don't have at all idk
    # sort the regions (index 2 is by mean, index 0/1 for frequency bins,
    # 3 for standard deviation, 4 is coarse value)
    filtered = sortRegions(filtered, 2)

    # call every other region a transmitter
    for f in filtered:
        if(not adjacentOrOverlapping(edges, f[0], f[1])):
            edges.append([[f[0], 'r'], [f[1], 'f']])

    return edges

def findTransmittersMultiScale(input, regions, jaccard_threshold, scale, alpha, max_gap_rows):
    # list of Transmitters
    transmitters = []

    # integer : list[edge[]]
    changes = {}

    for i, row in enumerate(input):
        curr_edges = None
        #TODO make sure regions is correct
        curr_edges = coarseDetection(row, regions[i], math.log2(len(row) - scale), math.log2(len(row) - (scale +1), alpha))

        # add all of the changes to the map
        changes[row] = curr_edges
        #TODO updateTransmitters

    return transmitters


def multiscale_transform(input, scale1, scale2):
    '''
    Multiscale transform. Takes an input vector and two scales. Transforms
	the signal and builds a coefficient tree. Gets the relevant indices for
	each input level and reconstructs the signal twice, once using each set
	of indices. Finally, the element-wise product is taken between the two
	reconstructions and this is the multiscale transform.

    returns:

    '''
    t = wavelet_decomp.wavelet_decomposition(input)
    #r = wavelet_decomp.reconstruct1D(input)

    if len(t) != len(r):
        raise ValueError("Error: Cannot multiply vectors of unequal length.")
    
    # multiply element wise
    return np.multiply(t, r)


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