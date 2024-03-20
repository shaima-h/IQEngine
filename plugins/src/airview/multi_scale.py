import math

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