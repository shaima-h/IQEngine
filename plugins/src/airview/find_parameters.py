import multi_scale
import math

def findSumabsSumsqN_row(input, scale1, scale2):
    '''
    Calculates the sum of absolute values, sum of squares, and the size of one region.
    
    return:
    list[float]: sum of absolute values, sum of squares, and size.
    '''
    sumabs_sumsq_n = [0.0, 0.0, 0.0]
    # get regions with constant power - multiscale transform
    # regions is list of lists
    regions = multi_scale.multiscale_detection_getDefaultRegions(input, scale1, scale2)

    for i in range(1, len(regions)):
        sumabs_sumsq_n[0] += abs(regions[i - 1][4] - regions[i][4])
        sumabs_sumsq_n[1] += (regions[i - 1][4] - regions[i][4]) ** 2
        sumabs_sumsq_n[2] += 1
    
    return sumabs_sumsq_n, regions

def findSumabsSumsqN(input, scale1, scale2):
    '''
    Calculates the sum of absolute values, sum of squares, and size
    
    return:
    list[float]: sum of absolute values, sum of squares, and size
    '''
    # sumabs_sumsq_n[0] = sum of absolute values
    # sumabs_sumsq_n[1] = sum of squares
    # sumabs_sumsq_n = size
    sumabs_sumsq_n = [0.0, 0.0, 0.0]
    regions = []
    for i, row in enumerate(input): # iterate through rows
        loc_sumabs_sumsq_n, regions_row = findSumabsSumsqN_row(row, scale1, scale2) # compute the sum of absolute values, sum of squares, and size for each row
        regions.append(regions_row)
        # for j, s in enumerate(sumabs_sumsq_n):
        for j in range(len(sumabs_sumsq_n)):
            sumabs_sumsq_n[j] += loc_sumabs_sumsq_n[j] # accumulate results
    return sumabs_sumsq_n, regions

def findAvgAdjDiffCoarse(input, scale1, scale2):
    '''
    Finds the average/standard deviation of adjacent differences in coarse
	signals NOTE: This method just uses the "default" regions (those defined
	by the resolution) to construct the coarse signal
	
	return:
    list[float]: mean and standard deviation of data
    '''
    mean_stdev = [None, None]
    sumabs_sumsq_n, regions = findSumabsSumsqN(input, scale1, scale2)
    mean_stdev[0] = sumabs_sumsq_n[0]/sumabs_sumsq_n[2]
    mean_stdev[1] = math.sqrt(sumabs_sumsq_n[1]/sumabs_sumsq_n[2] - (mean_stdev[0]**2))
    return mean_stdev, regions

