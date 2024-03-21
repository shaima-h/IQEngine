def colIntersection(actual, detected):
    col_int = 0.0
    # if the columns don't intersect at all
    if actual.end_col < detected.start_col or detected.end_col < actual.start_col:
        return col_int
    elif actual.start_col >= detected.start_col and actual.end_col <= detected.end_col:
        # the actual transmitter is contained within the detected
        col_int = actual.end_col - actual.start_col
    elif detected.start_col >= actual.start_col and detected.end_col <= actual.end_col:
        # the detected transmitter is contained within the actual
        col_int = detected.end_col - detected.start_col
    elif actual.start_col >= detected.start_col and actual.end_col >= detected.end_col:
        # the actual transmitter starts after the detected, but isn't contained within it
        col_int = detected.end_col - actual.start_col
    elif detected.start_col >= actual.start_col and detected.end_col >= actual.end_col:
        # the detected transmitter starts after the detected, but isn't contained within it
        col_int = actual.end_col - detected.start_col
    elif actual.start_col <= detected.start_col and actual.end_col <= detected.end_col:
        # the actual transmitter ends before the detected, but also starts before it
        col_int = actual.end_col - detected.start_col
    elif detected.start_col <= actual.start_col and detected.end_col <= actual.end_col:
        # the detected transmitter ends before the actual transmitter, but also starts before it
        col_int = detected.end_col - actual.start_col
    
    return col_int

def jaccard_value(t, a):
    jaccard = 0.0
    intersection = colIntersection(t, Transmitter(1, 1, a[0][0], a[1][0]))
    union = (t.end_col - t.start_col) + (a[1][0] - a[0][0])
    jaccard = intersection / (union - intersection)
    return jaccard

def updateTransmitters(changes, t, r, jaccard_threshold, max_gap):
    # if the transmitter list is empty, just add all edges as transmitters
    if not t and changes[r]:
        for e in changes[r]:
            t.append(Transmitter(r, r, e[0][0], e[1][0]))
            t[len(t)-1].active_switch()
    else:
        # otherwise, compare with previous rows
	    # what has changed since the last row?
        # loop through current edges
        for curr in changes[r]:
            # initialize found to false for this edge array
            found = False
            # loop through transmitters, starting with most recent (to catch active transmitters)
            for i, tx in reversed(list(enumerate(t))):
                # if the transmitter + edges match within jaccard threshold
                if jaccard_value(tx, curr) >= jaccard_threshold:
                    found = True # we've matched the edge[] with a transmitter
                    # if the transmitter is currently active
                    if tx.active:
                        tx.set_row_fall(r) # set the latest row fall to the current row
                        tx.found = True
                    else:
                        # if the transmitter has already been deemed inactive, restart it
                        if r - tx.end_row <= max_gap:
                            # if it's been inactive for less than the max gap just restart it
                            tx.active_switch()
                            tx.set_row_fall(r)
                            tx.found = True
                        else:
                            # if it's been inactive for too long, create a new transmitter
                            t.append[Transmitter(r, r, curr[0][0], curr[1][0])]
                            t[len(t)-1].found = True
                            t[len(t)-1].active_switch()
                    break # no need to look at any more transmitters

                # if the edges weren't found in any previous transmitters
                if not found:
                    # make a new transmitter with current row as start and end
                    t.append(Transmitter(r, r, curr[0][0], curr[1][0]))
                    t[len(t)-1].active_switch() # list this transmitter as active
                    t[len(t)-1].found = True # set it to found so we don't deactivate it immediately below

            # loop through the transmitters again
            for tx in t:
                # only look at currently active transmitters that were not found
                if tx.active and not tx.found: # transmitter is no longer active
                    tx.found = False # reset found to false


class Transmitter:
    def __init__(self, start_row, start_col, end_row, end_col, mean=None, sd=None, found=False, active=False, priors=None):
        self.start_row = start_row
        self.start_col = start_col
        self.end_row = end_row
        self.end_col = end_col
        self.mean = mean
        self.sd = sd
        self.found = found
        self.active = active
        self.priors = priors

    def active_switch(self):
        self.active = not self.active

    def set_row_fall(self, end_row):
        self.end_row = end_row