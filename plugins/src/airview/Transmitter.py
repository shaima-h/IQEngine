class Transmitter:
    def __init__(self, mean, sd, start_row, start_col, end_row, end_col, found=False, active=False, priors=None):
        self.mean = mean
        self.sd = sd
        self.start_row = start_row
        self.start_col = start_col
        self.end_row = end_row
        self.end_col = end_col
        self.found = found
        self.active = active
        self.priors = priors if priors is not None else []

        #TODO diff constructors?