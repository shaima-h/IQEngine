import numpy as np

class Plugin:
    def inputs(num_buckets, num_samples):
        num_buckets = num_buckets  # Number of frequency buckets
        num_samples = num_samples  # Number of time samples
        maxhold_time = np.zeros(num_samples)  # Array to store maxhold over time
        self.maxhold_freq = np.zeros(num_buckets)  # Array to store maxhold over frequency

    def update_maxhold(self, fft_values):
        # Update maxhold over time
        self.maxhold_time = np.maximum(self.maxhold_time, fft_values)
        
        # Update maxhold over frequency
        self.maxhold_freq = np.maximum(self.maxhold_freq, np.max(fft_values, axis=1))

    def reset_maxhold(self):
        # Reset maxhold arrays
        self.maxhold_time = np.zeros(self.num_samples)
        self.maxhold_freq = np.zeros(self.num_buckets)

# Test herre
num_buckets = 1024  # Number of frequency buckets
num_samples = 1000  # Number of time samples
sa = Plugin(num_buckets, num_samples)

for i in range(num_samples):
    fft_values = np.random.rand(num_buckets)  #Random values representing FFT buckets
    sa.update_maxhold(fft_values)

#maxhold values
maxhold_time = sa.maxhold_time
maxhold_freq = sa.maxhold_freq
