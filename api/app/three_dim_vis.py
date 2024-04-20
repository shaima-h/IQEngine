import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#helpful for our 3D panel
import tempfile
from pydantic.dataclasses import dataclass
from fastapi import APIRouter
import base64
from typing import List, Optional
from pydantic import BaseModel
from io import BytesIO
import json

class SamplesB64(BaseModel):
    samples: str

class SpectrogramData(BaseModel):
    samples_b64: Optional[list[SamplesB64]] = None

# @dataclass
# class SpectrogramData:
def generate_plot(samples):
    # TODO have to pass in mutiple spectogram data

    # plt.plot(samples)
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Example Plot')

    fft_size = 1024
    num_rows = int(np.floor(len(samples)/fft_size))
    spectrogram = np.zeros((num_rows, fft_size))
    for i in range(num_rows):
        spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples[i*fft_size:(i+1)*fft_size])))**2)

    fig = plt.figure()
    axis = fig.add_subplot(111, projection='3d')

    # create grid for time and frequency axes
    t = np.arange(num_rows)
    f = np.arange(fft_size)
    T, F = np.meshgrid(t, f)

    # plot the spectrogram
    axis.plot_surface(T, F, spectrogram.T, cmap='viridis')

    axis.set_xlabel('Time')
    axis.set_ylabel('Frequency')
    axis.set_zlabel('Intensity (dB)')

    # plt.show()

    # save the plot to a BytesIO buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # convert the image to base64
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return {'plot_image_base64': base64_image}

router = APIRouter()

@router.post('/api/three-dim-plot')
async def get_plot(spectograms: SpectrogramData):
    # TODO pass in multiple spectogram data
    samples = np.frombuffer(base64.decodebytes(spectograms.samples_b64[0].samples.encode()), dtype=np.complex64)
    print(samples)
    return generate_plot(samples)


if __name__ == "__main__":
    # Example of how to test your detector locally
    fname = "/Users/shaimahussaini/classes/icsi499/file_pairs/karyns_sample" # base name
    samples = np.fromfile(fname + '.sigmf-data', dtype=np.complex64)
    generate_plot(samples)
