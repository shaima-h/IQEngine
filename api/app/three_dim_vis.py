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
import kaleido
import plotly.graph_objects as go
import plotly.io as pio

class SamplesB64(BaseModel):
    samples: str

class MultipleSamples(BaseModel):
    samples_b64: List[SamplesB64]

class SpectrogramData(BaseModel):
    samples_b64: Optional[list[SamplesB64]] = None

# @dataclass
# class SpectrogramData:
def generate_plot(samples_list):

    fft_size = 1024
    # num_rows = int(np.floor(len(samples_list[0])/fft_size))
    num_rows = 650

    data = []
    z_offset = 0
    for samples in samples_list:
        spectrogram = np.zeros((num_rows, fft_size))
        for i in range(num_rows):
            spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples[i*fft_size:(i+1)*fft_size])))**2)
    
        print(spectrogram.shape)
        data.append(go.Surface(z=spectrogram + z_offset, opacity=0.8, showscale=False, colorscale='viridis'))
        z_offset += 350  # Increase the z-offset to stack spectrograms vertically
        
    layout = go.Layout(
        scene=dict(
            zaxis=dict(title='Intensity (dB)'),
            xaxis=dict(title='Time'),
            yaxis=dict(title='Frequency'),
            bgcolor='#05041C',
            xaxis_title_font=dict(color='white'),
            yaxis_title_font=dict(color='white'),
            zaxis_title_font=dict(color='white'),
            xaxis_tickfont=dict(color='white'),
            yaxis_tickfont=dict(color='white'),
            zaxis_tickfont=dict(color='white'),
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )

    fig = go.Figure(data=data, layout=layout)
    fig.layout.template = None
    plot_json = pio.to_json(fig, pretty=True)
    fig.show() # show figure in a separate tab on localhost
    # return plot_json # for some reason returning as json causes no data to be plotted, even though frontend receives json perfectly

    buffer = BytesIO()
    fig_png = fig.to_image(format="png") # kaleido library
    buffer.write(fig_png)
    buffer.seek(0) 

    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return {'plot_image_base64': base64_image} # display image on IQEngine


    ''' MATPLOTLIB IMAGE '''

    # fig = plt.figure()
    # axis = fig.add_subplot(111, projection='3d')

    # fft_size = 1024
    # num_rows = int(np.floor(len(samples_list[0])/fft_size))

    # # create grid for time and frequency axes
    # t = np.arange(num_rows)
    # f = np.arange(fft_size)
    # T, F = np.meshgrid(t, f)

    # z_offset = 0
    # for samples in samples_list:
    #     spectrogram = np.zeros((num_rows, fft_size))
    #     for i in range(num_rows):
    #         spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples[i*fft_size:(i+1)*fft_size])))**2)
    
    #     # plot the spectrogram
    #     axis.plot_surface(T, F, spectrogram.T + z_offset, cmap='viridis')
    #     z_offset += 300  # Increase the z-offset to stack spectrograms vertically

    # axis.set_xlabel('Time')
    # axis.set_ylabel('Frequency')
    # axis.set_zlabel('Intensity (dB)')

    # # plt.show()

    # # save the plot to a BytesIO buffer
    # buffer = BytesIO()
    # plt.savefig(buffer, format='png')
    # buffer.seek(0)

    # # convert the image to base64
    # base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # return {'plot_image_base64': base64_image}

router = APIRouter()

@router.post('/api/three-dim-plot')
async def get_plot(spectograms: SpectrogramData):
    print(SpectrogramData)
    # TODO pass in multiple spectogram data
    samples_list = []
    # iterate through samples in samples_b64
    for samples_obj in spectograms.samples_b64:
        samples = np.frombuffer(base64.decodebytes(samples_obj.samples.encode()), dtype=np.complex64)
        print(samples)
        samples_list.append(samples)

    return generate_plot(samples_list)


if __name__ == "__main__":
    # Example of how to test your detector locally
    fname = "/Users/shaimahussaini/classes/icsi499/file_pairs/USRP_S1_K1_573000000.0_2024_04_05-01_05_47_PM" # base name
    samples1 = np.fromfile(fname + '.sigmf-data', dtype=np.complex64)
    
    fname = "/Users/shaimahussaini/classes/icsi499/file_pairs/USRP_S1_K1_573000000.0_2024_04_05-01_08_31_PM" # base name
    samples2 = np.fromfile(fname + '.sigmf-data', dtype=np.complex64)
    
    fname = "/Users/shaimahussaini/classes/icsi499/file_pairs/USRP_S1_K1_573000000.0_2024_04_05-01_08_49_PM" # base name
    samples3 = np.fromfile(fname + '.sigmf-data', dtype=np.complex64)
    
    fname = "/Users/shaimahussaini/classes/icsi499/file_pairs/USRP_S1_K1_573000000.0_2024_04_05-01_09_05_PM" # base name
    samples4 = np.fromfile(fname + '.sigmf-data', dtype=np.complex64)
    
    fname = "/Users/shaimahussaini/classes/icsi499/file_pairs/USRP_S1_K1_573000000.0_2024_04_05-01_09_40_PM" # base name
    samples5 = np.fromfile(fname + '.sigmf-data', dtype=np.complex64)
    
    samples_list = [samples1, samples2, samples3, samples4, samples5]
    generate_plot(samples_list)
