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

class SamplesB64(BaseModel):
    samples: str

class SpectrogramData(BaseModel):
    samples_b64: Optional[list[SamplesB64]] = None

# @dataclass
# class SpectrogramData:
def generate_plot(samples):
    # TODO
    # we have to pass in spectogram data
    x = [1, 2, 3, 4, 5]
    y = [2, 3, 5, 7, 11]

    plt.plot(samples)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Example Plot')

    # tmpfile = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    # plt.savefig(tmpfile.name, bbox_inches='tight')
    
    # return tmpfile.name

    # with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
    #     plt.savefig(tmpfile.name, bbox_inches='tight')
    #     return tmpfile.name

    # this has to be somewhere accessible here, not localhost
    # plot_file_path = '../../client/src/data/three-dim-plot-figure.png'
    # plt.savefig(plot_file_path)
    # return {'plot_file_path': plot_file_path}

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
    # TODO pass in spectogram data in json, like with plugins
    # then pass to generate_plot()
    samples = np.frombuffer(base64.decodebytes(spectograms.samples_b64[0].samples.encode()), dtype=np.complex64)
    print(samples)
    return generate_plot(samples)
