import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#helpful for our 3D panel
import tempfile
from pydantic.dataclasses import dataclass
from fastapi import APIRouter

router = APIRouter()

def generate_plot():
    # TODO samples
    x = [1, 2, 3, 4, 5]
    y = [2, 3, 5, 7, 11]

    plt.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Example Plot')

    # tmpfile = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    # plt.savefig(tmpfile.name, bbox_inches='tight')
    
    # return tmpfile.name

    # with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
    #     plt.savefig(tmpfile.name, bbox_inches='tight')
    #     return tmpfile.name

    plot_file_path = 'three-dim-plot-figure.png'
    plt.savefig(plot_file_path)
    return plot_file_path
    

@router.get('/api/plot')
async def get_plot():
    plot_file_path = generate_plot()
    return {'plot_file_path': plot_file_path}