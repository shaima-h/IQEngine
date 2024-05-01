import React, { useEffect, useState } from 'react';
import { convertFloat32ArrayToBase64 } from '@/utils/rf-functions';

interface ThreeDimPlotProps {
  displayedIQ: Float32Array;
}

export const ThreeDimPlot = ({ displayedIQ }: ThreeDimPlotProps) => {
  const [plotImageData, setPlotImageData] = useState('');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const timeout = setTimeout(() => {
      setIsLoading(false);
    }, 15000); // 15 seconds timeout

    return () => clearTimeout(timeout);
  }, []);

  useEffect(() => {
    // Set isLoading to false when plotImageData is available
    if (plotImageData) {
      setIsLoading(false);
    }
  }, [plotImageData]);

  // send data to python function
  useEffect(() => {
    let body = {
      samples_b64: [],
    };

    body = {
      samples_b64: [
        {
          samples: convertFloat32ArrayToBase64(displayedIQ),
        },
      ],
    };

    // console.log(body);

    fetch('/api/three-dim-plot', {
      method: 'POST',
      headers: {
        Accept: 'application/json',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    })
      .then((response) => response.json())
      .then((data) => {
        setPlotImageData(data.plot_image_base64);
        // console.log(plotImageData);
      });
  }, []);

  return (
    <div className="plot-container">
      {!isLoading && plotImageData && (
        <img src={`data:image/png;base64,${plotImageData}`} alt="Plot" style={{ width: '100%', height: 'auto' }} />
      )}
      {isLoading && <div>Loading...</div>}
      {!isLoading && !plotImageData && <div>No plot generated</div>}
    </div>
  );
};
