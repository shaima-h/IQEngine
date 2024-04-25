import Plot from 'react-plotly.js';
import React, { useEffect, useState } from 'react';
import { template } from '@/utils/plotlyTemplate';
import { useSpectrogramContext } from '../hooks/use-spectrogram-context';
import { convertFloat32ArrayToBase64, convertBase64ToFloat32Array } from '@/utils/rf-functions';

interface IQPlotProps {
  multipleIQ: Float32Array[];
}

export const ThreeDimPlot = ({ multipleIQ }: IQPlotProps) => {
  const { spectrogramWidth, spectrogramHeight } = useSpectrogramContext();
  const [I, setI] = useState<Float32Array>();
  const [Q, setQ] = useState<Float32Array>();

  const [plotImageData, setPlotImageData] = useState('');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const timeout = setTimeout(() => {
      setIsLoading(false);
    }, 30000); // 30 seconds timeout

    return () => clearTimeout(timeout);
  }, []);

  useEffect(() => {
    // Set isLoading to false when plotImageData is available
    if (plotImageData) {
      setIsLoading(false);
    }
  }, [plotImageData]);

  useEffect(() => {
    let body = {
      samples_b64: [],
    };

    // const newSamps1 = convertFloat32ArrayToBase64(multipleIQ);
    // console.log(newSamps1);

    // for (const iqData of multipleIQ) {
    //   const newSamps = convertFloat32ArrayToBase64(iqData);
    //   body.samples_b64.push({
    //     samples: newSamps,
    //   });
    // }

    body = {
      samples_b64: multipleIQ.map((iqData) => ({
        samples: convertFloat32ArrayToBase64(iqData),
      })),
    };

    // body = {
    //   samples_b64: [
    //     {
    //       samples: newSamps1,
    //     },
    //   ],
    // };
    console.log(body);

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
        console.log(data);
        setPlotImageData(data.plot_image_base64);
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
