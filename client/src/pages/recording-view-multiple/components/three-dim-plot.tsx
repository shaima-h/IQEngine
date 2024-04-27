import Plot from 'react-plotly.js';
import React, { useEffect, useState } from 'react';
import { template } from '@/utils/plotlyTemplate';
import { useSpectrogramContext } from '../hooks/use-spectrogram-context';
import { convertFloat32ArrayToBase64, convertBase64ToFloat32Array } from '@/utils/rf-functions';

interface IQPlotProps {
  multipleIQ: Float32Array[];
}

interface PlotImageData {
  data: [];
  layout: {};
}

export const ThreeDimPlot = ({ multipleIQ }: IQPlotProps) => {
  const { spectrogramWidth, spectrogramHeight, fftSize } = useSpectrogramContext();
  // const [I, setI] = useState<Float32Array>();
  // const [Q, setQ] = useState<Float32Array>();

  // const [plotImageData, setPlotImageData] = useState<PlotImageData>({ data: [], layout: {} });
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

  useEffect(() => {
    let body = {
      samples_b64: [],
    };

    body = {
      samples_b64: multipleIQ.map((iqData) => ({
        samples: convertFloat32ArrayToBase64(iqData),
      })),
    };

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

  // json interactive plotly plot - data is not plotted for some reason even though json is received perfectly
  // return (
  //   <div className="plot-container">
  //     {!isLoading && plotImageData && <Plot data={plotImageData.data} layout={plotImageData.layout} />}
  //     {isLoading && <div>Loading...</div>}
  //     {!isLoading && !plotImageData && <div>No plot generated</div>}
  //   </div>
  // );

  //******** INTERACTIVE REACT PLOTLY PLOT ********** */
  // very slow and can only display a portion of the data, colorscale doesn't work either

  // function splitSpectrogram(data) {
  //   // const numRows = data.length / fftSize;
  //   console.log('splitting', data);
  //   const numRows = spectrogramHeight;
  //   const fft = spectrogramHeight;
  //   const spectrogram = [];

  //   for (let i = 0; i < numRows; i++) {
  //     const start = i * fft;
  //     const end = start + fft;
  //     const row = data.subarray(start, end);
  //     spectrogram.push(Array.from(row));
  //   }

  //   return spectrogram;
  // }

  // // Create grid for time and frequency axes
  // const t = Array.from({ length: 10 }, (_, i) => i);
  // const f = Array.from({ length: 10 }, (_, i) => i);
  // const T = t.map((time) => Array(f.length).fill(time));
  // const F = Array(t.length).fill(f);

  // const spectogram1 = splitSpectrogram(multipleIQ[0]);
  // const spectogramSlice = spectogram1.slice(0, 100).map((row) => row.slice(0, 100));

  // const spectogram_portion = multipleIQ[0].subarray(0, 100);
  // // Create traces for each spectrogram
  // // const data = multipleIQ.map((spectrogram, index) => ({
  // const data = [
  //   {
  //     x: T.flat(),
  //     y: F.flat(),
  //     z: spectogram_portion,
  //     type: 'mesh3d',
  //     opacity: 0.8,
  //     colorscale: 'viridis',
  //     // name: `Spectrogram ${index + 1}`,
  //   },
  // ];
  // // }));

  // console.log(data);

  // return (
  //   <Plot
  //     data={data}
  //     layout={{
  //       xaxis: { title: 'Time' },
  //       yaxis: { title: 'Frequency' },
  //       zaxis: { title: 'Intensity (dB)' },
  //       // template: template,
  //     }}
  //   />
  // );
};
