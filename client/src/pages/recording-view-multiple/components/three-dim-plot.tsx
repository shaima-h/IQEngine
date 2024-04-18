import Plot from 'react-plotly.js';
import React, { useEffect, useState } from 'react';
import { template } from '@/utils/plotlyTemplate';
import { useSpectrogramContext } from '../hooks/use-spectrogram-context';

interface IQPlotProps {
  displayedIQ: Float32Array;
}

export const ThreeDimPlot = ({ displayedIQ }: IQPlotProps) => {
  const { spectrogramWidth, spectrogramHeight } = useSpectrogramContext();
  const [I, setI] = useState<Float32Array>();
  const [Q, setQ] = useState<Float32Array>();

  const [plotFilePath, setPlotFilePath] = useState('');

  useEffect(() => {
    let body = {
      samples_b64: [],
    };

    // const newSamps = convertFloat32ArrayToBase64(cursorData);
    // console.log(newSamps);

    body = {
      samples_b64: [
        {
          // samples: newSamps,
          // sample_rate: sampleRate,
          // center_freq: freq,
          // data_type: MimeTypes[meta.getDataType()],
        },
      ],
    };

    const BlobFromSamples = (samples_base64, data_type) => {
      const samples = window.atob(samples_base64);
      var blob_array = new Uint8Array(samples.length);
      for (var i = 0; i < samples.length; i++) {
        blob_array[i] = samples.charCodeAt(i);
      }
      return new Blob([blob_array], { type: data_type });
    };

    fetch('/api/plot')
      .then((response) => response.json())
      .then((data) => {
        console.log(data);
        setPlotFilePath(data.plot_file_path);
      });
  }, []);

  return (
    <div className="plot-container">
      {plotFilePath ? <img src={plotFilePath} alt="Plot" /> : <div>No plot available</div>}
    </div>
  );
};
