import Plot from 'react-plotly.js';
import React, { useEffect, useState } from 'react';
import { template } from '@/utils/plotlyTemplate';
import { useSpectrogramContext } from '../hooks/use-spectrogram-context';
import { convertFloat32ArrayToBase64, convertBase64ToFloat32Array } from '@/utils/rf-functions';

interface IQPlotProps {
  displayedIQ: Float32Array;
}

export const ThreeDimPlot = ({ displayedIQ }: IQPlotProps) => {
  const { spectrogramWidth, spectrogramHeight } = useSpectrogramContext();
  const [I, setI] = useState<Float32Array>();
  const [Q, setQ] = useState<Float32Array>();

  const [plotImageData, setPlotImageData] = useState('');

  useEffect(() => {
    let body = {
      samples_b64: [],
    };

    const newSamps1 = convertFloat32ArrayToBase64(displayedIQ);
    console.log(newSamps1);

    // body = {
    //   samples_b64: [
    //     {
    //       // samples: newSamps,
    //       // sample_rate: sampleRate,
    //       // center_freq: freq,
    //       // data_type: MimeTypes[meta.getDataType()],
    //     },
    //   ],
    // };

    // samples.forEach((sample) => {
    //   body.samples_b64.push({
    //     samples: sample.base64Data, // Assuming each sample has a base64Data property
    //     sample_rate: sample.sampleRate,
    //     center_freq: sample.centerFrequency,
    //     data_type: sample.dataType,
    //   });
    // });

    body = {
      samples_b64: [
        {
          samples: newSamps1,
        },
      ],
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
        console.log(data);
        setPlotImageData(data.plot_image_base64);
      });
  }, []);

  return (
    //this has to be localhost/... (client folder)
    <div className="plot-container">
      {/* {plotFilePath ? <img src={`http://localhost:3000/${plotFilePath}`} alt="Plot" /> : <div>No plot available</div>} */}
      {plotImageData ? (
        <img src={`data:image/png;base64,${plotImageData}`} alt="Plot" style={{ width: '100%', height: 'auto' }} />
      ) : (
        <div>No plot available</div>
      )}
    </div>
  );
};
