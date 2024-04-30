// import Plot from 'react-plotly.js';
import React, { useEffect, useState } from 'react';
// import { template } from '@/utils/plotlyTemplate';
import { useSpectrogramContext } from '../hooks/use-spectrogram-context';
import { convertFloat32ArrayToBase64, convertBase64ToFloat32Array } from '@/utils/rf-functions';
import { useSpectrogram } from '../hooks/use-spectrogram';
// import { list } from 'postcss';

// interface IQPlotProps {
//   multipleIQ: Float32Array[];
// }

interface PlotImageData {
  data: [];
  layout: {};
}

export const ThreeDimPlot = ({ filePaths, currentFFT }) => {
  const { spectrogramWidth, spectrogramHeight, fftSize } = useSpectrogramContext();
  // const [plotImageData, setPlotImageData] = useState<PlotImageData>({ data: [], layout: {} });
  const [plotImageData, setPlotImageData] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const context = useSpectrogramContext();
  const currIQ = useSpectrogram(currentFFT);
  const [multipleIQ, setMultipleIQ] = useState([]); // State variable to hold IQ data for each file
  const [currentIndex, setCurrentIndex] = useState(-1);
  const [done, setDone] = useState(false);

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
      // setDone(true);
      // console.log(done);
      context.setFilePath(filePaths[0]); // reset to initial filepath once plot is received
    }
  }, [plotImageData]);

  // // Set done to true once when the component mounts
  // useEffect(() => {
  //   setDone(true);
  // }, []);

  // Populate multipleIQ to send to python function
  useEffect(() => {
    const listIQData = filePaths.map((file_path) => {
      context.setFilePath(file_path);
      console.log('file_path', file_path);
      console.log('context.filePath', context.filePath);
      console.log('currIQ', currIQ.displayedIQ);
      return currIQ.displayedIQ; // filepath iqdata
    });

    console.log('listIQData', listIQData);
    setMultipleIQ(listIQData);
  }, [filePaths]);

  // // first & last
  // useEffect(() => {
  //   if (done) {
  //     // Reset state when file paths change (page first loads) or plot received and done
  //     console.log('resetting');
  //     setMultipleIQ([]);
  //     setCurrentIndex(0);
  //     context.setFilePath(filePaths[currentIndex]);
  //   }
  // }, [done]);

  // // second
  // useEffect(() => {
  //   if (!done && currentIndex < filePaths.length) {
  //     // If there are files left to process, fetch data for the current file
  //     context.setFilePath(filePaths[currentIndex]);
  //     console.log('context.filepath', context.filePath);
  //   }
  // }, [currentIndex]);

  // // third
  // useEffect(() => {
  //   console.log('in context.filepath effect');
  //   if (!done && currIQ.displayedIQ && !multipleIQ.includes(currIQ.displayedIQ)) {
  //     setMultipleIQ((prevIQ) => [...prevIQ, currIQ.displayedIQ]);
  //     console.log('multipleIQ', multipleIQ);
  //     setCurrentIndex(currentIndex + 1);
  //   }
  // }, [currIQ.displayedIQ]); // or currIQ.displayedIQ?

  // send data to python function
  useEffect(() => {
    if (multipleIQ.length == filePaths.length) {
      let body = {
        samples_b64: [],
      };

      console.log('sending multipleIQ', multipleIQ);
      body = {
        samples_b64: multipleIQ.map((iqData) => ({
          samples: convertFloat32ArrayToBase64(iqData),
        })),
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
    }
  }, [multipleIQ]);

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
