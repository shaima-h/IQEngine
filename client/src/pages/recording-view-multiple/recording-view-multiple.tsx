import React, { useEffect, useState } from 'react';
import { useParams, useLocation } from 'react-router-dom';
import { useSpectrogram } from './hooks/use-spectrogram';
import { Layer, Stage, Image } from 'react-konva';
import { useGetImage } from './hooks/use-get-image';
import { KonvaEventObject } from 'konva/lib/Node';
import { RulerTop } from './components/ruler-top';
import { RulerSide } from './components/ruler-side';
import { SpectrogramContextProvider, useSpectrogramContext } from './hooks/use-spectrogram-context';
import { CursorContextProvider, useCursorContext } from './hooks/use-cursor-context';
import { useMeta } from '@/api/metadata/queries';
import { IQPlot } from './components/iq-plot';
import { FrequencyPlot } from './components/frequency-plot';
import { TimePlot } from './components/time-plot';
import { Sidebar } from './components/sidebar';
import GlobalProperties from './components/global-properties';
import MetaViewer from './components/meta-viewer';
import MetaRaw from './components/meta-raw';
import AnnotationList from './components/annotation/annotation-list';
import ScrollBar from './components/scroll-bar';
import { MINIMAP_FFT_SIZE, MIN_SPECTROGRAM_HEIGHT } from '@/utils/constants';
import FreqSelector from './components/freq-selector';
import TimeSelector from './components/time-selector';
import { AnnotationViewer } from './components/annotation/annotation-viewer';
import TimeSelectorMinimap from './components/time-selector-minimap';
import { useWindowSize } from 'usehooks-ts';
import {} from './components/fusion-pane';
import { ThreeDimPlot } from './components/three-dim-plot';

export function DisplaySpectrogram({ currentFFT, setCurrentFFT, currentTab, filePaths }) {
  const {
    spectrogramWidth,
    magnitudeMin,
    magnitudeMax,
    colmap,
    windowFunction,
    fftSize,
    fftStepSize,
    meta,
    setSpectrogramWidth,
    setSpectrogramHeight,
    filePath,
    setFilePath,
  } = useSpectrogramContext();
  // console.log("in DISPLAY spectrogram")
  // console.log("currentFFT: ", currentFFT);
  const { displayedIQ, spectrogramHeight } = useSpectrogram(currentFFT);
  const [multipleIQ, setMultipleIQ] = useState([]); // State variable to hold IQ data for each file
  // console.log("displayedIQ: ", displayedIQ.slice(0,10));
  const { width, height } = useWindowSize();
  // console.log("width, height: ", width, height);
  // const context = useSpectrogramContext();
  const newIQ = useSpectrogram(currentFFT);
  // const [currFilePath, setCurrFilePath] = useState('');

  useEffect(() => {
    const spectrogramHeight = height - 450; // hand-tuned for now
    //console.log('spectrogramHeight: ', spectrogramHeight);
    setSpectrogramHeight(Math.max(MIN_SPECTROGRAM_HEIGHT, spectrogramHeight));
    const newSpectrogramWidth = width - 430; // hand-tuned for now
    setSpectrogramWidth(newSpectrogramWidth);
  }, [width, height]);

  const { image, setIQData } = useGetImage(
    fftSize,
    spectrogramHeight,
    magnitudeMin,
    magnitudeMax,
    colmap,
    windowFunction
  );

  function handleWheel(evt: KonvaEventObject<WheelEvent>): void {
    evt.evt.preventDefault();
    const scrollAmount = Math.floor(evt.evt.deltaY);
    const nextPosition = currentFFT + scrollAmount + spectrogramHeight * (fftStepSize + 1);
    const maxPosition = meta.getTotalSamples() / fftSize;

    if (nextPosition < maxPosition) {
      setCurrentFFT(Math.max(0, currentFFT + scrollAmount));
    }
  }

  useEffect(() => {
    // Check if the current tab is ThreeDimensionalVisualization
    if (currentTab === Tab.ThreeDimensionalVisualization) {
      console.log('in 3d effect', filePaths);
      const listIQData = filePaths.map((file_path) => {
        setFilePath(file_path);
        console.log('file_path', file_path);
        console.log('context.filePath', filePath); // this is always karyn_sample (first file), is never changing
        console.log('****new IQ', newIQ);
        return newIQ.displayedIQ; // filepath iqdata
      });

      console.log('****listIQs', listIQData);
      setMultipleIQ(listIQData);
      setFilePath(filePaths[0]); // reset to initial file path
    }
  }, [currentTab]);

  useEffect(() => {
    if (displayedIQ && displayedIQ.length > 0) {
      console.log('displayspectrogram EFFECT displayedIQ');
      setIQData(displayedIQ);

      // works: Populate multipleIQ with IQ data for all files
      // const newIQs = [displayedIQ, displayedIQ]; // just two of the first file to test
      // setMultipleIQ(newIQs);

      // =========== works, same file 2x
      // console.log(filePaths);
      // const listIQs = filePaths.map((file_path) => {
      //   setCurrFilePath(file_path);
      //   console.log('currFile', currFilePath);
      //   // setFilePath(file_path);

      //   // so newIQ is same as displayedIQ
      //   // so changing context.filepath is not affecting useSpectogram(currrentfft)
      //   // what is connection between context and useSpectogram??
      //   // how to like pass context into useSpectogram??
      //   console.log('file_path', file_path);
      //   console.log('context.filePath', filePath);
      //   console.log('****new IQ', newIQ);
      //   return newIQ.displayedIQ; // filepath iqdata
      // });

      // console.log('****listIQs', listIQs);
      // setMultipleIQ(listIQs);
      // setFilePath(filePaths[0]); // reset to initial file path
    }
  }, [displayedIQ]);

  useEffect(() => {
    console.log('EFFECT context.filePath', filePath);
  }, [filePath]);

  // had to comment out a bunch of features/components that were causing errors (I think
  // to do with the metadata file and the fact that we now have multiple metadata files...)
  // I thought we could get the multi-trace code working first and then add these features
  // back in one by one??
  return (
    <>
      {currentTab === Tab.Spectrogram && (
        <>
          <Stage width={spectrogramWidth + 110} height={30}>
            {/* <RulerTop /> */}
          </Stage>
          <div className="flex flex-row" id="spectrogram">
            <Stage width={spectrogramWidth} height={spectrogramHeight}>
              <Layer onWheel={handleWheel}>
                <Image image={image} x={0} y={0} width={spectrogramWidth} height={spectrogramHeight} />
              </Layer>
              {/* <AnnotationViewer currentFFT={currentFFT} /> */}
              <FreqSelector />
              <TimeSelector currentFFT={currentFFT} />
            </Stage>
            <Stage width={50} height={spectrogramHeight} className="mr-1">
              {/* <RulerSide currentRowAtTop={currentFFT} /> */}
            </Stage>
            <Stage width={MINIMAP_FFT_SIZE + 5} height={spectrogramHeight}>
              <ScrollBar currentFFT={currentFFT} setCurrentFFT={setCurrentFFT} />
              {/* <TimeSelectorMinimap /> */}
            </Stage>
          </div>
        </>
      )}
      {currentTab === Tab.Time && <TimePlot displayedIQ={displayedIQ} />}
      {currentTab === Tab.Frequency && <FrequencyPlot displayedIQ={displayedIQ} />}
      {currentTab === Tab.IQ && <IQPlot displayedIQ={displayedIQ} />}
      {currentTab === Tab.ThreeDimensionalVisualization && <ThreeDimPlot multipleIQ={multipleIQ} />}
    </>
  );
}

export function DisplayMetadataRaw() {
  const { meta } = useSpectrogramContext();
  return <MetaRaw meta={meta} />;
}

export function DisplayMetaSummary() {
  const { meta } = useSpectrogramContext();
  return <MetaViewer meta={meta} />;
}

// export function threeDimPlot() {
//   const location = useLocation();
//   const { multipleFilePath } = location.state || {};
//   const context = useSpectrogramContext();

//   const listIQData = multipleFilePath.map((file_path) => {
//     context.setFilePath(file_path);
//     console.log('file_path', file_path);
//     console.log('context.filePath', context.filePath); // this is always karyn_sample (first file), is never changing
//     const newIQ = useSpectrogram(context.fftSize);
//     console.log('****new IQ', newIQ);
//     return newIQ.displayedIQ; // filepath iqdata
//   });

//   console.log('****listIQs', listIQData);
//   const multipleIQ = listIQData;
//   context.setFilePath(multipleFilePath[0]); // reset to initial file path
//   return <ThreeDimPlot multipleIQ={multipleIQ} />;
// }

enum Tab {
  Spectrogram,
  Time,
  Frequency,
  IQ,
  ThreeDimensionalVisualization,
}

export function RecordingViewMultiplePage() {
  const { type, account, container } = useParams();
  const location = useLocation();
  const { multipleFilePath } = location.state || {};
  const filePath = multipleFilePath[0];
  const initialFilePath = multipleFilePath[0];
  const initialArray: string[] = [];
  initialArray.push(initialFilePath);
  const { data: meta } = useMeta(type, account, container, initialFilePath);
  // const meta = null;
  const [currentTab, setCurrentTab] = useState<Tab>(Tab.Spectrogram);
  const [currentFFT, setCurrentFFT] = useState<number>(0);
  const [filePaths, setFilePaths] = useState<number>(0);
  const Tabs = Object.keys(Tab).filter((key) => isNaN(Number(key)));

  console.log({ meta });
  if (!meta) {
    return (
      <div className="flex flex-col items-center justify-center h-full">
        <div className="text-2xl font-bold">Loading...</div>
      </div>
    );
  }
  return (
    <SpectrogramContextProvider
      type={type}
      account={account}
      container={container}
      initialFilePath={initialFilePath}
      initialFusionType=""
    >
      <CursorContextProvider>
        <div className="mb-0 ml-0 mr-0 p-0 pt-3">
          <div className="flex flex-row w-full">
            <Sidebar currentFFT={currentFFT} filePaths={multipleFilePath} />
            <div className="flex flex-col pl-3">
              <div className="flex space-x-2 border-b border-primary w-full sm:pl-12 lg:pl-32" id="tabsbar">
                {Tabs.map((key) => {
                  return (
                    <div
                      key={key}
                      onClick={() => {
                        setCurrentTab(Tab[key as keyof typeof Tab]);
                      }}
                      className={` ${
                        currentTab === Tab[key as keyof typeof Tab] ? 'bg-primary !text-base-100' : ''
                      } inline-block px-3 py-0 outline outline-primary outline-1 text-lg text-primary hover:text-accent hover:shadow-lg hover:shadow-accent`}
                    >
                      {key}
                    </div>
                  );
                })}
              </div>
              {/* The following displays the spectrogram, time, freq, and IQ plots depending on which one is selected*/}
              <DisplaySpectrogram
                currentFFT={currentFFT}
                setCurrentFFT={setCurrentFFT}
                currentTab={currentTab}
                filePaths={multipleFilePath}
              />
              {/* <DisplayMetaSummary /> */}
            </div>
          </div>
          <div className="mt-3 mb-0 px-2 py-0" style={{ margin: '5px' }}>
            <details>
              <summary className="pl-2 mt-2 bg-primary outline outline-1 outline-primary text-lg text-base-100 hover:bg-green-800">
                Annotations
              </summary>
              <div className="outline outline-1 outline-primary p-2">
                <AnnotationList setCurrentFFT={setCurrentFFT} currentFFT={currentFFT} />
              </div>
            </details>

            <details>
              <summary className="pl-2 mt-2 bg-primary outline outline-1 outline-primary text-lg text-base-100 hover:bg-green-800">
                Global Properties
              </summary>
              <div className="outline outline-1 outline-primary p-2">
                <GlobalProperties />
              </div>
            </details>
            <details>
              <summary className="pl-2 mt-2 bg-primary outline outline-1 outline-primary text-lg text-base-100 hover:bg-green-800">
                Raw Metadata
              </summary>
              <div className="outline outline-1 outline-primary p-2">{/* <DisplayMetadataRaw /> */}</div>
            </details>
          </div>
        </div>
      </CursorContextProvider>
    </SpectrogramContextProvider>
  );
}
