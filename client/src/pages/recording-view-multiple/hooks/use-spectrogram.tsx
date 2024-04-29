import { useGetIQData, useGetIQDataMultiple } from '@/api/iqdata/Queries';
import { useMemo } from 'react';
import { useSpectrogramContext } from './use-spectrogram-context';
import { useDebounce } from 'usehooks-ts';
import { FETCH_PADDING } from '@/utils/constants';

export function useSpectrogram(currentFFT) {
  // console.log("in use-spectrogram");
  const {
    type,
    account,
    container,
    filePath,
    fftSize,
    spectrogramHeight,
    fftStepSize,
    setFFTStepSize,
    setSpectrogramHeight,
    meta,
    taps,
    squareSignal,
    pythonSnippet,
    fusionType,
  } = useSpectrogramContext();
    console.log("in use-spectrogram");
    const { currentData, setFFTsRequired, fftsRequired, processedDataUpdated, processedDataUpdatedMultiple } = useGetIQDataMultiple(
      type,
      account,
      container,
      filePath,
      fftSize,
      taps,
      squareSignal,
      pythonSnippet,
      fftStepSize,
      fusionType,
    );
    // console.log("fftSize: ", fftSize);
    const totalFFTs = Math.ceil(meta?.getTotalSamples() / fftSize);
    // console.log("totalFTTs", totalFFTs);
    const debouncedCurrentFFT = useDebounce<string>(currentFFT, 50);
    // console.log("debouncedCurrentFFTs", debouncedCurrentFFT);

    // console.log("currentData: ", currentData);
    // This is the list of ffts we display
    const displayedIQ = useMemo<Float32Array>(() => {
      if (!totalFFTs || !spectrogramHeight) {
        return null;
      }
      // get the current required blocks
      const requiredBlocks: number[] = [];
      const displayedBlocks: number[] = [];

      // make the padding dependent on the size of fft so we avoid to fetch too much data for large ffts
      const currentPadding = Math.floor(FETCH_PADDING / (fftSize / 1024));
      for (let i = 0; i < spectrogramHeight; i++) {
        const nextFFT = currentFFT + i * (fftStepSize + 1);
        if (nextFFT <= totalFFTs && nextFFT >= 0) {
          requiredBlocks.push(nextFFT);
          displayedBlocks.push(nextFFT);
        }
      }
      // add the padding
      for (let i = 1; i <= currentPadding; i++) {
        let start = displayedBlocks[0];
        let end = displayedBlocks[displayedBlocks.length - 1];
        let step = i * (fftStepSize + 1);
        if (start - step >= 0) {
          requiredBlocks.push(start - step);
        }
        if (end + step <= totalFFTs) {
          requiredBlocks.push(end + step);
        }
      }

      if (!currentData || Object.keys(currentData).length === 0) {
        setFFTsRequired(requiredBlocks);
        // console.log("in this NULL if, new ffts: ", requiredBlocks);
        return null;
      }
      // check if the blocks are already loaded
      const blocksToLoad = requiredBlocks.filter((block) => !currentData[block]);
      setFFTsRequired(blocksToLoad);
      // console.log("there's currentData, new ffts: ", blocksToLoad);

      // return the data with 0s for the missing blocks
      const iqData = new Float32Array(spectrogramHeight * fftSize * 2);
      let offset = 0;
      for (let i = 0; i < spectrogramHeight; i++) {
        if (currentData[displayedBlocks[i]]) {
          if (currentData[displayedBlocks[i]].length + offset > iqData.length) {
            continue;
          }
          iqData.set(currentData[displayedBlocks[i]], offset);
        } else {
          if (offset + fftSize * 2 > iqData.length) {
            continue;
          }
          iqData.fill(-Infinity, offset, offset + fftSize * 2);
        }
        offset += fftSize * 2;
      }
      return iqData;
    }, [
      processedDataUpdated,
      processedDataUpdatedMultiple,
      fftSize,
      debouncedCurrentFFT,
      fftStepSize,
      totalFFTs,
      spectrogramHeight,
      taps,
      squareSignal,
    ]);
    // console.log("displayedIQ: ", displayedIQ);
    return {
      totalFFTs,
      currentFFT,
      spectrogramHeight,
      displayedIQ,
      currentData,
      fftsRequired,
      setFFTStepSize,
      setSpectrogramHeight,
    };
  }