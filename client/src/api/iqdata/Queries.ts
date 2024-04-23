import { useQuery, useQueryClient } from '@tanstack/react-query';
import { IQDataClientFactory, IQDataClientFactoryMultiple } from './IQDataClientFactory';
import { INITIAL_PYTHON_SNIPPET } from '@/utils/constants';
import { useUserSettings } from '@/api/user-settings/use-user-settings';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useMeta } from '@/api/metadata/queries';
import { useMsal } from '@azure/msal-react';
import { applyProcessing } from '@/utils/fetch-more-data-source';
import { groupContiguousIndexes } from '@/utils/group';

declare global {
  interface Window {
    loadPyodide: any;
  }
}

const MAXIMUM_SAMPLES_PER_REQUEST = 1024 * 256;

export function useDataCacheFunctions(
  type: string,
  account: string,
  container: string,
  filePath: string,
  fftSize: number
) {
  const queryClient = useQueryClient();
  function clearIQData() {
    queryClient.removeQueries(['iqData', type, account, container, filePath, fftSize]);
    queryClient.removeQueries(['rawiqdata', type, account, container, filePath, fftSize]);
    queryClient.removeQueries(['processedIQData', type, account, container, filePath, fftSize]);
    queryClient.removeQueries(['iqDataMultiple', type, account, container, filePath, fftSize]);
    queryClient.removeQueries(['rawiqdataMultiple', type, account, container, filePath, fftSize]);
    queryClient.removeQueries(['processedIQDataMultiple', type, account, container, filePath, fftSize]);
  }
  return {
    clearIQData,
  };
}

export function useGetIQData(
  type: string,
  account: string,
  container: string,
  filePath: string,
  fftSize: number, // we grab 2x this many floats/ints
  taps: number[] = [1],
  squareSignal: boolean = false,
  pythonScript: string = INITIAL_PYTHON_SNIPPET,
  fftStepSize: number = 0
) {
  console.log("IN USEGETIQDATA");
  const [pyodide, setPyodide] = useState<any>(null);

  async function initPyodide() {
    console.log('Loading pyodide...');
    const pyodide = await window.loadPyodide();
    await pyodide.loadPackage('numpy');
    await pyodide.loadPackage('matplotlib');
    return pyodide;
  }

  useEffect(() => {
    if (!pyodide && pythonScript && pythonScript !== INITIAL_PYTHON_SNIPPET && fftStepSize === 0) {
      initPyodide().then((pyodide) => {
        setPyodide(pyodide);
      });
    }
  }, [pythonScript, fftStepSize]);

  const queryClient = useQueryClient();
  const { filesQuery, dataSourcesQuery } = useUserSettings();
  const [fftsRequired, setStateFFTsRequired] = useState<number[]>([]);

  // enforce MAXIMUM_SAMPLES_PER_REQUEST by truncating if need be
  function setFFTsRequired(fftsRequired: number[]) {
    fftsRequired = fftsRequired.slice(
      0,
      fftsRequired.length > Math.ceil(MAXIMUM_SAMPLES_PER_REQUEST / fftSize)
        ? Math.ceil(MAXIMUM_SAMPLES_PER_REQUEST / fftSize)
        : fftsRequired.length
    );
    setStateFFTsRequired(fftsRequired);
    // console.log("new fftsRequired in single: ", fftsRequired);
  }

  const { data: meta } = useMeta(type, account, container, filePath);

  const { instance } = useMsal();

  const iqDataClient = IQDataClientFactory(type, filesQuery.data, dataSourcesQuery.data, instance);

  // fetches iqData, this happens first, and the iqData is in one big continuous chunk
  const { data: iqData } = useQuery({
    queryKey: ['iqData', type, account, container, filePath, fftSize, fftsRequired],
    queryFn: async ({ signal }) => {
      const iqData = await iqDataClient.getIQDataBlocks(meta, fftsRequired, fftSize, signal);
      return iqData;
    },
    enabled: !!meta && !!filesQuery.data && !!dataSourcesQuery.data,
  });
  if (iqData) {
    console.log("iqData in single: ", iqData);
  }

  // This sets rawiqdata, rawiqdata contains all the data, while the iqData above is just the recently fetched one
  useEffect(() => {
    if (iqData) {
      const previousData = queryClient.getQueryData<Float32Array[]>([
        'rawiqdata',
        type,
        account,
        container,
        filePath,
        fftSize,
      ]);
      const sparseIQReturnData = [];
      iqData.forEach((data) => {
        sparseIQReturnData[data.index] = data.iqArray;
      });
      const content = Object.assign([], previousData, sparseIQReturnData);
      console.log("content in single: ", content.slice(0,20));
      queryClient.setQueryData(['rawiqdata', type, account, container, filePath, fftSize], content);
    }
  }, [iqData, fftSize]);

  // fetches rawiqdata
  const { data: processedIQData, dataUpdatedAt: processedDataUpdated } = useQuery<number[][]>({
    queryKey: ['rawiqdata', type, account, container, filePath, fftSize],
    queryFn: async () => {
      return [];
    },
    select: useCallback(
      (data) => {
        if (!data) {
          return [];
        }
        // performance.mark('start');
        let currentProcessedData = queryClient.getQueryData<number[][]>([
          'processedIQData',
          type,
          account,
          container,
          filePath,
          fftSize,
          taps,
          squareSignal,
          pythonScript,
          !!pyodide,
        ]);

        if (!currentProcessedData) {
          currentProcessedData = [];
        }
        let currentIndexes = data.map((_, i) => i);
        // remove any data that have already being processed
        const dataRange = currentIndexes.filter((index) => !currentProcessedData[index]);

        groupContiguousIndexes(dataRange).forEach((group) => {
          const iqData = data.slice(group.start, group.start + group.count);
          const iqDataFloatArray = new Float32Array((iqData.length + 1) * fftSize * 2);
          iqData.forEach((data, index) => {
            iqDataFloatArray.set(data, index * fftSize * 2);
          });
          const result = applyProcessing(iqDataFloatArray, taps, squareSignal, pythonScript, pyodide);

          for (let i = 0; i < group.count; i++) {
            currentProcessedData[group.start + i] = result.slice(i * fftSize * 2, (i + 1) * fftSize * 2);
          }
        });
        // performance.mark('end');
        // const performanceMeasure = performance.measure('processing', 'start', 'end');
        queryClient.setQueryData(
          ['processedIQData', type, account, container, filePath, fftSize, taps, squareSignal, pythonScript, !!pyodide],
          currentProcessedData
        );

        return currentProcessedData;
      },
      [!!pyodide, pythonScript, taps.join(','), squareSignal]
    ),
    enabled: !!meta && !!filesQuery.data && !!dataSourcesQuery.data,
  });

  const currentData = processedIQData;

  return {
    fftSize,
    currentData,
    fftsRequired,
    setFFTsRequired,
    processedDataUpdated,
  };
}

export function useGetIQDataMultiple(
  type: string,
  account: string,
  container: string,
  filePath: string,
  outerfftSize: number, // we grab 2x this many floats/ints
  taps: number[] = [1],
  squareSignal: boolean = false,
  pythonScript: string = INITIAL_PYTHON_SNIPPET,
  fftStepSize: number = 0,
  fusionType: string,
) {
  // this if block should be entered when the recording-view-multiple page first loads,
  // so there's no fusion type yet and the normal single-file process happens to load
  // the first trace into the spectrogram only
  if (fusionType === "") {
    const inputfftSize = outerfftSize;  // had duplicate naming issues
    const { fftSize, currentData, setFFTsRequired, fftsRequired, processedDataUpdated } = useGetIQData(type, account, container, filePath, inputfftSize, taps, squareSignal, pythonScript, fftStepSize);
    return {
      fftSize,
      currentData,
      fftsRequired,
      setFFTsRequired,
      processedDataUpdated,
    }
  }
  else {
    console.log("IN USEGETIQDATAMULTIPLE");
    const fftSize = outerfftSize;
    const [pyodide, setPyodide] = useState<any>(null);

    async function initPyodide() {
      console.log('Loading pyodide...');
      const pyodide = await window.loadPyodide();
      await pyodide.loadPackage('numpy');
      await pyodide.loadPackage('matplotlib');
      return pyodide;
    }

    useEffect(() => {
      if (!pyodide && pythonScript && pythonScript !== INITIAL_PYTHON_SNIPPET && fftStepSize === 0) {
        initPyodide().then((pyodide) => {
          setPyodide(pyodide);
        });
      }
    }, [pythonScript, fftStepSize]);

    const queryClient = useQueryClient();
    const { filesQuery, dataSourcesQuery } = useUserSettings(); 
    const [fftsRequired, setStateFFTsRequired] = useState<number[]>([]);

    // enforce MAXIMUM_SAMPLES_PER_REQUEST by truncating if need be
    function setFFTsRequired(fftsRequired: number[]) {
      fftsRequired = fftsRequired.slice(
        0,
        fftsRequired.length > Math.ceil(MAXIMUM_SAMPLES_PER_REQUEST / fftSize)
          ? Math.ceil(MAXIMUM_SAMPLES_PER_REQUEST / fftSize)
          : fftsRequired.length
      );
      // console.log("new fftsRequired in multiple: ", fftsRequired);
      setStateFFTsRequired(fftsRequired);
    }

    const { data: meta } = useMeta(type, account, container, filePath);

    const { instance } = useMsal();

    const iqDataClient = IQDataClientFactoryMultiple(type, filesQuery.data, dataSourcesQuery.data, instance);

    // fetches iqData, this happens first, and the iqData is in one big continuous chunk
    
    const { data: iqData } = useQuery({
      queryKey: ['iqDataMultiple', type, account, container, filePath, fftSize, fftsRequired],
      queryFn: async ({ signal }) => {
        // console.log("inside query function!");
        const iqData = await iqDataClient.getIQDataBlocksMultiple(meta, fftsRequired, fftSize, signal);
        return iqData;
      },
      enabled: !!meta && !!filesQuery.data && !!dataSourcesQuery.data,
    });

    if (iqData) {
      console.log("iqData[0] in multiple: ", iqData[0]);
    }

    // This sets rawiqdata, rawiqdata contains all the data, while the iqData above is just the recently fetched one
    useEffect(() => {
      // console.log("in useEffect");
      if (iqData) {
        // console.log("in this useEffect if block");
        const previousData = queryClient.getQueryData<Float32Array[]>([
          'rawiqdataMultiple',
          type,
          account,
          container,
          filePath,
          fftSize,
        ]);
        const sparseIQReturnData = [];
        iqData.forEach((sliceArray) => {
          sliceArray.forEach((slice) => {
            if (sparseIQReturnData[slice.index]) {
              // element-wise addition if row already exists in the output matrix
              // just assuming addition for now
              sparseIQReturnData[slice.index] = sparseIQReturnData[slice.index].map((value, i) => value + slice.iqArray.at(i));
            } else {
              // just assign the iqArray if no existing row is present yet in the output matrix
              sparseIQReturnData[slice.index] = new Float32Array(slice.iqArray);
            }
          });
        });
        const content = Object.assign([], previousData, sparseIQReturnData);
        console.log("content in multiple: ", content.slice(0,20));
        queryClient.setQueryData(['rawiqdataMultiple', type, account, container, filePath, fftSize], content);
      }
    }, [iqData, fftSize]);

    // fetches rawiqdata
    const { data: processedIQDataMultiple, dataUpdatedAt: processedDataUpdated } = useQuery<number[][]>({
      queryKey: ['rawiqdataMultiple', type, account, container, filePath, fftSize],
      queryFn: async () => {
        return [];
      },
      select: useCallback(
        (data) => {
          if (!data) {
            return [];
          }
          // performance.mark('start');
          let currentProcessedData = queryClient.getQueryData<number[][]>([
            'processedIQDataMultiple',
            type,
            account,
            container,
            filePath,
            fftSize,
            taps,
            squareSignal,
            pythonScript,
            !!pyodide,
          ]);

          if (!currentProcessedData) {
            currentProcessedData = [];
          }
          let currentIndexes = data.map((_, i) => i);
          // remove any data that have already being processed
          const dataRange = currentIndexes.filter((index) => !currentProcessedData[index]);

          groupContiguousIndexes(dataRange).forEach((group) => {
            const iqData = data.slice(group.start, group.start + group.count);
            const iqDataFloatArray = new Float32Array((iqData.length + 1) * fftSize * 2);
            iqData.forEach((data, index) => {
              iqDataFloatArray.set(data, index * fftSize * 2);
            });
            const result = applyProcessing(iqDataFloatArray, taps, squareSignal, pythonScript, pyodide);

            for (let i = 0; i < group.count; i++) {
              currentProcessedData[group.start + i] = result.slice(i * fftSize * 2, (i + 1) * fftSize * 2);
            }
          });
          // performance.mark('end');
          // const performanceMeasure = performance.measure('processing', 'start', 'end');
          queryClient.setQueryData(
            ['processedIQDataMultiple', type, account, container, filePath, fftSize, taps, squareSignal, pythonScript, !!pyodide],
            currentProcessedData
          );

          return currentProcessedData;
        },
        [!!pyodide, pythonScript, taps.join(','), squareSignal]
      ),
      enabled: !!meta && !!filesQuery.data && !!dataSourcesQuery.data,
    });

    const currentData = processedIQDataMultiple;

    return {
      fftSize,
      currentData,
      fftsRequired,
      setFFTsRequired,
      processedDataUpdated,
    };
  }
}

export function useRawIQData(type, account, container, filePath, fftSize) {
  const rawIQQuery = useQuery<Float32Array[]>({
    queryKey: ['rawiqdata', type, account, container, filePath, fftSize],
    queryFn: async () => null,
  });
  const downloadedIndexes = useMemo<number[]>(() => {
    if (!rawIQQuery.data) {
      return [];
    }
    // get all the array positions that have any data without use of reduce
    const downloadedIndexes = [];
    rawIQQuery.data.forEach((data, index) => {
      if (data) {
        downloadedIndexes.push(index);
      }
    });
    return downloadedIndexes;
  }, [rawIQQuery.data]);
  return {
    downloadedIndexes,
    rawIQQuery,
  };
}

export function useGetMinimapIQ(type: string, account: string, container: string, filePath: string, enabled = true) {
  const { data: meta } = useMeta(type, account, container, filePath);
  const { filesQuery, dataSourcesQuery } = useUserSettings();
  const { instance } = useMsal();
  const iqDataClient = IQDataClientFactory(type, filesQuery.data, dataSourcesQuery.data, instance);
  const minimapQuery = useQuery<Float32Array[]>({
    queryKey: ['minimapiq', type, account, container, filePath],
    queryFn: async ({ signal }) => {
      const minimapIQ = await iqDataClient.getMinimapIQ(meta, signal);
      return minimapIQ;
    },
    enabled: enabled && !!meta && !!filesQuery.data && !!dataSourcesQuery.data,
  });
  return minimapQuery;
}
