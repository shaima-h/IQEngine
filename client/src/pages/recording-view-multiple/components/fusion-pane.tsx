// Copyright (c) 2022 Microsoft Corporation
// Copyright (c) 2023 Marc Lichtman
// Licensed under the MIT License

import React, { useState, useCallback } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faArrowRight } from '@fortawesome/free-solid-svg-icons';
import HelpOutlineOutlinedIcon from '@mui/icons-material/HelpOutlineOutlined';
import ArrowRightIcon from '@mui/icons-material/ArrowRight';
import DualRangeSlider from '@/features/ui/dual-range-slider/DualRangeSlider';
import { COLORMAP_DEFAULT } from '@/utils/constants';
import { colMaps } from '@/utils/colormap';
import { useSpectrogramContext } from '../hooks/use-spectrogram-context';
import { useCursorContext } from '../hooks/use-cursor-context';
import { vscodeDark } from '@uiw/codemirror-theme-vscode';
import CodeMirror from '@uiw/react-codemirror';
import { langs } from '@uiw/codemirror-extensions-langs';
import { DisplaySpectrogram } from '../recording-view-multiple';
// import { fuseFiles } from '@/utils/fusion';

interface FusionPaneProps {
  currentFFT: number;
}

const FusionPane = ({ currentFFT, filePaths }) => {
  const { meta, account, type, container, spectrogramWidth, spectrogramHeight, fftSize, selectedAnnotation, setMeta } =
    useSpectrogramContext();
  const fftSizes = [128, 256, 512, 1024, 2048, 4096, 16384, 65536];
  const context = useSpectrogramContext();
  const cursorContext = useCursorContext();
  const [fileStates, setFileStates] = useState({});
  const [fusionType, setFusionType] = useState<string>(null);
  const [isFusionRunning, setIsFusionRunning] = useState(false); // TODO
  //   const [localTaps, setLocalTaps] = useState(JSON.stringify(context.taps));

  const handleToggleChange = (filePath) => (event) => {
    setFileStates((prev) => ({
      ...prev,
      [filePath]: event.target.checked,
    }));
  };

  const handleFusionTypeChange = (event) => {
    setFusionType(event.target.value);
  };

  const onSubmitSelectedFiles = () => {
    // const selectedFiles = Object.entries(fileStates).filter(([_, isSelected]) => isSelected).map(([filePath, _]) => filePath);
    const selectedFiles: string[] = Object.keys(fileStates).filter((key: string) => fileStates[key]);
    // context.setFilePaths(selectedFiles);
    // context.setFusionType(fusionType);
    console.log('Selected files: ', selectedFiles);
    console.log('Selected fusion type: ', fusionType);
    context.setFusionType(fusionType);
    // console.log("Selected files in context: ", context.filePaths);
    // console.log("Selected fusion type in context: ", context.fusionType);
  };

  return (
    <div className="form-control">
      <p className="text-base">Select files to fuse:</p>
      <div className="p-1">
        {filePaths?.map((filePath, index) => {
          const fileName = filePath.replace(/\.[^/.]+$/, '');
          if (fileStates[filePath] === undefined) {
            setFileStates((prev) => ({
              ...prev,
              [filePath]: false,
            }));
          }
          return (
            <div key={index} id={`toggleFile-${index}`}>
              <label className="label pb-0 pt-2">
                <span className="label-text text-base">{fileName}</span>
                <input
                  type="checkbox"
                  className="toggle toggle-primary"
                  checked={fileStates[fileName]}
                  onChange={handleToggleChange(filePath)}
                />
              </label>
            </div>
          );
        })}
      </div>
      <div>
        <label className="label pb-2 pt-2">
          <span className="label-text text-base">Fusion Type</span>
          <select
            className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-32 p-1 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
            value={fusionType}
            onChange={handleFusionTypeChange}
          >
            <option value="" disabled selected>
              Select fusion
            </option>
            <option value="addition">Addition</option>
            <option value="subtraction">Subtraction</option>
            <option value="average">Average</option>
          </select>
        </label>
      </div>

      <button onClick={onSubmitSelectedFiles}>Fuse data</button>
    </div>
  );
};

export default FusionPane;
