// Copyright (c) 2022 Microsoft Corporation
// Copyright (c) 2023 Marc Lichtman
// Licensed under the MIT License

import React from 'react';
import FusionPane from './fusion-pane';
import SettingsPane from './settings-pane';
import { PluginsPane } from './plugins-pane';

interface SidebarProps {
  currentFFT: number;
}

const Sidebar = ({ currentFFT, filePaths }) => {
  return (
    <div className="flex flex-col w-64 ml-3">
      <details open>
        <summary className="pl-2 bg-primary outline outline-1 outline-primary text-lg text-base-100 hover:bg-green-800">
          Fusion
        </summary>
        <div className="outline outline-1 outline-primary p-2">
          <FusionPane currentFFT={currentFFT} filePaths={filePaths} />
        </div>
      </details>
      {/* I commented out the Settings and Plugins panes for now since they were causing extra issues.. 
          I thought we could get the multi-trace stuff working first, then add back in these panes? */}
      <details open>
        <summary className="pl-2 bg-primary outline outline-1 outline-primary text-lg text-base-100 hover:bg-green-800">
          Settings
        </summary>
        <div className="outline outline-1 outline-primary p-2">
          <SettingsPane currentFFT={currentFFT} filePaths={filePaths} />
        </div>
      </details>

      {/* 
      <details>
        <summary className="pl-2 mt-2 bg-primary outline outline-1 outline-primary text-lg text-base-100 hover:bg-green-800">
          Plugins
        </summary>
        <div className="outline outline-1 outline-primary p-2">
          <PluginsPane />
        </div>
      </details> */}
    </div>
  );
};

export { Sidebar };
