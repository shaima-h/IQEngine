// export function fuseFiles(selectedFiles: Float32Array[], fusionType: string): {
//   // TODO selected files type- string array of float 32 arrays? I'm very confused
//   if (selectedFiles.length === 0) {
//     console.error('No files selected for fusion.');
//     return null;
//   }

//   let fusedData: number[] = [];

//   if (fusionType === 'Addition') {
//     fusedData = selectedFiles.reduce((acc, curr) => {
//       return acc.map((value, index) => value + curr[index]);
//     });
//   } else if (fusionType === 'Subtraction') {
//     const firstFileData = selectedFiles[0];
//     fusedData = firstFileData.slice(); // copy the data of the first file

//     for (let i = 1; i < selectedFiles.length; i++) {
//       const currentFileData = selectedFiles[i];
//       fusedData = fusedData.map((value, index) => value - currentFileData[index]);
//     }
//   } else if (fusionType === 'Average') {
//     const totalFiles = selectedFiles.length;
//     fusedData = selectedFiles.reduce((acc, curr) => {
//       return acc.map((value, index) => value + curr[index] / totalFiles);
//     });
//   }

//   return fusedData;
// }
