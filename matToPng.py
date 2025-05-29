import os
import numpy as np
import h5py
from PIL import Image

inputFolder = 'C:/Users/aj/Desktop/Neural Network/Data'
outputFolder = 'C:/Users/aj/Desktop/Neural Network/DataSets'

fileList = [f for f in os.listdir(inputFolder) if f.endswith('.mat')]

for fileName in fileList:
    filePath = os.path.join(inputFolder, fileName)

    with h5py.File(filePath, 'r') as f:
        # Access nested MATLAB HDF5 structure
        img1 = np.array(f['cjdata']['image']).T  # Transpose due to column-major format
        label = int(np.array(f['cjdata']['label'])[0][0])

    # Normalize image to 0–255 and convert to uint8
    img1 = img1.astype(np.float64)
    min1 = img1.min()
    max1 = img1.max()
    img = ((img1 - min1) * 255 / (max1 - min1)).astype(np.uint8)

    # Create label folder
    labelFolder = os.path.join(outputFolder, str(label))
    os.makedirs(labelFolder, exist_ok=True)

    # Save image
    fileBase = os.path.splitext(fileName)[0]
    outputFilePath = os.path.join(labelFolder, fileBase + '.jpg')
    Image.fromarray(img).save(outputFilePath)

print("Finished.")