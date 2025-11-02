# Cell Image Analysis Tool

A Python-based GUI for detecting and analyzing fluorescence signals in microscopy images.  
The tool identifies distinct cellular regions in red and green fluorescence channels, calculates their Pearson correlation coefficients (PCC), and visualizes the overlap through bounding boxes and color-coded overlays.

---

## Features

- **Graphical interface** to load `.tif` or other image formats.
- **Automatic thresholding** (Otsu’s method) for segmentation of fluorescent regions.
- **Region property extraction** using `scikit-image`.
- **Pearson correlation computation** to quantify co-localization between color channels.
- **Bounding box visualization** with region-specific PCC values overlaid on the image.

---

## How It Works

1. **Image Input**  
   You select a multi-channel image (e.g., fluorescence microscopy `.tif`) through a simple GUI.

2. **Channel Separation**  
   The red and green channels are extracted for independent processing.

3. **Segmentation**  
   Each channel is thresholded using Otsu’s method, labeling connected bright regions as potential cellular structures.

4. **Filtering**  
   Small detections (area below 1000 pixels by default) are discarded to avoid noise.

5. **Correlation Analysis**  
   For each detected region:
   - The corresponding area in both channels is cropped.
   - The Pearson correlation coefficient (PCC) is computed between the red and green cropped regions.
   - Higher PCC values indicate stronger co-localization.

6. **Visualization**  
   The tool generates figures showing:
   - Detected regions (blue boxes).
   - PCC values (yellow text).
   - Overlays for each fluorescence channel.

---

## Example Input and Output

Below is an example showing the input image and the results of the analysis for each fluorescence channel.

| Input Image | Red Channel Analysis | Green Channel Analysis | Blue Channel Analysis | Combined Overlay |
|--------------|---------------------|------------------------|-----------------------|------------------|
|  <img width="500" alt="red" src="https://github.com/user-attachments/assets/d68e8a24-1295-420b-8319-546e90b1a166" />| <img width="500" alt="red" src="https://github.com/user-attachments/assets/d68e8a24-1295-420b-8319-546e90b1a166" /> | <img width="500" alt="green" src="https://github.com/user-attachments/assets/e16444ab-b978-4f53-8d8d-b2f591688bd1" /> | <img width="500" alt="blue" src="https://github.com/user-attachments/assets/2f1481a9-6c01-4599-9ff1-23b27b265448" /> | <img width="500" alt="combined" src="https://github.com/user-attachments/assets/e7d10a1a-cdd4-4845-afd2-0fcf0fa6c34d" /> |



---
