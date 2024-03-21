import numpy as np
import skimage.io as skio
import skimage.filters as skf
import skimage.measure as skm
import skimage.draw as skd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Analysis")

        self.label = tk.Label(root, text="Select an image:")
        self.label.pack()

        self.btn_browse = tk.Button(root, text="Browse", command=self.browse_image)
        self.btn_browse.pack()

        self.img_path = None

        self.processor = ImageProcessing()

    def browse_image(self):
        self.img_path = filedialog.askopenfilename()
        if self.img_path:
            self.processor.process_image(self.img_path)

class ImageProcessing:

    # Define a function to fill holes in the image and return the contours
    def fill_image(self, channel):
        thresh = channel > skf.threshold_otsu(channel)
        filled_image = skm.label(thresh)
        return filled_image, skm.regionprops(filled_image)

    def get_pearson(self, image_1, image_2):
        # Compute Pearson correlation coefficient
        pcc, _ = skm.pearson_corr_coeff(image_1, image_2)
        return pcc

    def figure_pcc(self, img, data):
        fig, ax = plt.subplots()
        if img.ndim < 3:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)

        for pcc, coordinate, props in zip(
            data["pcc"], data["coordinates"], data["contour"]
        ):
            plt.text(
                coordinate[1],
                coordinate[0],
                f"{pcc:.4f}",
                fontsize=8,
                color="yellow",
                ha="center",
            )

            minr, minc, maxr, maxc = props.bbox
            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)
            ax.plot(bx, by, "-b", linewidth=2.5)

    def crop_image(self, image, props):
        # Get image dimensions
        if image.ndim == 2:
            height, width = image.shape
        else:
            height, width, _ = image.shape

        # Extract bounding box coordinates from region properties
        minr, minc, maxr, maxc = props.bbox

        # Ensure the crop coordinates are within image bounds
        top = max(0, minr)
        bottom = min(height, maxr)
        left = max(0, minc)
        right = min(width, maxc)

        # Crop the image
        cropped_image = image[top:bottom, left:right]

        return cropped_image

    def process_image(self, img_path):
        data = {
            "green": {"coordinates": [], "pcc": [], "contour": []},
            "red": {"coordinates": [], "pcc": [], "contour": []},
        }

        # Read the image
        img = skio.imread(img_path)

        # Split the image into channels
        red = img[:, :, 0]
        green = img[:, :, 1]

        # Fill holes in the red and green channels and get contours
        red_filled, red_contours = self.fill_image(red)
        green_filled, green_contours = self.fill_image(green)

        # Filter contours based on size (area)
        min_contour_area = 1000  # Adjust as needed
        red_contours = [props for props in red_contours if props.area > min_contour_area]
        green_contours = [
            props for props in green_contours if props.area > min_contour_area
        ]

        # Fill different figures using contours and compute Pearson correlation coefficient for each
        for props in red_contours:
            green_cropped = self.crop_image(green_filled, props)
            red_cropped = self.crop_image(red_filled, props)
            pcc = self.get_pearson(red_cropped, green_cropped)
            data["red"]["pcc"].append(pcc)
            data["red"]["contour"].append(props)

            centroid = props.centroid
            data["red"]["coordinates"].append((centroid[0], centroid[1]))

        for props in green_contours:
            green_cropped = self.crop_image(green_filled, props)
            red_cropped = self.crop_image(red_filled, props)
            pcc = self.get_pearson(green_cropped, red_cropped)
            data["green"]["pcc"].append(pcc)
            data["green"]["contour"].append(props)

            centroid = props.centroid
            data["green"]["coordinates"].append((centroid[0], centroid[1]))

        self.figure_pcc(red_filled, data["red"])

        self.figure_pcc(green_filled, data["green"])

        # Create red and green masks with the same shape as the original image
        red_mask = np.zeros_like(img)
        green_mask = np.zeros_like(img)

        for props in red_contours:
            rr, cc = skd.polygon(props.coords[:, 0], props.coords[:, 1])
            red_mask[rr, cc, 0] = red[rr, cc]

        for props in green_contours:
            rr, cc = skd.polygon(props.coords[:, 0], props.coords[:, 1])
            green_mask[rr, cc, 1] = green[rr, cc]

        self.figure_pcc(red_mask, data["red"])
        self.figure_pcc(green_mask, data["green"])

        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()
