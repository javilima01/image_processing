import numpy as np
import skimage.io as skio
import skimage.filters as skf
import skimage.measure as skm                       
import skimage.draw as skd
import matplotlib.pyplot as plt
import argparse

class ImageProcessing:

    def __init__(self):
        self.windows = []

    # Define a function to fill holes in the image and return the contours
    def fill_image(self, channel):
        thresh = channel > skf.threshold_otsu(channel)
        filled_image = skm.label(thresh)
        return filled_image, skm.regionprops(filled_image)

    def get_pearson(self, image_1, image_2):
        red_figure = image_1 > 0
        green_figure = image_2 > 0

        # Compute Pearson correlation coefficient
        pcc = np.corrcoef(red_figure.ravel(), green_figure.ravel())[0, 1]
        return pcc

    def figure_pcc(self, img, window_name, data):
        self.windows.append(window_name)
        plt.figure(window_name)
        if img.ndim < 3:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)

        for pcc, coordinate in zip(data["pcc"], data["coordinates"]):
            plt.text(
                coordinate[1],
                coordinate[0],
                f"{pcc:.4f}",
                fontsize=8,
                color="yellow",
                ha="center",
            )

def parse_args():
    parser = argparse.ArgumentParser(description='Program to calculate pearson correlation of images.')

    # Add command line arguments
    parser.add_argument('--iamge', '-i', type=str, required=True, help='Image to analyze')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    img = args.image

    data = {
        "green": {"coordinates": [], "pcc": []},
        "red": {"coordinates": [], "pcc": []},
    }
    processor = ImageProcessing()
    # Read the image
    img = skio.imread(img)

    # Check if the image is sufficiently large
    if img.shape[0] * img.shape[1] < 500 * 500:  # Adjust the threshold as needed
        print("Image is too small. Skipping processing.")
        exit()

    # Split the image into channels
    red = img[:, :, 0]
    green = img[:, :, 1]

    # Fill holes in the red and green channels and get contours
    red_filled, red_contours = processor.fill_image(red)
    green_filled, green_contours = processor.fill_image(green)

    # Filter contours based on size (area)
    min_contour_area = 1000  # Adjust as needed
    red_contours = [props for props in red_contours if props.area > min_contour_area]
    green_contours = [
        props for props in green_contours if props.area > min_contour_area
    ]

    # Fill different figures using contours and compute Pearson correlation coefficient for each
    for props in red_contours:
        print(props.filled_image)
        pcc = processor.get_pearson(red_filled, green_filled)
        data["red"]["pcc"].append(pcc)

        centroid = props.centroid
        data["red"]["coordinates"].append((centroid[0], centroid[1]))

    for props in green_contours:
        pcc = processor.get_pearson(green_filled, red_filled)
        data["green"]["pcc"].append(pcc)

        centroid = props.centroid
        data["green"]["coordinates"].append((centroid[0], centroid[1]))

    processor.figure_pcc(red_filled, "Filled Red Figures", data["red"])

    processor.figure_pcc(green_filled, "Filled Green Figures", data["green"])

    # Create red and green masks with the same shape as the original image
    red_mask = np.zeros_like(img)
    green_mask = np.zeros_like(img)

    for props in red_contours:
        rr, cc = skd.polygon(props.coords[:, 0], props.coords[:, 1])
        red_mask[rr, cc, 0] = red[rr, cc]

    for props in green_contours:
        rr, cc = skd.polygon(props.coords[:, 0], props.coords[:, 1])
        green_mask[rr, cc, 1] = green[rr, cc]

    processor.figure_pcc(red_mask, "RGB RED MASKED", data["red"])
    processor.figure_pcc(green_mask, "RGB GREEN MASKED", data["green"])

    plt.show()
