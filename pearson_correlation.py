import cv2
import numpy as np
import skimage as ski

class ImageProcessing:

    def __init__(self):
        self.windows = []

    # Define a function to fill holes in the image and return the contours
    def fill_image(self, channel):
        # blurred = ski.filters.gaussian(channel, sigma=1.0).astype(np.uint8)
        _, thresh = cv2.threshold(channel, 1, 255, cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        filled_image = np.zeros_like(channel)
        cv2.drawContours(filled_image, contours, -1, 255, thickness=cv2.FILLED)
        return filled_image, contours

    def get_pearson(self, image_1, image_2):
        mask = np.zeros_like(image_1)
        red_figure = cv2.bitwise_and(image_1, image_1, mask=mask)
        green_figure = cv2.bitwise_and(image_2, image_2, mask=mask)

        # Compute Pearson correlation coefficient
        pcc = np.corrcoef(red_figure.ravel(), green_figure.ravel())[0, 1]
        print(pcc)
        return pcc

    def plot_pcc(self, img, window_name, data):
        self.windows.append(window_name)

        cv2.imshow(window_name, img)
        for pcc, coordinate in zip(data['pcc'], data['coordinates']):
            cv2.putText(
                img,
                f"{pcc:.2f}",
                (coordinate[0], coordinate[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

    def wait_close(self, wait_time: int = 1000):
        while any(
            cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1
            for window in self.windows
        ):
            keyCode = cv2.waitKey(wait_time)
            if (keyCode & 0xFF) == ord("q"):
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    data = {
        "green": {"coordinates": [], "pcc": []},
        "red": {"coordinates": [], "pcc": []},
    }
    processor = ImageProcessing()
    # Read the image
    img = cv2.imread("celula3.tif")

    # Check if the image is sufficiently large
    if img.shape[0] * img.shape[1] < 500 * 500:  # Adjust the threshold as needed
        print("Image is too small. Skipping processing.")
        exit()

    # Split the image into channels
    blue, green, red = cv2.split(img)

    # Fill holes in the red and green channels and get contours
    red_filled, red_contours = processor.fill_image(red)
    green_filled, green_contours = processor.fill_image(green)

    # Filter contours based on size (area)
    min_contour_area = 1000  # Adjust as needed
    red_contours = [
        cnt for cnt in red_contours if cv2.contourArea(cnt) > min_contour_area
    ]
    green_contours = [
        cnt for cnt in green_contours if cv2.contourArea(cnt) > min_contour_area
    ]

    # Fill different figures using contours and compute Pearson correlation coefficient for each
    for contour in red_contours:
        pcc = processor.get_pearson(red_filled, green_filled)
        data["red"]["pcc"].append(pcc)

        M = cv2.moments(contour)

        data["red"]["coordinates"].append(
            (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        )

    for contour in green_contours:
        pcc = processor.get_pearson(green_filled, red_filled)
        data["green"]["pcc"].append(pcc)

        M = cv2.moments(contour)

        data["green"]["coordinates"].append(
            (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        )

    processor.plot_pcc(red_filled, "Filled Red Figures", data["red"])

    processor.plot_pcc(green_filled, "Filled Green Figures", data["green"])

    green_exp = np.repeat(np.expand_dims(green_filled, axis=-1), 3, axis=-1)
    green_masked = cv2.bitwise_and(green_exp, img)
    processor.plot_pcc(green_masked, "RGB GREEN MASKED", data["green"])

    red_exp = np.repeat(np.expand_dims(red_filled, axis=-1), 3, axis=-1)
    red_masked = cv2.bitwise_and(red_exp, img)
    processor.plot_pcc(red_masked, "RGB RED MASKED", data["green"])

    processor.wait_close()
