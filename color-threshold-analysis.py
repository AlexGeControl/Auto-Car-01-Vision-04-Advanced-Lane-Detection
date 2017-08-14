# Set up session:
import argparse

from os.path import basename
import json

import numpy as np
import cv2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA

def get_region_mask(image):
    """ Generate quadrilateral region mask for lane detection
    """
    H, W = image.shape

    return cv2.fillPoly(
        # Black canvas:
        np.zeros((H, W), dtype=np.uint8),
        # Quadrilateral region mask:
        np.array(
            [
                [
                    (0, H-1),
                    (int(0.46 * W), int(0.59 * H)),
                    (int(0.54 * W), int(0.59 * H)),
                    (W-1, H-1)
                ]
            ],
            dtype=np.int
        ),
        # White for further bitwise operation:
        255
    )

def get_annotations(annotation_filename):
    # Load annotations:
    with open(annotation_filename) as annotation_file:
        annotations = json.load(annotation_file)

    pixel_values = []
    gradient_values = []
    for annotation in annotations:
        # Pixel coordinates:
        coords = np.array(
            [
                [record["y"], record["x"]] for record in annotation["annotations"]
            ],
            dtype=np.int
        )

        # Image
        image = cv2.imread(annotation["filename"])
        hls = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV
        )

        # Gradient magnitude image:
        grayscale = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
        grad_x = np.abs(
            cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, 3)
        )
        grad_y = np.abs(
            cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, 3)
        )
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_mag = grad_x
        grad_mask = get_region_mask(grad_mag)
        grad_mag[grad_mask == 0] = 0
        grad_mag[
            ~(
                (19 <= hls[:, :, 0]) & (hls[:, :, 0] <= 23) |
                (170 <= hls[:, :, 2]) & (hls[:, :, 2] <= 255)
            )
        ] = 0
        grad_mag_scaled = (255 * grad_mag / np.max(grad_mag)).astype(np.uint8)
        grad_mag_scaled[
            (16 <= grad_mag_scaled) & (grad_mag_scaled <= 128)
        ] = 255

        cv2.imwrite(
            "grad-{}".format(basename(annotation["filename"])),
            np.dstack(
                tuple([grad_mag_scaled]*3)
            )
        )

        pixel_values.append(
            hls[coords[:, 0], coords[:, 1], :].reshape(
                (-1, 3)
            )
        )
        gradient_values.append(
            grad_mag_scaled[coords[:, 0], coords[:, 1]]
        )

    pixel_values = np.vstack(
        tuple(pixel_values)
    ).reshape(
        (-1, 3)
    )

    pca = PCA(n_components=3)
    pca.fit(pixel_values)

    print(pca.components_)
    print(pca.explained_variance_ratio_)

    counts, edges = np.histogram(pixel_values[:, 0], bins=360)
    idx = np.argmax(counts)
    print(edges[idx], edges[idx + 1])

    counts, edges = np.histogram(pixel_values[:, 1], bins=360)
    idx = np.argmax(counts)
    print(edges[idx], edges[idx + 1])

    counts, edges = np.histogram(pixel_values[:, 2], bins=360)
    idx = np.argmax(counts)
    print(edges[idx], edges[idx + 1])

    gradient_values

    return (pixel_values, gradient_values)


if __name__ == "__main__":
    # Parse command-line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-y", "--yellow",
        type=str,
        required=True,
        help="Yellow lane annotation filename."
    )
    parser.add_argument(
        "-w", "--white",
        type=str,
        required=True,
        help="White lane annotation filename."
    )
    args = vars(parser.parse_args())

    (pixels_yellow, grads_yellow) = get_annotations(args["yellow"])
    (pixels_white, grads_white) = get_annotations(args["white"])

    pixel_values = np.vstack(
        (pixels_yellow, pixels_white)
    )
    colors = ['r'] * len(pixels_yellow) + ['b'] * len(pixels_white)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(
        pixel_values[:, 0],
        pixel_values[:, 1],
        pixel_values[:, 2],
        c = colors
    )

    ax.set_xlabel('H')
    ax.set_ylabel('S')
    ax.set_zlabel('V')

    plt.show()
