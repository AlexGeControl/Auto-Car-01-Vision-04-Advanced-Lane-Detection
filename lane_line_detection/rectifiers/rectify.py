# Set up session:
import argparse

import glob
from os.path import basename, splitext
import pickle

import re

from sys import stderr

import numpy as np
import cv2

import random

# Print iterations progress
def print_progressbar (
    iteration,
    total,
    prefix = '',
    suffix = '',
    decimals = 1,
    length = 100,
    fill = '█'
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    # Calculate percent:
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    # Generate bar:
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    # Display:
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')

# Sklearn transformer interface:
from sklearn.base import BaseEstimator, TransformerMixin

class DistortionRectifier(TransformerMixin):
    """ Distortion rectification transformer for single view camera
    """

    def __init__(
        self,
        images_descriptor
    ):
        self.camera_matrix = None
        self.dist_coeffs = None

        # Get object and image points:
        (obj_points, img_points, img_shape) = self._get_object_and_image_points(
            images_descriptor
        )

        # Calibrate:
        if not img_shape is None:
            status_code, camera_matrix, dist_coeffs, rotations, translations = cv2.calibrateCamera(
                obj_points,
                img_points,
                img_shape,
                None,
                None
            )
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs

    def transform(self, X):
        """ Undistort input image
        """
        return cv2.undistort(
            X,
            self.camera_matrix,
            self.dist_coeffs
        )

    def fit(self, X, y=None):
        """ Estimate camera matrix
        """
        return self

    def set_params(self, **kwargs):
        self.__dict__.update(kwargs)

    def _get_object_and_image_points(self, images_descriptor):
        # Initialize:
        obj_points_list = []
        img_points_list = []
        shape = None

        # Parse calibration dataset:
        pattern_config_parser = re.compile(r'[a-zA-Z]+(\d+)--(\d+)-by-(\d+)')
        image_filenames = glob.glob(images_descriptor)

        for i, image_filename in enumerate(image_filenames):
            # Parse pattern configuration:
            image_name, _ = splitext(basename(image_filename))
            parsed = pattern_config_parser.match(image_name)
            if not parsed is None:
                # Image:
                image = cv2.cvtColor(
                    cv2.imread(image_filename),
                    cv2.COLOR_BGR2GRAY
                )

                # Shape:
                shape = image.shape

                # Pattern config:
                (index, num_x, num_y) = (
                    int(parsed.group(1)),
                    int(parsed.group(2)),
                    int(parsed.group(3))
                )

                # Detect:
                status_code, corners = cv2.findChessboardCorners(image, (num_x, num_y), None)

                # Succeeded:
                if status_code != 0:
                    # Object points--3D:
                    obj_points = np.zeros((num_x*num_y, 3), np.float32)
                    obj_points[:, :2] = np.mgrid[
                        0:num_x,
                        0:num_y
                    ].T.reshape(
                        (-1, 2)
                    )
                    # Image points--2D:
                    cv2.cornerSubPix(
                        image,
                        corners,
                        (11,11),
                        (-1,-1),
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    )
                    img_points = corners

                    obj_points_list.append(obj_points)
                    img_points_list.append(img_points)
                else:
                    print(
                        "[Calibration {:>02}]: Failed--{}".format(
                            index,
                            image_filename
                        ),
                        file=stderr
                    )

                print_progressbar(
                    i + 1,
                    len(image_filenames),
                    prefix = 'Progress:',
                    suffix = 'Complete',
                    length = 50
                )

        return (obj_points_list, img_points_list, shape)

if __name__ == "__main__":
    # Parse command-line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Calibration image files descriptor."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Rectifier output filename."
    )
    args = vars(parser.parse_args())

    rectifier = DistortionRectifier(args["input"])

    # Dump rectifier:
    with open(args["output"], "wb") as rectifier_pkl:
        pickle.dump(rectifier, rectifier_pkl)

    # Load rectifier:
    with open(args["output"], "rb") as rectifier_pkl:
        rectifier = pickle.load(rectifier_pkl)

    image_original = cv2.imread(
        random.choice(
            glob.glob(args["input"])
        )
    )
    cv2.imshow("Original", image_original)
    cv2.waitKey(0)

    image_rectified = rectifier.transform(
        image_original
    )
    cv2.imshow("Rectified", image_rectified)
    cv2.waitKey(0)
