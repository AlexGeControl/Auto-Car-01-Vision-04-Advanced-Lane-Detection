# Set up session:
import argparse

import numpy as np
import cv2

from .utils import get_left_and_right_lane_masks
from .utils import are_both_lane_lines_detected, get_params

# Sklearn transformer interface:
from sklearn.base import TransformerMixin

class Analyzer(TransformerMixin):
    def __init__(
        self,
        window_size,
        offset,
        meter_per_pixel,
    ):
        self.window_size = window_size
        self.offset = offset
        self.meter_per_pixel = meter_per_pixel

    def transform(self, X):
        """ Binarize input image
        """
        # Get left & right lane line masks:
        lane_masks = get_left_and_right_lane_masks(
            X,
            self.window_size,
            self.offset
        )

        if are_both_lane_lines_detected(X, lane_masks):
            return get_params(
                X,
                lane_masks,
                self.meter_per_pixel
            )

        return None

    def fit(self, X, y=None):
        """ Do nothing
        """
        return self

    def set_params(self, **kwargs):
        self.__dict__.update(kwargs)

if __name__ == "__main__":
    # Parse command-line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input image filename."
    )
    args = vars(parser.parse_args())

    binary = cv2.imread(args["input"])[:,:,0]
    binary[binary > 0] = 1

    analyzer = Analyzer(
        [81, 80],
        120,
        [0.00578125, 0.04166666]
    )

    result = analyzer.transform(binary)

    if result is None:
        print("[Analyzer]: Failed to attain params.")
    else:
        (
            (left_lane_line_params, right_lane_line_params),
            (left_curverad, right_curverad),
            offset
        ) = result

        print(left_lane_line_params)

        print(right_lane_line_params)

        print(
            "Radius of Curvature: ({:.1f}, {:.1f})m".format(
                left_curverad,
                right_curverad
            )
        )

        print(
            "Vehicle is {:.2f}m {} of center".format(
                abs(offset),
                "left" if offset < 0 else "right",
            )
        )
