# Set up session:
import argparse

import numpy as np
import cv2

class Painter:
    def __init__(
        self,
        transformer
    ):
        self.transformer = transformer

    def transform(
        self,
        canvas,
        lane_line_params,
        curverads,
        offset
    ):
        # Parse lane line params:
        (left_lane_line_params, right_lane_line_params) = lane_line_params

        # Identify lane region:
        H, W, _ = canvas.shape
        y = np.linspace(0, H-1, H).astype(np.int)
        Y = np.vstack(
            (y**2, y, np.ones_like(y))
        )
        left_lane_line_x = np.dot(left_lane_line_params, Y).astype(np.int)
        right_lane_line_x = np.dot(right_lane_line_params, Y).astype(np.int)

        left_lane_line_points = np.vstack((left_lane_line_x, y)).T
        right_lane_line_points = np.flipud(np.vstack((right_lane_line_x, y)).T)
        lane_line_polygon = np.vstack(
            (left_lane_line_points, right_lane_line_points)
        )

        # Generate overlay:
        overlay = self.transformer.inverse_transform(
            cv2.fillPoly(
                np.zeros((H, W, 3), dtype=np.uint8),
                [lane_line_polygon],
                (0, 255, 0)
            )
        )

        # Add lane region:
        processed = cv2.addWeighted(
            canvas, 1, overlay, 0.3, 0
        )
        # Add curverads:
        (left_curverad, right_curverad) = curverads
        cv2.putText(
            processed,
            "Radius of Curvature: ({:.1f}, {:.1f}) m".format(
                left_curverad, right_curverad
            ),
            (10,45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2,
            cv2.LINE_AA
        )
        # Add offset:
        cv2.putText(
            processed,
            "Vehicle is {:.2f}m {} of center".format(
                abs(offset),"right" if offset < 0 else "left"
            ),
            (10,90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2,
            cv2.LINE_AA
        )

        return processed

    def fit(self, X, y=None):
        """ Do nothing
        """
        return self

    def set_params(self, **kwargs):
        self.__dict__.update(kwargs)
