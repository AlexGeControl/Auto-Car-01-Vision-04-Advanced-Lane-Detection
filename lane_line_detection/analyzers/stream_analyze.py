# Set up session:
from collections import deque

import numpy as np
import cv2

from .analyze import Analyzer

# Sklearn transformer interface:
from sklearn.base import TransformerMixin

class StreamAnalyzer(TransformerMixin):
    def __init__(
        self,
        window_size,
        offset,
        meter_per_pixel,
        temporal_filter_len
    ):
        self.analyzer = Analyzer(
            window_size,
            offset,
            meter_per_pixel
        )
        self.queue = deque()
        self.temporal_filter_len = temporal_filter_len

        self._optimal = np.zeros(9)

    def transform(self, X):
        """ Binarize input image
        """
        # Get params of current frame
        params = self.analyzer.transform(
            X
        )

        if params is None:
            if len(self.queue) > 0:
                self.queue.popleft()
        else:
            # Parse params:
            (
                (
                    left_lane_line_params,
                    right_lane_line_params
                ),
                (
                    left_curverad, right_curverad
                ),
                offset
            ) = params

            if (
                self._is_valid_lane_params(
                    left_lane_line_params,
                    right_lane_line_params
                ) and self._is_valid_curverads(
                    left_curverad,
                    right_curverad
                )
            ):
                # Format as state vector:
                state = np.hstack(
                    (
                        left_lane_line_params,
                        right_lane_line_params,
                        left_curverad,
                        right_curverad,
                        offset
                    )
                )

                # Remove the outdated state:
                if len(self.queue) == self.temporal_filter_len:
                    self.queue.popleft()

                # Append:
                self.queue.append(state)
            else:
                if len(self.queue) > 0:
                    self.queue.popleft()

        # Update:
        if len(self.queue ) > 0:
            self._optimal = np.array(
                self.queue
            ).mean(axis = 0)

        return self._optimal_state_to_params()

    def fit(self, X, y=None):
        """ Do nothing
        """
        return self

    def set_params(self, **kwargs):
        self.__dict__.update(kwargs)

    def _optimal_state_to_params(self):
        return (
            (
                tuple(self._optimal[0:3]),
                tuple(self._optimal[3:6])
            ),
            tuple(self._optimal[6:8]),
            self._optimal[8]
        )

    def _is_valid_lane_params(
        self,
        left_lane_line_params,
        right_lane_line_params
    ):
        top_diff = (
            np.polyval(right_lane_line_params, 0) - np.polyval(left_lane_line_params, 0)
        )
        bottom_diff = (
            np.polyval(right_lane_line_params, 719) - np.polyval(left_lane_line_params, 719)
        )

        ratio = top_diff/ bottom_diff

        return (1/3 <= ratio) and (ratio <= 3)

    def _is_valid_curverads(
        self,
        left_curverad,
        right_curverad
    ):
        ratio = left_curverad / right_curverad

        return (1/4 <= ratio) and (ratio <= 4)
