# Set up session:
import numpy as np
import cv2

def get_left_and_right_centers(
    binary,
    window,
    left_bounding_box,
    right_bounding_box
):
    # Parse window size:
    window_length = len(window)

    # Parse left & right search ranges:
    left_top, left_bottom, left_left, left_right = left_bounding_box
    right_top, right_bottom, right_left, right_right = right_bounding_box

    # Histograms:
    left_histogram = binary[
        left_top:left_bottom,
        left_left:left_right
    ].sum(axis = 0)
    right_histogram = binary[
        right_top:right_bottom,
        right_left:right_right
    ].sum(axis = 0)

    # Identify peaks:
    left_stats = np.convolve(left_histogram, window, 'valid')
    left_center = np.argmax(left_stats)
    left_maximum = left_stats[left_center]

    right_stats = np.convolve(right_histogram[::-1], window, 'valid')
    right_center = np.argmax(right_stats)
    right_maximum = right_stats[right_center]

    # Transform to pixel coordinates:
    left_center = left_left + left_center + window_length // 2
    right_center = right_right - 1 - right_center - window_length // 2

    return (
        (left_center, left_maximum),
        (right_center, right_maximum)
    )

def get_left_and_right_lane_masks(
    binary,
    window_size,
    offset
):
    # Image dimensions:
    (H, W) = binary.shape

    # Parse window configuration:
    window_length, v_step_size = window_size
    window = np.ones(window_length)

    # Initialize container:
    bounding_boxes = []

    # Initialize centers:
    (
        (left_center, left_center_value),
        (right_center, right_center_value)
    ) = get_left_and_right_centers(
        binary,
        window,
        (0, H, 0, W // 2),
        (0, H, W // 2, W)
    )

    # Detect in subsequent windows:
    for top in range(H - v_step_size, -1, -v_step_size):
        (
            (left_center_detected, left_center_value),
            (right_center_detected, right_center_value)
        ) = get_left_and_right_centers(
            binary,
            window,
            (
                top,
                (top + v_step_size),
                max(0, left_center - window_length // 2 - offset),
                min(left_center + window_length // 2 + offset, W // 2)
            ),
            (
                top,
                (top + v_step_size),
                max(W // 2, right_center - window_length // 2 - offset),
                min(right_center + window_length // 2 + offset, W)
            )
        )

        # Use previous centers if maximum value is 0:
        left_center = left_center if left_center_value == 0 else left_center_detected
        right_center = right_center if right_center_value == 0 else right_center_detected

        bounding_boxes.append(
            (
                (top, top + v_step_size, left_center - window_length // 2, left_center + window_length // 2),
                (top, top + v_step_size, right_center - window_length // 2, right_center + window_length // 2)
            )
        )

    # Initialize:
    left_lane_mask = np.zeros_like(binary)
    right_lane_mask = np.zeros_like(binary)

    for (left_bounding_box, right_bounding_box) in bounding_boxes:
        left_top, left_bottom, left_left, left_right = left_bounding_box
        cv2.rectangle(
            left_lane_mask,
            (left_left, left_top),
            (left_right, left_bottom),
            1,
            -1
        )

        right_top, right_bottom, right_left, right_right = right_bounding_box
        cv2.rectangle(
            right_lane_mask,
            (right_left, right_top),
            (right_right, right_bottom),
            1,
            -1
        )
    return (left_lane_mask, right_lane_mask)

def are_both_lane_lines_detected(
    binary,
    lane_masks
):
    # Parse lane masks:
    (left_lane_mask, right_lane_mask) = lane_masks

    # Whether both left & right lanes could be fitted:
    left_lane_point_count = (binary & left_lane_mask).sum()
    right_lane_point_count = (binary & right_lane_mask).sum()

    if left_lane_point_count < 3 or right_lane_point_count < 3:
        return False

    return True

def get_lane_line_params(
    left_lane,
    right_lane
):
    # Parse lane lines:
    left_lane_y, left_lane_x = left_lane
    right_lane_y, right_lane_x = right_lane

    # Get params:
    left_lane_line_params = np.polyfit(left_lane_y, left_lane_x, 2)
    right_lane_line_params = np.polyfit(right_lane_y, right_lane_x, 2)

    return (left_lane_line_params, right_lane_line_params)

def get_curverads(
    left_lane,
    right_lane,
    meter_per_pixel,
    y_eval
):
    # Parse lane lines:
    left_lane_y, left_lane_x = left_lane
    right_lane_y, right_lane_x = right_lane

    # Parse unit:
    meter_per_pixel_x, meter_per_pixel_y = meter_per_pixel

    # Fit lanes::
    left_lane_w = np.polyfit(
        meter_per_pixel_y * left_lane_y,
        meter_per_pixel_x * left_lane_x,
        2
    )
    right_lane_w = np.polyfit(
        meter_per_pixel_y * right_lane_y,
        meter_per_pixel_x * right_lane_x,
        2
    )
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_lane_w[0] * y_eval * meter_per_pixel_y + left_lane_w[1])**2)**1.5) / np.absolute(2*left_lane_w[0])
    right_curverad = ((1 + (2 * right_lane_w[0] * y_eval * meter_per_pixel_y + right_lane_w[1])**2)**1.5) / np.absolute(2*right_lane_w[0])

    return (left_curverad, right_curverad)

def get_offset(
    lane_line_params,
    meter_per_pixel,
    y_eval,
    ref
):
    # Parse lane line params:
    (left_lane_line_params, right_lane_line_params) = lane_line_params

    # Parse unit:
    meter_per_pixel_x, _ = meter_per_pixel

    # Calculate offset:
    offset = meter_per_pixel_x * (
        (np.polyval(left_lane_line_params, y_eval) + np.polyval(right_lane_line_params, y_eval)) / 2 - ref
    )

    return offset

def get_params(
    binary,
    lane_masks,
    meter_per_pixel
):
    # Parse image dimensions:
    H, W = binary.shape

    # Extract lane line points:
    (left_lane_mask, right_lane_mask) = lane_masks
    left_lane = np.nonzero(binary & left_lane_mask)
    right_lane = np.nonzero(binary & right_lane_mask)

    # Fit lane lines:
    lane_line_params = get_lane_line_params(
        left_lane,
        right_lane
    )

    y_eval = H - 1

    curverads = get_curverads(
        left_lane,
        right_lane,
        meter_per_pixel,
        y_eval
    )
    offset = get_offset(
        lane_line_params,
        meter_per_pixel,
        y_eval,
        640
    )

    return (lane_line_params, curverads, offset)
