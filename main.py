import cv2
import numpy as np

def locate_note(image, fov, ring_radius, cam_elevation):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for orange color in HSV
    lower_orange = np.array([0, 100, 100])
    upper_orange = np.array([20, 255, 255])

    # Create a binary mask for the orange color
    mask = cv2.inRange(hsv_image, lower_orange, upper_orange)

    # Apply morphological operations to reduce noise
    morphed_image = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    morphed_image = cv2.dilate(morphed_image, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    # Find contours in the binary mask
    contours, _ = cv2.findContours(morphed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on size
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 100]

    leftmost_x = np.inf
    rightmost_x = -np.inf

    for contour in filtered_contours:
        for point in contour[:, 0]:
            x = point[0]

            if x < leftmost_x:
                leftmost_x = x

            if x > rightmost_x:
                rightmost_x = x

    # Refine contours based on the shape of the note
    refined_contours = []

    for contour in filtered_contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx_curve = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx_curve) == 4:
            refined_contours.append(approx_curve)

    refined_center_x = 0.0
    refined_leftmost_x = np.inf
    refined_rightmost_x = -np.inf

    if refined_contours:
        for contour in refined_contours:
            for point in contour[:, 0]:
                x = point[0]

                if x < refined_leftmost_x:
                    refined_leftmost_x = x

                if x > refined_rightmost_x:
                    refined_rightmost_x = x

        refined_center_x = (refined_leftmost_x + refined_rightmost_x) / 2.0

    left_x = int(leftmost_x)
    right_x = int(rightmost_x)

    center_offset = (left_x - right_x) / 2

    # Draw polygons on the image for visualization
    rgb = cv2.cvtColor(morphed_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(rgb, refined_contours, -1, (0, 255, 0), 2)

    cv2.line(rgb, (left_x, 0), (left_x, image.shape[0]), (0, 0, 255), 1)
    cv2.line(rgb, (right_x, 0), (right_x, image.shape[0]), (0, 0, 255), 1)

    cv2.line(rgb, (0, image.shape[0] // 2), (image.shape[1], image.shape[0] // 2), (255, 0, 0), 1)
    cv2.line(rgb, (image.shape[1] // 2, 0), (image.shape[1] // 2, image.shape[0]), (255, 0, 0), 1)

    cv2.line(rgb, (80 - center_offset, 0), (80 - center_offset, image.shape[0]), (0, 255, 0), 1)
    cv2.line(rgb, (80 + center_offset, 0), (80 + center_offset, image.shape[0]), (0, 255, 0), 1)

    final_distance = 0.0
    final_theta = 0.0

    internal_angle_offset = (((refined_leftmost_x + refined_rightmost_x) / 2 - refined_leftmost_x) * 2 * fov) / image.shape[1]
    hyp_distance = ring_radius / np.tan(internal_angle_offset)

    if hyp_distance > cam_elevation:
        n_d = np.sqrt(hyp_distance ** 2 - cam_elevation ** 2)
        n_theta = (fov / image.shape[1]) * ((refined_leftmost_x + refined_rightmost_x - image.shape[1]) / 2)

        if n_theta <= 2:
            final_distance = n_d
            final_theta = n_theta

    return final_distance, final_theta
