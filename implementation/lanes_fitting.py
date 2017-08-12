from visualization import Visualization


class LanesFitting:
    @staticmethod
    def get_lanes_fit(img):
        """
        Using Histogram and Window Sliding Algorithm
        :param img: binary warped image
        :return: lanes_img: image with detected lanes drawn
        :return: lanes_fit: polynomial fit for left and right lanes
        :return: left_xy: xy positions for left lane in pixel
        :return: right_xy: xy positions for right lane in pixel
        """
        import numpy as np
        import cv2

        # Take a histogram of the bottom half of the image
        histogram = np.sum(img[np.int(img.shape[0] / 2):, :], axis=0)

        # Create an output image to draw on and visualize the result
        lanes_img = np.dstack((img, img, img)) * 255

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        n_windows = 9

        # Set height of windows
        window_height = np.int(img.shape[0] / n_windows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzero_x, nonzero_y = np.array(nonzero[1]), np.array(nonzero[0])

        # Current positions to be updated for each window
        leftx_current, rightx_current = leftx_base, rightx_base

        # Set the width of the windows +/- margin
        margin = 100

        # Set minimum number of pixels found to recenter window
        min_pixels = 50

        left_lane_inds, right_lane_inds = [], []

        for window in range(n_windows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height

            win_xleft_low, win_xleft_high = leftx_current - margin, leftx_current + margin
            win_xright_low, win_xright_high = rightx_current - margin, rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(lanes_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(lanes_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (
                nonzero_x < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (
                nonzero_x < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds), right_lane_inds.append(good_right_inds)

            # If you found > min_pixels pixels, recenter next window on their mean position
            if len(good_left_inds) > min_pixels:
                leftx_current = np.int(np.mean(nonzero_x[good_left_inds]))
            if len(good_right_inds) > min_pixels:
                rightx_current = np.int(np.mean(nonzero_x[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        left_x, left_y = nonzero_x[left_lane_inds], nonzero_y[left_lane_inds]
        right_x, right_y = nonzero_x[right_lane_inds], nonzero_y[right_lane_inds]

        # Fit a second order polynomial to each lane
        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)

        # indices, lanes, positions as tuples
        indices = left_lane_inds, right_lane_inds
        lanes_fit = left_fit, right_fit
        nonzero = nonzero_x, nonzero_y
        right_xy = right_x, right_y
        left_xy = left_x, left_y

        lanes_img = Visualization.visualize_lanes_fit(img, lanes_img, nonzero, indices, lanes_fit)

        return lanes_img, lanes_fit, left_xy, right_xy

    @staticmethod
    def update_lanes_fit(img, fit):
        """
         Using Histogram and Window Sliding Algorithm
         Using the previously calculated Lanes Fit to avoid blind search
         Searching in a margin around previous Lanes position
         """
        import numpy as np

        left_fit, right_fit = fit

        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = (
            (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
                nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = (
            (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
                nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # indices, lanes, positions as tuples
        indices = left_lane_inds, right_lane_inds
        fit = right_fit, left_fit
        nonzero = nonzerox, nonzeroy
        right_xy = rightx, righty
        left_xy = leftx, lefty

        # image to draw the lanes line on
        lanes_img = np.dstack((img, img, img)) * 255
        lanes_img = Visualization.visualize_updated_lanes_fit(img, lanes_img, nonzero, indices, fit)

        return lanes_img, fit, left_xy, right_xy
