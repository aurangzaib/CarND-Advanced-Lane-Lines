import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


class Visualization:
    @staticmethod
    def visualize_lanes_fit(img, lanes_img, nonzero, inds, fit):
        """
        visualize the lanes fit
        :param img: source warped binary image
        :param lanes_img: destination warped binary image with lanes drawn
        :param nonzero: nonzero pixels
        :param inds: indices of the nonzero xy pixels
        :param fit: left and right fit
        :return lanes_img: destination warped binary image with lanes drawn
        """
        # Generate x and y values for plotting
        left_lane_inds, right_lane_inds = inds
        nonzero_x, nonzero_y = nonzero

        # Color in left and right line pixels
        lanes_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
        lanes_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]

        return lanes_img

    @staticmethod
    def visualize_updated_lanes_fit(img, lanes_img, nonzero, inds, fit):
        """
        visualize updated lanes fit
        :param img: source warped binary image
        :param lanes_img: destination warped binary image with lanes drawn
        :param nonzero: nonzero pixels
        :param inds: indices of the nonzero xy pixels
        :param fit: left and right fit
        :return lanes_img: destination warped binary image with lanes drawn
        """
        # Generate x and y values for plotting
        img_height = img.shape[0]
        left_lane_inds, right_lane_inds = inds
        nonzero_x, nonzero_y = nonzero
        left_fit, right_fit = fit
        margin = 100

        # using Ay^2 + By + C
        plot_y = np.linspace(0, img_height - 1, img_height)
        left_fitx = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
        right_fitx = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]

        # Create an image to draw on and an image to show the selection window
        window_img = np.zeros_like(lanes_img)

        # Color in left and right line pixels
        lanes_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
        lanes_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, plot_y]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, plot_y])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))

        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, plot_y]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, plot_y])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))

        lanes_img = cv.addWeighted(lanes_img, 1, window_img, 0.3, 0)

        return lanes_img

    @staticmethod
    def visualize_pipeline_pyplot(img1, img2, img3, img4, img5, img6,
                                  desc_img1, desc_img2, desc_img3, desc_img4,
                                  desc_img5, desc_img6):
        f, ax_array = plt.subplots(2, 3, figsize=(8, 8))
        f.tight_layout()
        ax_array[0, 0].imshow(img1, cmap="gray"), ax_array[0, 0].set_title(desc_img1)
        ax_array[0, 1].imshow(img2, cmap="gray"), ax_array[0, 1].set_title(desc_img2)
        ax_array[0, 2].imshow(img3, cmap="gray"), ax_array[0, 2].set_title(desc_img3)
        ax_array[1, 0].imshow(img4, cmap="gray"), ax_array[1, 0].set_title(desc_img4)
        ax_array[1, 1].imshow(img5, cmap="gray"), ax_array[1, 1].set_title(desc_img5)
        ax_array[1, 2].imshow(img6, cmap="gray"), ax_array[1, 2].set_title(desc_img6)
        # ax_array[1, 2].imshow(img7, cmap="gray"), ax_array[1, 2].set_title(desc_img7)
        # ax_array[1, 3].imshow(img8, cmap="gray"), ax_array[1, 3].set_title(desc_img8)
        plt.show()

    @staticmethod
    def get_lanes_histogram(img):
        histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
        return histogram

    @staticmethod
    def visualize_pipeline(resultant, img_dst,
                           binary_image, lane_lines,
                           radius, center_distance,
                           lane_width):
        """
        visualize the important steps of the pipeline
        :param resultant: resultant image of the pipeline
        :param img_dst: wrapped binary image
        :param binary_image: binary image
        :param lane_lines: wrapped binary image with lane lines
        :param radius: radius of curvature
        :param center_distance: car distance from center lane
        :param lane_width: width of the lane
        :return: None
        """
        # resize the image for better visualization
        resultant = cv.resize(resultant, None, fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
        resultant = cv.cvtColor(resultant, cv.COLOR_BGR2RGB)
        binary_image = cv.cvtColor(binary_image, cv.COLOR_GRAY2BGR)

        # FONT_HERSHEY_SIMPLEX
        font = cv.QT_FONT_NORMAL

        radius_txt = "Radius of Curvature = {}m".format(str(round(radius[0], 3)))
        left_or_right = "left" if center_distance > 0 else "right"
        distance_txt = "Vehicle is {}m {} of center ".format(str(round(abs(center_distance), 2)), left_or_right)

        cv.putText(resultant, radius_txt, (10, 30), font, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        cv.putText(resultant, distance_txt, (10, 60), font, 0.6, (255, 255, 255), 1, cv.LINE_AA)

        # resize the image for better visualization
        img_dst = cv.resize(img_dst, None, fx=0.4, fy=0.3, interpolation=cv.INTER_LINEAR)
        lane_lines = cv.resize(lane_lines, None, fx=0.4, fy=0.3, interpolation=cv.INTER_LINEAR)

        cv.imshow("result", resultant)
        cv.waitKey(1)
