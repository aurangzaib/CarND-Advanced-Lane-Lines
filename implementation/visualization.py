class Visualization:
    @staticmethod
    def visualize_lanes_fit(img, lanes_img, nonzero, inds, fit):
        # import numpy as np

        # Generate x and y values for plotting
        left_lane_inds, right_lane_inds = inds
        nonzero_x, nonzero_y = nonzero

        # Color in left and right line pixels
        lanes_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
        lanes_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]

        """
        
        Generate Plots in PyPlot:
        
        img_height = img.shape[0]
        left_fit, right_fit = fit

        using Ay^2 + By + C
        plot_y = np.linspace(0, img_height - 1, img_height)
        left_fitx = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
        right_fitx = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]
        
        plt.imshow(out_img)
        plt.plot(left_fitx, plot_y, color='yellow')
        plt.plot(right_fitx, plot_y, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()
        """

        return lanes_img

    @staticmethod
    def visualize_updated_lanes_fit(img, lanes_img, nonzero, inds, fit):
        import numpy as np
        import cv2 as cv
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

        # plt.imshow(lanes_img)
        # plt.plot(left_fitx, plot_y, color='yellow')
        # plt.plot(right_fitx, plot_y, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)

        return lanes_img

    @staticmethod
    def visualization_pipeline(img1, img2, img3, img4, img5, img6, img7, img8,
                               desc_img1, desc_img2, desc_img3, desc_img4,
                               desc_img5, desc_img6, desc_img7, desc_img8):
        import matplotlib.pyplot as plt
        f, ax_array = plt.subplots(2, 4, figsize=(8, 8))
        f.tight_layout()
        ax_array[0, 0].imshow(img1, cmap="gray"), ax_array[0, 0].set_title(desc_img1)
        ax_array[0, 1].imshow(img2, cmap="gray"), ax_array[0, 1].set_title(desc_img2)
        ax_array[0, 2].imshow(img3, cmap="gray"), ax_array[0, 2].set_title(desc_img3)
        ax_array[0, 3].imshow(img4, cmap="gray"), ax_array[0, 3].set_title(desc_img4)
        ax_array[1, 0].imshow(img5, cmap="gray"), ax_array[1, 0].set_title(desc_img5)
        ax_array[1, 1].imshow(img6, cmap="gray"), ax_array[1, 1].set_title(desc_img6)
        ax_array[1, 2].imshow(img7, cmap="gray"), ax_array[1, 2].set_title(desc_img7)
        ax_array[1, 3].imshow(img8, cmap="gray"), ax_array[1, 3].set_title(desc_img8)
        plt.show()
        # visualize the results
        # visualization_pipeline(img, undistorted_image, thresholded_image,
        #               img_src, img_dst,
        #               transformed_image, lane_lines, img,
        #               "Original", "Undistorted", "Thresholded",
        #               "Source Points", "Destination Points",
        #               "Transformed", "Lane Line", "Original")

    @staticmethod
    def get_lanes_histogram(img):
        import numpy as np
        histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
        return histogram

    @staticmethod
    def visualize_pipeline(img, resultant, img_dst, lane_lines, radius, center_distance, lane_width):
        import cv2 as cv
        # debugging parameters
        print("left radius: {}, right radius: {}".format(radius[0], radius[1]))
        print("distance from center: {}".format(center_distance))
        print("lane width: {}".format(lane_width))
        print("\n")

        img = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        resultant = cv.resize(resultant, None, fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
        resultant = cv.cvtColor(resultant, cv.COLOR_BGR2RGB)

        img_dst = cv.resize(img_dst, None, fx=0.4, fy=0.3, interpolation=cv.INTER_LINEAR)
        lane_lines = cv.resize(lane_lines, None, fx=0.4, fy=0.3, interpolation=cv.INTER_LINEAR)

        cv.imshow("input image", img)
        cv.imshow("warped detected lines", lane_lines)
        cv.imshow("warped lines", img_dst)
        cv.imshow("result", resultant)
        cv.waitKey(1)
