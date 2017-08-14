class Metrics:
    @staticmethod
    def get_curvature_radius(img, left, right):
        """
        find the radius of curvature of the track
        the provide track has a radius around 1km
        :param img: warped binary image with detected lane lines drawn
        :param left:  xy pixel positions of left lane
        :param right: xy pixel positions of right lane
        :return: radius of curvature
        """
        import numpy as np

        img_height = img.shape[0]
        # get evenly spaces array over the range of image height
        ploty = np.linspace(0, img_height - 1, img_height)
        y = np.max(ploty)

        # pixel to meter factor
        y_meter_per_pixel = 30 / img_height
        x_meter_per_pixel = 3.7 / (img_height - 20)

        # xy pixel positions for left and right lanes
        rightx, righty = right
        leftx, lefty = left

        # left and right lanes in meter
        left_fit_meter = np.polyfit(lefty * y_meter_per_pixel,
                                    leftx * x_meter_per_pixel, 2)

        right_fit_meter = np.polyfit(righty * y_meter_per_pixel,
                                     rightx * x_meter_per_pixel, 2)

        # using r = ((1+(f')^2)^1.5)/f''
        left_radius = (1 + (2 * left_fit_meter[0] * y * y_meter_per_pixel + left_fit_meter[1]) ** 2) ** (3 / 2)
        left_radius /= np.absolute(2 * left_fit_meter[0])
        right_radius = (1 + (2 * right_fit_meter[0] * y * y_meter_per_pixel + right_fit_meter[1]) ** 2) ** (3 / 2)
        right_radius /= np.absolute(2 * right_fit_meter[0])

        return int(left_radius), int(right_radius)

    @staticmethod
    def get_distance_from_center(img, fit):
        """
        find the distance of the car from the center lane
        get the car position -> center of the image
        get lane width  -> difference of left fit and right fit
        get center lane -> midpoint of left fit and right fit
        get distance from center lane -> (car position - lane center) * (meter per pixel)
        :param img: warped binary image with detected lane lines drawn
        :param fit: lanes polynomial fit
        :return: center_distance: car distance from center
        """
        # image dimensions
        img_height, img_width = img.shape[0], img.shape[1]

        # pixel to meter factor
        x_meter_per_pixel = 3.7 / (img_height - 20)

        # camera is mounted at the center of the car
        car_position = img_width / 2

        # left and right polynomial fits
        right_fit, left_fit = fit

        # lane width in which car is being driven
        lane_width = abs(left_fit - right_fit)

        # lane center is the midpoint at the bottom of the image
        lane_center = (left_fit + right_fit) / 2

        # how much car is away from lane center
        center_distance = (car_position - lane_center) * x_meter_per_pixel

        print("center distance: {}".format(center_distance))
        return center_distance[2], abs(lane_width[2])
