class Metrics:
    @staticmethod
    def get_distance_from_center(img, fit):
        import numpy as np
        # image dimensions
        img_height, img_width = img.shape[0], img.shape[1]
        # pixel to meter factor
        x_meter_per_pixel = 3.7 / (img_height - 20)
        # camera is mounted at the center of the car
        car_position = img_width / 2
        # left and right polynomial fits
        right_fitx, left_fitx = fit
        # lane width in which car is being driven
        lane_width = abs(left_fitx - right_fitx)
        # lane center is the midpoint at the bottom of the image
        lane_center = (left_fitx + right_fitx) / 2
        # how much car is away from lane center
        center_distance = (car_position - lane_center) * x_meter_per_pixel

        return center_distance[2], abs(lane_width[2])

    @staticmethod
    def get_curvature_radius(img, left, right):
        import numpy as np

        img_height = img.shape[0]
        ploty = np.linspace(0, img_height - 1, img_height)
        y = np.max(ploty)

        y_meter_per_pixel = 30 / img_height
        x_meter_per_pixel = 3.7 / (img_height - 20)

        rightx, righty = right
        leftx, lefty = left

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
