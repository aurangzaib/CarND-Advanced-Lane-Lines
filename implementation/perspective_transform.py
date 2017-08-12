class PerspectiveTransform:
    @staticmethod
    def get_perspective_points(img, offset=100):
        """
        get the source and destination points required for perspective transform
        source points cover the lane in the road
        destination points forms a bird eye view for lane detection
        :param img: binary image after preprocessing
        :param offset: margin
        :return: source and destination points
        """
        import numpy as np

        # y tilt --> img_height / 2 + offset
        # x tilt --> spacing between both lanes
        x_tilt, y_tilt = 55, 450
        img_height, img_width = img.shape[0], img.shape[1]
        img_center = (img_width / 2)

        # covers the lane in the road
        src = np.float32([
            [offset, img_height],
            [img_center - x_tilt, y_tilt],
            [img_center + x_tilt, y_tilt],
            [img_width - offset, img_height]
        ])

        # forms a bird eye
        dst = np.float32([
            [offset, img_width],
            [offset, 0],
            [img_height - offset, 0],
            [img_height - offset, img_width]
        ])

        return src, dst

    @staticmethod
    def get_sample_wrapped_images(img, src, dst):
        """
        draw source and destination on a sample image for visualization.
        apply perspective transform on destination image.
        draw source and destination points on respective images
        :param img: source image
        :param src: source points
        :param dst: destination points
        :return: images with source and destination points drawn
        """
        import numpy as np
        import cv2 as cv

        # dimensions of the image
        img_height, img_width = img.shape[0], img.shape[1]

        # transformation matrix
        transform_matrix = cv.getPerspectiveTransform(src, dst)
        img_src = np.copy(img)

        # apply perspective transform
        img_dst = cv.warpPerspective(img, transform_matrix,
                                     (img_height, img_width),
                                     flags=cv.INTER_LINEAR)

        # in polyfill suitable format
        src_pts = np.array(src, np.int32).reshape((-1, 1, 2))
        dst_pts = np.array(dst, np.int32).reshape((-1, 1, 2))

        # draw source and destination points on the images
        cv.polylines(img_src, [src_pts], True, (255, 0, 0), thickness=5)
        cv.polylines(img_dst, [dst_pts], True, (255, 0, 0), thickness=5)

        return img_src, img_dst

    @staticmethod
    def get_wrapped_image(img, src, dst):
        """
        apply perspective transform on an image using transform matrix (M)
        :param img: binary image from preprocessing
        :param src: source points
        :param dst: destination points
        :return: warped image with perspective transform applied
        """
        import cv2 as cv
        # get the transform matrix
        transform_matrix = cv.getPerspectiveTransform(src, dst)
        # apply perspective tranform
        transformed_image = PerspectiveTransform.wrap(img, transform_matrix)
        return transformed_image

    @staticmethod
    def get_unwrapped_image(img, transformed_image, src, dst, fit):
        """
        translate the detection lanes in warped image back to the source image
        :param img: source image
        :param transformed_image: warped binary image
        :param src: source points
        :param dst: destination points
        :param fit: polynomial fit for lanes
        :return:
        """
        import cv2 as cv
        # get the inverse transform matrix
        inv_transform_matrix = cv.getPerspectiveTransform(dst, src)
        inv_transformed_image = PerspectiveTransform.unwrap(img, transformed_image, inv_transform_matrix, fit)
        return inv_transformed_image

    @staticmethod
    def wrap(img, transform_matrix):
        """
        :param img: binary image
        :param transform_matrix: transformation matrix (M)
        :return:  warped binary image
        """
        import numpy as np
        import cv2 as cv
        transformed_image = np.copy(img)
        img_height, img_width = img.shape[0], img.shape[1]
        return cv.warpPerspective(transformed_image, transform_matrix,
                                  (img_height, img_width),
                                  flags=cv.INTER_LINEAR)

    @staticmethod
    def unwrap(img, transformed_image, inv_transform_matrix, fit):
        """
        :param img: source image
        :param transformed_image: warped binary image
        :param inv_transform_matrix: inverse transformation matrix (Minv)
        :param fit: polynomial fit for lanes
        :return: unwarped 3 channel image
        """
        import numpy as np
        import cv2 as cv
        # polynomial fit for left and right lanes
        right_fit, left_fit = fit

        # generate x and y values for plotting
        ploty = np.linspace(0, transformed_image.shape[0] - 1, transformed_image.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(transformed_image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        new_warp = cv.warpPerspective(color_warp, inv_transform_matrix, (img.shape[1], img.shape[0]))

        # Combine the result with the original image
        result = cv.addWeighted(img, 1, new_warp, 0.3, 0)

        return result
