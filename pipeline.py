def save_parameters(camera_matrix, dist_coef, corners, filename="calibration_parameters.p"):
    """
    save the matrix and coef in a pickle file
    """
    import pickle
    parameters = {
        'camera_matrix': camera_matrix,
        'dist_coef': dist_coef,
        'corners': corners
    }
    pickle.dump(parameters, open(filename, "wb"))
    print("data saved to disk")


def load_parameters(filename="calibration_parameters.p"):
    """
    load pickle files for train, validation and test
    """
    import pickle
    with open(filename, mode='rb') as f:
        parameters = pickle.load(f)
    return parameters['camera_matrix'], parameters['dist_coef']


def get_undistorted_image(nx, ny, img, load_params=True):
    camera_matrix, dist_coef = load_parameters() if load_params else get_calibration_parameters(nx, ny)
    undistorted = undistort(camera_matrix, dist_coef, img)
    return undistorted


def get_calibration_parameters(nx, ny, channels=3):
    import matplotlib.image as mpimg
    import cv2 as cv
    import numpy as np
    import glob
    imgs = glob.glob("camera_cal/*.jpg")
    """
    find the corners of the image using cv.findChessboardCorners
    find camera matrix and distortion coef. using cv.calibrateCamera with corners and pattern size as arguments
    undistort the image using cv.undistort with camera matrix and distortion coef. as arguments
    """
    # img_pts --> 2D in image
    # obj_pts --> 3D in real world
    img_pts, obj_pts, = [], []
    # to create a matrix of 4x5 --> np.mgrid[0:4, 0:5]
    obj_pt = np.zeros(shape=(nx * ny, channels), dtype=np.float32)
    obj_pt[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    for file_name in imgs:
        img = mpimg.imread(file_name)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # find the corners
        found, corners = cv.findChessboardCorners(image=gray, patternSize=(nx, ny))
        if found is True:
            obj_pts.append(obj_pt)
            img_pts.append(corners)
            # draw the found corner points in the image
            draw_pts = np.copy(img)
            cv.drawChessboardCorners(image=draw_pts, patternSize=(nx, ny), corners=corners, patternWasFound=found)
    # test image
    test_img = mpimg.imread("camera_cal/calibration4.jpg")
    # find camera matrix and distortion coef.
    ret, camera_matrix, dist_coef, rot_vector, trans_vector = cv.calibrateCamera(objectPoints=obj_pts,
                                                                                 imagePoints=img_pts,
                                                                                 imageSize=test_img.shape[0:2],
                                                                                 cameraMatrix=None,
                                                                                 distCoeffs=None)
    save_parameters(camera_matrix, dist_coef, corners)
    return camera_matrix, dist_coef


def undistort(camera_matrix, dist_coef, img):
    import cv2 as cv
    # undistort the image
    undistorted = cv.undistort(src=img,
                               cameraMatrix=camera_matrix,
                               distCoeffs=dist_coef,
                               dst=None,
                               newCameraMatrix=camera_matrix)
    return undistorted


def color_gradient_threshold(img):
    import numpy as np
    import cv2 as cv

    # grayscale
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    gray_binary = np.zeros_like(gray)
    gray_binary[(gray >= 20) & (gray <= 80)] = 1

    # sobelx gradient threshold
    sx_thresh = (20, 200)
    dx, dy = (1, 0)
    sx = cv.Sobel(gray, cv.CV_64F, dx, dy, ksize=3)
    sx_abs = np.absolute(sx)
    sx_8bit = np.uint8(255 * sx_abs / np.max(sx_abs))
    sx_binary = np.zeros_like(sx_8bit)
    sx_binary[(sx_8bit > sx_thresh[0]) & (sx_8bit <= sx_thresh[1])] = 1

    # RGB color space
    rgb_thresh = (170, 255)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    r_binary = np.zeros_like(r)
    r_binary[(r >= rgb_thresh[0]) & (r <= rgb_thresh[1])] = 1

    # HLS color space
    hls_thresh = (120, 255)
    hls = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    h, l, s = hls[:, :, 0], hls[:, :, 1], hls[:, :, 2]
    s_binary = np.zeros_like(s)
    s_binary[(s >= hls_thresh[0]) & (s <= hls_thresh[1])] = 1

    # resultant of r, s and sx
    resultant = np.zeros_like(sx_binary)
    resultant[((sx_binary == 1) | (s_binary == 1)) & (r_binary == 1)] = 1

    return resultant


def get_source_destination_points(img, offset=100):
    import numpy as np
    # y tilt --> img_height / 2 + offset
    # x tilt --> spacing between both lanes
    x_tilt, y_tilt = 55, 450
    img_height, img_width = img.shape[0], img.shape[1]
    img_center = (img_width / 2)

    src = np.float32([
        [offset, img_height],
        [img_center - x_tilt, y_tilt],
        [img_center + x_tilt, y_tilt],
        [img_width - offset, img_height]
    ])
    dst = np.float32([
        [offset, img_width],
        [offset, 0],
        [img_height - offset, 0],
        [img_height - offset, img_width]
    ])

    return src, dst


def get_source_destination_images(img, src, dst):
    import numpy as np
    import cv2 as cv

    img_height, img_width = img.shape[0], img.shape[1]
    transform_matrix = cv.getPerspectiveTransform(src, dst)
    img_src = np.copy(img)
    img_dst = cv.warpPerspective(img, transform_matrix,
                                 (img_height, img_width),
                                 flags=cv.INTER_LINEAR)

    src_pts = np.array(src, np.int32).reshape((-1, 1, 2))
    dst_pts = np.array(dst, np.int32).reshape((-1, 1, 2))

    cv.polylines(img_src, [src_pts], True, (255, 0, 0), thickness=5)
    cv.polylines(img_dst, [dst_pts], True, (255, 0, 0), thickness=5)

    return img_src, img_dst


def perspective_transform(img, src, dst):
    import cv2 as cv
    img_height, img_width = img.shape[0], img.shape[1]
    # use cv2.getPerspectiveTransform() to get M, the transform matrix
    transform_matrix = cv.getPerspectiveTransform(src, dst)
    # use cv2.warpPerspective() to warp your image to a top-down view
    transformed_image = cv.warpPerspective(img, transform_matrix,
                                           (img_height, img_width),
                                           flags=cv.INTER_LINEAR)
    return transformed_image


def inverse_perpective_transform(img, nx, ny):
    import cv2 as cv
    img_height, img_width = img.shape[0], img.shape[1]
    src, dst = get_source_destination_points(img, img_height, img_width)
    # use cv2.getPerspectiveTransform() to get M, the transform matrix
    transform_matrix = cv.getPerspectiveTransform(dst, src)
    # use cv2.warpPerspective() to warp your image to a top-down view
    transformed_image = cv.warpPerspective(img, transform_matrix, (img_height, img_width), flags=cv.INTER_LINEAR)
    return transformed_image


def histogram_sliding_window(img):
    import numpy as np
    import cv2

    # Assuming you have created a warped binary image called "img"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[np.int(img.shape[0] / 2):, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((img, img, img)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9

    # Set height of windows
    window_height = np.int(img.shape[0] / nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100

    # Set minimum number of pixels found to recenter window
    minpix = 50

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Reverse to match top-to-bottom in y
    # leftx = leftx[::-1]
    # rightx = rightx[::-1]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    inds = left_lane_inds, right_lane_inds
    fit = right_fit, left_fit
    left = leftx, lefty
    right = rightx, righty
    nonzero = nonzerox, nonzeroy

    visualize_sliding_window(img, out_img, nonzero, inds, fit)

    return out_img, fit, left, right


def visualize_sliding_window(img, out_img, nonzero, inds, fit):
    import matplotlib.pyplot as plt
    import numpy as np

    # Generate x and y values for plotting
    img_height = img.shape[0]
    ploty = np.linspace(0, img_height - 1, img_height)
    right_fit, left_fit = fit
    left_lane_inds, right_lane_inds = inds
    nonzerox, nonzeroy = nonzero

    # using Ay^2 + By + C
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.show()
    return out_img


def visualization(img1, img2, img3, img4, img5, img6, img7, img8,
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
    # plt.pause(0.05)


def visualize_lanes_histogram(img):
    import matplotlib.pyplot as plt
    import numpy as np
    histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
    plt.plot(histogram)


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

    return center_distance[2], lane_width[2]


def __main__():
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import glob
    nx, ny, channels = 9, 6, 3
    imgs = glob.glob("test_images/*.jpg")
    # Create empty lists to receive left and right lane pixel indices
    for file_name in imgs:
        img = mpimg.imread(file_name)
        # calibrate camera and undistort the image
        undistorted_image = get_undistorted_image(nx, ny, img)
        # get the color and gradient threshold image
        thresholded_image = color_gradient_threshold(undistorted_image)
        # get source and destination points
        src, dst = get_source_destination_points(img)
        # get image with source and destination points drawn
        img_src, img_dst = get_source_destination_images(img, src, dst)
        # perspective transform to bird eye view
        transformed_image = perspective_transform(thresholded_image, src, dst)
        # find the lanes lines and polynomial fit
        lane_lines, fit, left, right = histogram_sliding_window(transformed_image)
        # find the radius of curvature
        radius = get_curvature_radius(lane_lines, left, right)
        # find the car distance from center lane
        center_distance, lane_width = get_distance_from_center(lane_lines, fit)
        # visualize the results
        visualization(img, undistorted_image, thresholded_image,
                      img_src, img_dst,
                      transformed_image, lane_lines, img,
                      "Original", "Undistorted", "Thresholded",
                      "Source Points", "Destination Points",
                      "Transformed", "Lane Line", "Original")

        print("left radius: {}, right radius: {}".format(radius[0], radius[1]))
        print("distance from center: {}".format(center_distance))
        print("lane width: {}".format(lane_width))

        # while True:
        #     plt.pause(1)


__main__()
