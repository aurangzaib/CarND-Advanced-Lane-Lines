class PreProcessing:
    @staticmethod
    def save_calibration_params(camera_matrix, dist_coef, filename="calibration_parameters.p"):
        """
        save the matrix and coef in a pickle file
        :param camera_matrix: camera matrix found using cv.calibrateCamera
        :param dist_coef: distortion coef found using cv.calibrateCamera
        :param filename: to store the pickle file
        :return: None
        """
        import pickle
        parameters = {
            'camera_matrix': camera_matrix,
            'dist_coef': dist_coef
        }
        pickle.dump(parameters, open(filename, "wb"))
        print("parameters saved to disk")

    @staticmethod
    def load_calibration_params(filename="calibration_parameters.p"):
        """
        load pickle files for train, validation and test
        :param filename: to read the pickle file
        :return: calibration params
        """
        import pickle
        with open(filename, mode='rb') as f:
            parameters = pickle.load(f)
        return parameters['camera_matrix'], parameters['dist_coef']

    @staticmethod
    def get_calibration_params(nx, ny, channels=3):
        """
        find the corners of the image using cv.findChessboardCorners
        find camera matrix and distortion coef. using cv.calibrateCamera with corners and
        pattern size as arguments. undistort the image using cv.undistort with camera matrix
        and distortion coef. as arguments
        :param nx: number of corners in x direction
        :param ny: number of corners in y direction
        :param channels: channels in image, 3 here
        :return: camera matrix and distortion coef
        """
        import matplotlib.image as mpimg
        import cv2 as cv
        import numpy as np
        import glob
        imgs = glob.glob("camera_cal/*.jpg")
        # img_pts --> 2D coordinates in image
        # obj_pts --> 3D coordinates in real world
        img_pts, obj_pts, = [], []
        # to create a matrix of 4x5 --> np.mgrid[0:4, 0:5]
        obj_pt = np.zeros(shape=(nx * ny, channels), dtype=np.float32)
        obj_pt[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        # loop over all images and append the image and object points
        for file_name in imgs:
            # read the image
            img = mpimg.imread(file_name)
            # grayscale
            gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            # find the corners
            found, corners = cv.findChessboardCorners(image=gray, patternSize=(nx, ny))
            if found is True:
                obj_pts.append(obj_pt)
                img_pts.append(corners)
                # draw the found corner points in the image
                draw_pts = np.copy(img)
                cv.drawChessboardCorners(image=draw_pts,
                                         patternSize=(nx, ny),
                                         corners=corners,
                                         patternWasFound=found)

        # use an image to find camera matrix and distortion coef
        test_img = mpimg.imread("camera_cal/calibration4.jpg")
        # find camera matrix and distortion coef
        ret, camera_matrix, dist_coef, rot_vector, trans_vector = cv.calibrateCamera(objectPoints=obj_pts,
                                                                                     imagePoints=img_pts,
                                                                                     imageSize=test_img.shape[0:2],
                                                                                     cameraMatrix=None,
                                                                                     distCoeffs=None)
        # store calibration params as pickle to avoid recalibration
        PreProcessing.save_calibration_params(camera_matrix, dist_coef)
        return camera_matrix, dist_coef

    @staticmethod
    def get_undistorted_image(nx, ny, img, load_params=True):
        """
        Using cv.undistort with calibration params as arguments, get the undistorted image
        :param nx: number of corners in x direction
        :param ny: number of corners in y direction
        :param img:source image
        :param load_params: flag to load or find calibration params
        :return: undistorted image
        """
        import cv2 as cv
        # load calibration params from pickle or else find the params
        camera_matrix, dist_coef = PreProcessing.load_calibration_params()
        # undistorted image
        undistorted = cv.undistort(src=img,
                                   cameraMatrix=camera_matrix,
                                   distCoeffs=dist_coef,
                                   dst=None,
                                   newCameraMatrix=camera_matrix)

        return undistorted

    @staticmethod
    def get_binary_image(img, sx_thresh=(20, 200), rgb_thresh=(170, 255), hls_thresh=(120, 255)):
        """
        apply color and gradient threshold
        color threshold --> R channel and S channel binarized
        gradient threshold --> sobel x binarized
        :param img: undistorted image
        :param sx_thresh: threshold range for sobel x
        :param rgb_thresh: threshold range for r channel in RGB
        :param hls_thresh: threshold range for s channel in HLS
        :return: binary image
        """
        import numpy as np
        import cv2 as cv

        # grayscale
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        gray_binary = np.zeros_like(gray)
        gray_binary[(gray >= 20) & (gray <= 80)] = 1

        # sobelx gradient threshold
        dx, dy = (1, 0)
        sx = cv.Sobel(gray, cv.CV_64F, dx, dy, ksize=3)
        sx_abs = np.absolute(sx)
        sx_8bit = np.uint8(255 * sx_abs / np.max(sx_abs))
        sx_binary = np.zeros_like(sx_8bit)
        sx_binary[(sx_8bit > sx_thresh[0]) & (sx_8bit <= sx_thresh[1])] = 1

        # RGB color space
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        r_binary = np.zeros_like(r)
        r_binary[(r >= rgb_thresh[0]) & (r <= rgb_thresh[1])] = 1

        # HLS color space
        hls = cv.cvtColor(img, cv.COLOR_RGB2HLS)
        h, l, s = hls[:, :, 0], hls[:, :, 1], hls[:, :, 2]
        s_binary = np.zeros_like(s)
        s_binary[(s >= hls_thresh[0]) & (s <= hls_thresh[1])] = 1

        # resultant of r, s and sx
        binary_image = np.zeros_like(sx_binary)
        binary_image[((sx_binary == 1) | (s_binary == 1)) & (r_binary == 1)] = 1

        return binary_image
