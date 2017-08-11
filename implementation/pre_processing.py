class PreProcessing:
    @staticmethod
    def save_calibration_params(camera_matrix, dist_coef, filename="calibration_parameters.p"):
        """
        save the matrix and coef in a pickle file
        """
        import pickle
        parameters = {
            'camera_matrix': camera_matrix,
            'dist_coef': dist_coef
        }
        pickle.dump(parameters, open(filename, "wb"))
        print("data saved to disk")

    @staticmethod
    def load_calibration_params(filename="calibration_parameters.p"):
        """
        load pickle files for train, validation and test
        """
        import pickle
        with open(filename, mode='rb') as f:
            parameters = pickle.load(f)
        return parameters['camera_matrix'], parameters['dist_coef']

    @staticmethod
    def get_calibration_params(nx, ny, channels=3):
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
        PreProcessing.save_calibration_params(camera_matrix, dist_coef)
        return camera_matrix, dist_coef

    @staticmethod
    def get_undistorted_image(nx, ny, img, load_params=True):
        import cv2 as cv
        camera_matrix, dist_coef = PreProcessing.load_calibration_params() \
            if load_params \
            else PreProcessing.get_calibration_params(nx, ny)

        undistorted = cv.undistort(src=img,
                                   cameraMatrix=camera_matrix,
                                   distCoeffs=dist_coef,
                                   dst=None,
                                   newCameraMatrix=camera_matrix)
        return undistorted

    @staticmethod
    def get_binary_image(img):
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
