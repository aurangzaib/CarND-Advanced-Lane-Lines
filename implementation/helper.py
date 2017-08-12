from pre_processing import PreProcessing


class Helper:
    @staticmethod
    def get_undistorted_sample_images():
        import matplotlib.image as mpimg
        import cv2 as cv
        import glob

        imgs = glob.glob("../camera_cal/*.jpg")

        # load calibration params from pickle or else find the params
        camera_matrix, dist_coef = PreProcessing.load_calibration_params()
        for file_name in imgs:
            # read the image
            img = mpimg.imread(file_name)
            # grayscale
            gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            # undistorted image
            undistorted = cv.undistort(src=gray,
                                       cameraMatrix=camera_matrix,
                                       distCoeffs=dist_coef,
                                       dst=None,
                                       newCameraMatrix=camera_matrix)
            img_name = file_name.rsplit('.', 1)[0]
            img_ext = '.' + file_name.rsplit('.', 1)[1]
            img_updated_name = img_name + "-undistorted" + img_ext
            mpimg.imsave(img_updated_name, undistorted, cmap="gray")

    @staticmethod
    def get_undistorted_sample_video():
        import matplotlib.image as mpimg
        import cv2 as cv
        import imageio

        video = imageio.get_reader("../project_video.mp4")
        camera_matrix, dist_coef = PreProcessing.load_calibration_params()

        for index, img in enumerate(video):
            if index % 100 == 0:
                undistorted = cv.undistort(src=img,
                                           cameraMatrix=camera_matrix,
                                           distCoeffs=dist_coef,
                                           dst=None,
                                           newCameraMatrix=camera_matrix)
                mpimg.imsave("../documentation/undistorted-original-" + str(index) + ".jpg", img)
                mpimg.imsave("../documentation/undistorted-" + str(index) + ".jpg", undistorted)

    @staticmethod
    def save_binarized_image(img, binary_image):
        import matplotlib.image as mpimg
        import time
        seconds = time.time() % 60
        if seconds % 10:
            t = int(time.time())
            mpimg.imsave("../buffer/binary-original-" + str(t) + ".jpg", img, cmap="gray")
            mpimg.imsave("../buffer/binary-" + str(t) + ".jpg", binary_image, cmap="gray")
