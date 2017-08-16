import sys

import imageio

from pre_processing import PreProcessing

sys.path.append("implementation/")
from lane_detection import LaneDetection


def __main__():
    path = "../project_video.mp4"
    video_cap = imageio.get_reader(path)

    # polynomial lane fit
    lanes_fit = []

    # load calibration parameters:
    camera_matrix, dist_coef = PreProcessing.load_calibration_params()
    for img in video_cap:
        lanes_fit = LaneDetection.pipeline(img, lanes_fit, camera_matrix, dist_coef)


__main__()
