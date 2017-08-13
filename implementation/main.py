def __main__():
    import imageio
    import sys
    sys.path.append("implementation/")
    from lane_detection import LaneDetection

    path = "../harder_challenge_video.mp4"
    video_cap = imageio.get_reader(path)
    # polynomial lane fit
    lanes_fit = []
    for img in video_cap:
        lanes_fit = LaneDetection.pipeline(img, lanes_fit)


__main__()
