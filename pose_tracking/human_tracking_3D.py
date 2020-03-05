#!/usr/bin/env python3

import cv2
import numpy as np

from real_sense_wrapper import Real_Sense_Camera
from cubemos_wrapper import Cubemos_Tacker

class Track_Human_Pose():
    def __init__(self, display=True, verbose=True):
        self.verbose = verbose
        self.display = display

        self.camera = Real_Sense_Camera(5, 3)
        self.cubemos = Cubemos_Tacker(self.camera.intrinsics)

    def get_pose(self):
        # capture
        self.camera.capture()
        # get skeletons
        self.cubemos.track_skeletons(self.camera.color_image, self.camera.depth_image_align)
        self.cubemos.render_skeletons(self.camera.color_image)
        if self.display:
            # Stack both images horizontally
            images = np.hstack((self.camera.color_image,
                                self.camera.depth_colormap))
            images = np.hstack((images, self.camera.color_image))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)

    def cleanup(self):
        self.camera.cleanup()

if __name__ == "__main__":
    track_pose = Track_Human_Pose(display=True)
    while True:
        track_pose.get_pose()
        if track_pose.display:
            key = cv2.waitKey(1) & 0xFF
            # press the 'q' key to stop the video stream
            if key == ord("q"):
                break