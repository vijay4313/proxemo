#!/usr/bin/env python
# Title           :loader.py
# Author          :Venkatraman Narayanan, Bala Murali Manoghar, Vishnu Shashank Dorbala, Aniket Bera, Dinesh Manocha
# Copyright       :"Copyright 2020, Proxemo project"
# Version         :1.0
# License         :"MIT"
# Maintainer      :Venkatraman Narayanan, Bala Murali Manoghar
# Email           :vnarayan@terpmail.umd.edu, bsaisudh@terpmail.umd.edu
# ==============================================================================
import pyrealsense2 as rs
import numpy as np
import cv2


class Real_Sense_Camera():
    """Real sense camera interface class."""

    def __init__(self, clipping_distance_meters, kernel_size):
        """Constructor.

        Args:
            clipping_distance_meters (float): Max depth set for detection
            kernel_size (int): Noise removal kernel size (average filter)
        """
        #  clipping_distance_in_meters meters away
        self.init_camera()
        self.set_clipping_distance(clipping_distance_meters)
        self.avg_kernel = np.ones(
            (kernel_size, kernel_size), np.float32)/(kernel_size**2)

    def set_clipping_distance(self, clipping_distance_meters):
        """Set clipping distance parameter.

        Args:
            clipping_distance_meters (float): Max depth set for detection
        """
        # We will be removing the background of objects more than
        self.clipping_distance = clipping_distance_meters / self.depth_scale

    def init_camera(self):
        """Setup camera."""
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # self.config.enable_device_from_file(self.config, '/home/emotiongroup/Desktop/rosbag/run2/20200305_225145.bag', False)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(
            rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.profile = self.pipeline.start(self.config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # Downcast to video_stream_profile and fetch intrinsics
        self.intrinsics = self.profile.get_stream(rs.stream.depth).\
            as_video_stream_profile().\
            get_intrinsics()

    def cleanup(self):
        """End camera capture."""
        self.pipeline.stop()

    def capture(self):
        """Capture camera frames."""
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = self.align.process(frames)

            # Get aligned frames
            # aligned_depth_frame is a 640x480 depth image
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame_align = aligned_frames.get_color_frame()

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame \
                or not color_frame \
                    or not aligned_depth_frame \
            or not color_frame_align:
                continue

            # Convert images to numpy arrays
            self.depth_image = np.asanyarray(depth_frame.get_data())
            self.color_image = np.asanyarray(color_frame.get_data())
            self.depth_image_align = np.asanyarray(
                aligned_depth_frame.get_data())

            # average depth image
            self.depth_image_align = cv2.filter2D(self.depth_image_align.astype(np.float32),
                                                  -1, self.avg_kernel)

            # Remove background - Set pixels further than clipping_distance to grey
            # depth image is 1 channel, color is 3 channels
            grey_color = 153
            depth_image_3d = np.dstack((self.depth_image_align,
                                        self.depth_image_align,
                                        self.depth_image_align))
            depth_mask = (depth_image_3d > self.clipping_distance) | (
                depth_image_3d <= 0)
            self.bg_removed = np.where(
                depth_mask, grey_color, self.color_image)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            self.depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.03),
                                                    cv2.COLORMAP_JET)
            break
