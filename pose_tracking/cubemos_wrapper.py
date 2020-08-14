#!/usr/bin/env python3

# cubemos init
import os
import numpy as np
import cv2

from cubemos.core.nativewrapper import CM_TargetComputeDevice
from cubemos.core.nativewrapper import initialise_logging, CM_LogLevel
from cubemos.skeleton_tracking.nativewrapper import Api

from cubemos_api import default_log_dir,\
    default_license_dir,\
    check_license_and_variables_exist,\
    render_result

"""
18  19    1-Root,
19  20    2-Spine,
01  02    3-Neck,
00  01    4-Head,
05  06    5-Left Shoulder,
06  07    6-Left Elbow,
07  08    7- Left Hand,
02  03    8- Right Shoulder
03  04    9- Right Elbow
04  05    10- Right Hand
11  12    11-Left Thigh,
12  13   12-Left Knee,
13  14    13-Left Foot,
08  09    14-Right Thigh,
09  10    15-Right Knee,
10  11   16-Right Foot"

[18, 19, 1, 0, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10]
"""


class Cubemos_Tacker():
    def __init__(self, intrinsics, verbose=False):
        self.verbose = verbose
        self.intrinsics = intrinsics
        self.confidence_threshold = 0.3
        self.network_height = 16*20
        self.skeletons = []
        self.new_skeletons = []
        self.skel3d_np = []
        self.skel_ids = []
        self.roots_2d = []
        self.backs_2d = []
        self.keypoint_map = np.asarray(
            [18, 19, 1, 0, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10])
        self.init_skel_track()
        self.init_cubemos_api()

    def init_skel_track(self):
        check_license_and_variables_exist()
        # Get the path of the native libraries and ressource files
        self.sdk_path = os.environ["CUBEMOS_SKEL_SDK"]
        if self.verbose:
            initialise_logging(self.sdk_path,
                               CM_LogLevel.CM_LL_DEBUG,
                               True,
                               default_log_dir())

    def init_cubemos_api(self):
        # initialize the api with a valid license key in default_license_dir()
        self.api = Api(default_license_dir())
        model_path = os.path.join(self.sdk_path,
                                  "models",
                                  "skeleton-tracking",
                                  "fp32",
                                  "skeleton-tracking.cubemos")
        self.api.load_model(CM_TargetComputeDevice.CM_CPU, model_path)

    def track_skeletons(self, image, depth_image):
        # perform inference
        self.skeletons = self.api.estimate_keypoints(
            image, self.network_height)

        # perform inference again to demonstrate tracking functionality.
        # usually you would estimate the keypoints on another image and then
        # update the tracking id
        self.new_skeletons = self.api.estimate_keypoints(
            image, self.network_height)
        self.new_skeletons = self.api.update_tracking_id(
            self.skeletons, self.new_skeletons)
        self.skel2D_to_skel3D(depth_image)

    def render_skeletons(self, image):
        render_result(self.skeletons, image, self.confidence_threshold)
        for skel_num, joints in enumerate(self.skel3d_np):
            for joint_ndx, pt_3D in enumerate(joints[[0, -1]]):
                x_pos = int(self.skeletons[skel_num][0][joint_ndx][0])
                y_pos = int(self.skeletons[skel_num][0][joint_ndx][1])
                text = f"{pt_3D[0]:.2f}, {pt_3D[1]:.2f}, {pt_3D[2]:.2f}"
                text = f"{pt_3D[2]:.2f}"
                # text = f"{self.skel_ids[skel_num]}"
                cv2.putText(image,
                            text,
                            (x_pos, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                            cv2.LINE_AA)
                # add root and back
                cv2.circle(image,
                           (int(self.roots_2d[skel_num][0]),
                            int(self.roots_2d[skel_num][1])),
                           2, (255, 0, 255), -1)
                cv2.circle(image,
                           (int(self.backs_2d[skel_num][0]),
                            int(self.backs_2d[skel_num][1])),
                           2, (255, 0, 255), -1)

    def map_2D_3D(self, pixel, depth):
        x = (pixel[0] - self.intrinsics.ppx) / self.intrinsics.fx
        y = (pixel[1] - self.intrinsics.ppy) / self.intrinsics.fy

        r2 = x * x + y * y
        f = 1 + self.intrinsics.coeffs[0] * r2 + \
            self.intrinsics.coeffs[1] * r2 * r2 + \
            self.intrinsics.coeffs[4] * r2 * r2 * r2

        ux = x * f + 2 * self.intrinsics.coeffs[2] * x * y + \
            self.intrinsics.coeffs[3] * (r2 + 2 * x * x)

        uy = y * f + 2 * self.intrinsics.coeffs[3] * x * y + \
            self.intrinsics.coeffs[2] * (r2 + 2 * y * y)

        x = ux
        y = uy

        X = depth * x
        Y = depth * y
        Z = depth

        return (X, Y, Z)

    def skel2D_to_skel3D(self, depth_image):
        self.skel_ids = []
        self.skel3d_np = []
        self.roots_2d = []
        self.backs_2d = []
        for skeleton in self.skeletons:
            joints = skeleton.joints
            # confidences = skeleton[1]
            self.skel_ids.append(skeleton.id)
            joints_3d = []
            for joint in joints:
                x_ndx = min(int(joint.x), depth_image.shape[0]-1)
                y_ndx = min(int(joint.y), depth_image.shape[1]-1)
                depth = depth_image[x_ndx, y_ndx]
                pt_3D = self.map_2D_3D((x_ndx, y_ndx), depth)
                joints_3d.append(pt_3D)
            # add root
            self.roots_2d.append(((joints[8].x + joints[11].x)/2,
                                  (joints[8].y + joints[11].y)/2))
            l_hip = joints_3d[8]
            r_hip = joints_3d[11]
            root_x = (l_hip[0] + r_hip[0]) / 2
            root_y = (l_hip[1] + r_hip[1]) / 2
            root_z = (l_hip[2] + r_hip[2]) / 2
            root = (root_x, root_y, root_z)
            joints_3d.append(root)
            # add back
            self.backs_2d.append(((joints[1].x + self.roots_2d[-1][0])/2,
                                  (joints[1].y + self.roots_2d[-1][1])/2))
            neck = joints_3d[1]
            back_x = (neck[0] + root[0]) / 2
            back_y = (neck[1] + root[1]) / 2
            back_z = (neck[2] + root[2]) / 2
            back = (back_x, back_y, back_z)
            joints_3d.append(back)
            # all all joints
            self.skel3d_np.append(joints_3d)
        self.skel3d_np = np.asarray(self.skel3d_np)
        if len(self.skeletons) > 0:
            self.skel3d_np = self.skel3d_np[:, self.keypoint_map, :]
