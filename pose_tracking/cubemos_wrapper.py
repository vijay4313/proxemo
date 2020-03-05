#!/usr/bin/env python3

# cubemos init

import numpy as np

from cubemos_api import *

class Cubemos_Tacker():
    def __init__(self, intrinsics, verbose = False):
        self.verbose = verbose
        self.intrinsics = intrinsics
        self.confidence_threshold = 0.3
        self.network_height = 16*20
        self.skeletons = []
        self.new_skeletons = []
        self.skel3d_np = []
        self.skel_ids = []
        self.init_skel_track()
        self.init_cubemos_api()

    def init_skel_track(self):
        check_license_and_variables_exist()
        #Get the path of the native libraries and ressource files
        self.sdk_path = os.environ["CUBEMOS_SKEL_SDK"]
        if self.verbose:
            initialise_logging(self.sdk_path,
                               CM_LogLevel.CM_LL_DEBUG,
                               True,
                               default_log_dir())

    def init_cubemos_api(self):
        #initialize the api with a valid license key in default_license_dir()
        self.api = Api(default_license_dir())
        model_path = os.path.join(self.sdk_path,
                                  "models",
                                  "skeleton-tracking",
                                  "fp32",
                                  "skeleton-tracking.cubemos")
        self.api.load_model(CM_TargetComputeDevice.CM_CPU, model_path)

    def track_skeletons(self, image, depth_image):
        #perform inference
        self.skeletons = self.api.estimate_keypoints(image, self.network_height)

        # perform inference again to demonstrate tracking functionality.
        # usually you would estimate the keypoints on another image and then
        # update the tracking id
        self.new_skeletons = self.api.estimate_keypoints(image, self.network_height)
        self.new_skeletons = self.api.update_tracking_id(self.skeletons, self.new_skeletons)
        self.skel2D_to_skel3D(depth_image)

    def render_skeletons(self, image):
        render_result(self.skeletons, image, self.confidence_threshold)
        for skel_num, joints in enumerate(self.skel3d_np):
                for joint_ndx, pt_3D in enumerate(joints):
                    x_pos = int(self.skeletons[skel_num][0][joint_ndx][0])
                    y_pos = int(self.skeletons[skel_num][0][joint_ndx][1])
                    cv2.putText(image,
                                f"{pt_3D[0]:.2f}, {pt_3D[1]:.2f}, {pt_3D[2]:.2f}",
                                (x_pos, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA)
        
        
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
        for skeleton in self.skeletons:
            joints = skeleton[0]
            # confidences = skeleton[1]
            self.skel_ids.append(skeleton[2])
            joints_3d = []
            for joint in joints:
                x_ndx = min(int(joint[0]), depth_image.shape[0]-1)
                y_ndx = min(int(joint[1]), depth_image.shape[1]-1)
                depth = depth_image[x_ndx, y_ndx]
                pt_3D = self.map_2D_3D((x_ndx, y_ndx), depth)
                joints_3d.append(pt_3D)
            self.skel3d_np.append(joints_3d)
        self.skel3d_np = np.asarray(self.skel3d_np)
        