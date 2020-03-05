#!/usr/bin/env python3

# cubemos init
from cubemos.core.nativewrapper import CM_TargetComputeDevice
from cubemos.core.nativewrapper import initialise_logging, CM_LogLevel
from cubemos.skeleton_tracking.nativewrapper import Api, SkeletonKeypoints
import cv2
import argparse
import os
import platform
from pprint import pprint

# Realsense init
import pyrealsense2 as rs
import numpy as np

from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt

keypoint_ids = [
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
]


def default_log_dir():
    if platform.system() == "Windows":
        return os.path.join(os.environ["LOCALAPPDATA"], "Cubemos", "SkeletonTracking", "logs")
    elif platform.system() == "Linux":
        return os.path.join(os.environ["HOME"], ".cubemos", "skeleton_tracking", "logs")
    else:
        raise Exception("{} is not supported".format(platform.system()))


def default_license_dir():
    if platform.system() == "Windows":
        return os.path.join(os.environ["LOCALAPPDATA"], "Cubemos", "SkeletonTracking", "license")
    elif platform.system() == "Linux":
        return os.path.join(os.environ["HOME"], ".cubemos", "skeleton_tracking", "license")
    else:
        raise Exception("{} is not supported".format(platform.system()))


def check_license_and_variables_exist():
    license_path = os.path.join(default_license_dir(), "cubemos_license.json")
    if not os.path.isfile(license_path):
        raise Exception(
            "The license file has not been found at location \"" +
            default_license_dir() + "\". "
            "Please have a look at the Getting Started Guide on how to "
            "use the post-installation script to generate the license file")
    if "CUBEMOS_SKEL_SDK" not in os.environ:
        raise Exception(
            "The environment Variable \"CUBEMOS_SKEL_SDK\" is not set. "
            "Please check the troubleshooting section in the Getting "
            "Started Guide to resolve this issue." 
        )


def get_valid_limbs(keypoint_ids, skeleton, confidence_threshold):
    limbs = [
        (tuple(map(int, skeleton.joints[i])), tuple(map(int, skeleton.joints[v])))
        for (i, v) in keypoint_ids
        if skeleton.confidences[i] >= confidence_threshold
        and skeleton.confidences[v] >= confidence_threshold
    ]
    valid_limbs = [
        limb
        for limb in limbs
        if limb[0][0] >= 0 and limb[0][1] >= 0 and limb[1][0] >= 0 and limb[1][1] >= 0
    ]
    return valid_limbs


def render_result(skeletons, img, confidence_threshold):
    skeleton_color = (100, 254, 213)
    for index, skeleton in enumerate(skeletons):
        limbs = get_valid_limbs(keypoint_ids, skeleton, confidence_threshold)
        for limb in limbs:
            cv2.line(
                img, limb[0], limb[1], skeleton_color, thickness=2, lineType=cv2.LINE_AA
            )

parser = argparse.ArgumentParser(description="Perform keypoing estimation on an image")
parser.add_argument(
    "-c",
    "--confidence_threshold",
    type=float,
    default=0.5,
    help="Minimum confidence (0-1) of displayed joints",
)
parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    help="Increase output verbosity by enabling backend logging",
)

parser.add_argument(
    "-o",
    "--output_image",
    type=str,
    help="filename of the output image",
)

parser.add_argument("image", metavar="I", type=str, help="filename of the input image")

def init_skel_track(verbose = True):
    check_license_and_variables_exist()
    #Get the path of the native libraries and ressource files
    sdk_path = os.environ["CUBEMOS_SKEL_SDK"]
    if verbose:
        initialise_logging(sdk_path, CM_LogLevel.CM_LL_DEBUG, True, default_log_dir())
    return sdk_path

def init_cubemos_api(sdk_path):
    #initialize the api with a valid license key in default_license_dir()
    api = Api(default_license_dir())
    model_path = os.path.join(
        sdk_path, "models", "skeleton-tracking", "fp32", "skeleton-tracking.cubemos"
    )
    api.load_model(CM_TargetComputeDevice.CM_CPU, model_path)
    return api

def cubemos_track_skeletons(api, img, verbose = True, display = True):
    confidence_threshold = 0.5
    network_height = 16*20
    #perform inference
    skeletons = api.estimate_keypoints(img, network_height)

    # perform inference again to demonstrate tracking functionality.
    # usually you would estimate the keypoints on another image and then
    # update the tracking id
    new_skeletons = api.estimate_keypoints(img, network_height)
    new_skeletons = api.update_tracking_id(skeletons, new_skeletons)

    render_result(skeletons, img, confidence_threshold)

    return skeletons, new_skeletons, img
        

def init_realsense_capture():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)
    
    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 10 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    profile = profile.get_stream(rs.stream.depth)

    return pipeline, config, align, clipping_distance, profile

def realsense_cleanup(pipeline):
    pipeline.stop()

def realsense_capture(pipelene, align, clipping_distance):
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame_align = aligned_frames.get_color_frame()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame or not aligned_depth_frame or not color_frame_align:
            continue
        
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        depth_image_align = np.asanyarray(aligned_depth_frame.get_data())

        # average depth image
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.float32)/(kernel_size**2)
        depth_image_align = cv2.filter2D(depth_image_align.astype(np.float32), -1, kernel)
        
        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image_align, depth_image_align, depth_image_align)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        return color_image, depth_image, depth_colormap, depth_image_align, bg_removed
    
def map_2D_3D(pixel, depth, intrinsics):
    x = (pixel[0] - intrinsics.ppx) / intrinsics.fx
    y = (pixel[1] - intrinsics.ppy) / intrinsics.fy

    r2 = x * x + y * y
    f = 1 + intrinsics.coeffs[0] * r2 + intrinsics.coeffs[1] * r2 * r2 + intrinsics.coeffs[4] * r2 * r2 * r2
    ux = x * f + 2 * intrinsics.coeffs[2] * x * y + intrinsics.coeffs[3] * (r2 + 2 * x * x)
    uy = y * f + 2 * intrinsics.coeffs[3] * x * y + intrinsics.coeffs[2] * (r2 + 2 * y * y)

    x = ux
    y = uy

    X = depth * x
    Y = depth * y
    Z = depth

    return (X, Y, Z)



# Main content begins
if __name__ == "__main__":
    try:
        verbose = False
        display = True
        plot = False
        # init
        pipeline, config, align, clipping_distance, profile = init_realsense_capture()
        sdk_path = init_skel_track(verbose)
        api = init_cubemos_api(sdk_path)
        
        if plot:
            plt.ion()
            fig3D = plt.figure(0)
            ax = fig3D.add_subplot(111, projection='3d')
            ax.set_title('points')
            ax.set_xlabel('z')
            ax.set_ylabel('y')
            ax.set_zlabel('x')

        while True:
            # capture
            color_image, depth_image, depth_colormap, depth_image_align, bg_removed = realsense_capture(pipeline, align, clipping_distance)
            # get skeletons
            skeletons, new_skeletons, skel_img = cubemos_track_skeletons(api, bg_removed, verbose)
            # convert to 3D skeleton
            skeletons_3d = []

            for skel_num, skeleton in enumerate(skeletons):
                joints = skeleton[0]
                confidences = skeleton[1]
                skel_id = skeleton[2]
                joints_3d = []
                if plot:
                    xs = [0]
                    ys = [0]
                    zs = [0]
                for joint_num, joint in enumerate(skeleton[0]):
                    x_ndx = min(int(joint[0]), depth_image_align.shape[0]-1)
                    y_ndx = min(int(joint[1]), depth_image_align.shape[1]-1)
                    depth = depth_image_align[x_ndx, y_ndx]
                    joints_3d.append([joint[0], joint[1], depth])
                    skeletons_3d.append([joints_3d, confidences, skel_id])
                    intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
                    pt_3D = map_2D_3D((x_ndx, y_ndx), depth, intr)
                    cv2.putText(skel_img, '({x:.2f}, {y:.2f}, {z:.2f}'.format(x=pt_3D[0], y=pt_3D[1], z=pt_3D[2]),
                                (x_ndx, y_ndx), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
                    if plot:
                        if x_ndx > 0 and y_ndx > 0:
                            xs.append(x_ndx)
                            ys.append(y_ndx)
                            zs.append(depth_image_align[x_ndx, y_ndx])
                if plot:
                    ax.scatter(zs, ys, xs)
                    plt.draw()
                    plt.pause(0.5)
                    ax.clear()
            
            if verbose:
                print("Detected skeletons: ", len(skeletons))
                print(skeletons)
                print(new_skeletons)
              
            if display:
                # Stack both images horizontally
                images = np.hstack((color_image, depth_colormap))
                if len(skeletons) > 0:
                    images = np.hstack((images, skel_img))
                else:
                    images = np.hstack((images, color_image))

                # Show images
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', images)
                key = cv2.waitKey(1) & 0xFF
                # press the 'q' key to stop the video stream
                if key == ord("q"):
                    break

        realsense_cleanup(pipeline)

            
    except Exception as ex:
        realsense_cleanup(pipeline)
        print("Exception occured: \"{}\"".format(ex))