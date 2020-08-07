#!/usr/bin/env python3

import cv2
import numpy as np

from pose_tracking.real_sense_wrapper import Real_Sense_Camera
from pose_tracking.cubemos_wrapper import Cubemos_Tacker

class Skel_Temporal():
    def __init__(self, skel_id, do_not_ignore_false_limbs = True):
        self.id = skel_id
        self.skel_temporal = []
        self.do_not_ignore_false_limbs = do_not_ignore_false_limbs

    def add(self, skel_ti):
        if not np.any(skel_ti == -1) or self.ignore_false_limbs:
            if len(self.skel_temporal) > 75:
                self.skel_temporal.pop(0)
            self.skel_temporal.append(skel_ti)

    def __eq__(self, other):
        try:
            if self.id == other.id:
                return True
            else:
                return False
        except:
            if self.id == other:
                return True
            else:
                return False
    
    def get_embedding(self):
        skel_temporal_np = np.array(self.skel_temporal)
        # make root as (0, 0, 0)
        # even if number of frames is less than 75 it will be resized to 244*244
        skel_temporal_np = skel_temporal_np - \
                                np.expand_dims(skel_temporal_np[:, 0, :], axis=1)
        skel_temporal_img = cv2.resize(skel_temporal_np, (244, 244))
        return skel_temporal_img

class Skel_Tracker():
    def __init__(self, do_not_ignore_false_limbs = True):
        self.skel_tracks = []
        self.img_embeddings = []
        self.do_not_ignore_false_limbs = do_not_ignore_false_limbs

    def update(self, skel_nps, skel_ids):
        # add skeleton corresponding to id
        for skel_np, skel_id in zip(skel_nps, skel_ids):
            try:
                # ID already present - update
                ndx = self.skel_tracks.index(skel_id)
                skel_temporal = self.skel_tracks[ndx]
                skel_temporal.add(skel_np)
            except ValueError:
                # new human - add
                skel_temporal = Skel_Temporal(skel_id,  self.do_not_ignore_false_limbs)
                skel_temporal.add(skel_np)
                self.skel_tracks.append(skel_temporal)
        # delete obselete human ids
        skel_ids = np.asarray(skel_ids)
        ndx_to_delete = []
        for ndx, skel_temporal in enumerate(self.skel_tracks):
            if not any(skel_ids == skel_temporal.id):
                # tracked id is not present in current frame
                ndx_to_delete.append(ndx)
        for ndx, value in enumerate(ndx_to_delete):
            # considering ndx_to_delete will be sorted in ascending order
            # while poping elements one by one, the index has to be decreased
            # by number of elements already deleted
            self.skel_tracks.pop(value - ndx)

    def get_embedding(self):
        self.img_embeddings = []
        ids = []
        for skel_track in self.skel_tracks:
            self.img_embeddings.append(skel_track.get_embedding())
            ids.append(skel_track.id)
        self.img_embeddings = np.asarray(self.img_embeddings)
        return self.img_embeddings, ids
    
    def display_embedding(self):
        imgs = self.img_embeddings[0] # np.empty((244,244,3))
        print(self.img_embeddings.shape)
        for img in self.img_embeddings[1:]:
            print("--")
            imgs = np.hstack((imgs, img))
        print(imgs.shape)
        cv2.imshow("embeddings", imgs.astype(np.uint8))

class Track_Human_Pose():
    def __init__(self, display=True, verbose=True):
        self.verbose = verbose
        self.display = display

        self.camera = Real_Sense_Camera(5, 3)
        self.cubemos = Cubemos_Tacker(self.camera.intrinsics)
        
        self.skel_tracker = Skel_Tracker()

    def get_pose(self):
        # capture
        self.camera.capture()
        # get skeletons
        self.cubemos.track_skeletons(self.camera.color_image,
                                     self.camera.depth_image_align)
        self.cubemos.render_skeletons(self.camera.color_image)
        if self.display:
            # Stack both images horizontally
            images = np.hstack((self.camera.color_image,
                                self.camera.depth_colormap))
            images = np.hstack((images, self.camera.color_image))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
    
    def track_pose(self):
        self.skel_tracker.update(self.cubemos.skel3d_np,
                                 self.cubemos.skel_ids)

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
