from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil
import torch
import cv2
from scipy.spatial.transform import Rotation as R
from .mono_dataset import MonoDataset


class InteriorDataset(MonoDataset):
    """Superclass for different types of Interior dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(InteriorDataset, self).__init__(*args, **kwargs)
        self.K = np.array([[600/640, 0, 0.5, 0],
                           [0, 600/480, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.full_res_shape = (640, 480)
        self.gt_pose_dict = self.load_pose(self.data_path)

    def check_depth(self):
        line = self.filenames[0].split()
        folder = line[0]
        frame_index = line[1]
        depth_filename = os.path.join(
            self.data_path,
            folder,
            "depth/{}.png".format(frame_index))

        return os.path.isfile(depth_filename)

    def get_color(self, folder, frame_index):
        color = self.loader(self.get_image_path(folder, frame_index))
        return color

    def get_image_path(self, folder, frame_index):
        image_path = self.data_path + folder + "/rgb/{}{}".format(frame_index, self.img_ext)
        return image_path

    def get_depth(self, folder, frame_index):
        depth_path = self.data_path + folder + "/depth/{}{}".format(frame_index, self.img_ext)
        depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_gt = np.expand_dims(depth_gt, 0)
        return torch.from_numpy(depth_gt.astype(np.float))

    def get_gt_pose(self, folder, frame_index):
        [trajectory, cam] = folder.split('/')
        gt_pose = self.gt_pose_dict[(trajectory, cam, frame_index)]
        return torch.from_numpy(gt_pose)

    def load_pose(self, data_path):
        gt_pose = {}
        for trajectory in os.listdir(data_path):
            t = trajectory
            trajectory_pose_path = data_path + trajectory + '/' + trajectory
            for cam in os.listdir(trajectory_pose_path):
                c = cam
                cam_pose_path = trajectory_pose_path + '/' + cam + '/' + 'cam0.ccam'
                pose_file = open(cam_pose_path, 'r')
                pose_lines = pose_file.readlines()
                count = 0
                for line in pose_lines:
                    if line.startswith("#") or len(line) < 10:
                        continue
                    line = line.split()
                    img = count
                    count += 1
                    line = [float(i) for i in line]
                    [w, x, y, z] = line[6:10]
                    [tx, ty, tz] = line[10:13]
                    r = R.from_quat([x, y, z, w])
                    rot_matrix = r.as_matrix()
                    S = np.array([[0, 1, 0],
                                  [-1, 0, 0],
                                  [0, 0, 1]])
                    Rl = rot_matrix
                    Rr = np.matmul(Rl, S)
                    T = np.concatenate((Rr, np.array([[-ty], [tx], [tz]])), 1)
                    T = np.concatenate((T, np.array([[0], [0], [0], [1]]).transpose()), 0)
                    T = T.astype(np.float32)
                    gt_pose[(t, c, img)] = T
        print('poses: ', len(gt_pose))
        return gt_pose
