import os

import cv2
import imageio
import numpy as np

from config import OPENFACE_DIR
from lib.openface import openface
from src.video.VGG_model import VGG_face_model


class VGG_LSTM():

    def __init__(self):
        self.vgg_model = VGG_face_model()
        model_file = os.path.join(OPENFACE_DIR, 'models/dlib/shape_predictor_68_face_landmarks.dat')
        self.openface_model = openface.AlignDlib(model_file)

    def openface_extract_all_frames(self, video_path, depth, image_dim, color=True, all_frames=True):
        """
        Extract frame level features from the video.
        1. Extract
        :param video_path: the path to the video to extract features from
        :param depth: no of frames to extract
        :param image_dim: the
        :param color:
        :param all_frames:
        :return:
        """
        if image_dim != self.vgg_model.input_image_size:
            raise Exception('image_dim and VGG input image size should be same.')
        vgg_feature_size = 4096
        video = imageio.get_reader(video_path, 'ffmpeg')  # open video file
        meta_data = video.get_meta_data()
        nframes = meta_data['nframes']
        if all_frames:
            frame_idxs = list(range(nframes))
        else:
            # if not all frames, then determine the frames to be extracted based on nframes and depth.
            frame_idxs = [x * nframes / depth for x in range(depth)]

        frames = []
        for frame_idx in frame_idxs:
            try:
                frame = video.get_data(frame_idx)
            except Exception as e:
                continue

            bb = self.openface_model.getLargestFaceBoundingBox(frame)  # get bounding box co-ordinates of the face
            # crop the face as (image_dim x image_dim) image
            aligned_face = self.openface_model.align(image_dim, frame, bb,
                                                     landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            if aligned_face is not None:
                if not color:
                    # VGG needs 3 channels, so repeat gray features along last axis
                    aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
                    aligned_face = np.tile(aligned_face, [1, 1, 3])
                frames.append(aligned_face)

        seq_len = len(frames)
        # if no face detected, then just use pad. Can we do something much better?
        if frames:
            video_feat = np.asarray(frames)
            try:
                vid = self.vgg_model.extract_per_video_features(video_feat, len(video_feat))
            except Exception as e:
                print(video_path)
                print('video_feat shape', video_feat.shape, seq_len)
                print(len(frames))
                print(nframes)
                raise e
            vgg_feature_size = vid.shape[-1]
            pad = np.zeros(([depth - seq_len, vgg_feature_size]))
            vidd = np.concatenate([vid, pad], axis=0)
            # print('vid', vid.shape, ctr)
            # print('pad', pad.shape, depth - ctr)
            # print('before', vidd.shape, ctr)
        else:
            vidd = np.zeros(([depth, vgg_feature_size]))

        return vidd, seq_len
