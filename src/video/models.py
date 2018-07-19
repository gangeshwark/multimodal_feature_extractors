import os

import cv2
import imageio
import numpy as np
import tensorflow as tf

from config import OPENFACE_DIR
from lib.kaffe.tensorflow import Network
from lib.openface import openface


class VGG_FACE_16_Layer(Network):
    def setup(self):
        (self.feed('data')
         .conv(3, 3, 64, 1, 1, name='conv1_1')
         .conv(3, 3, 64, 1, 1, name='conv1_2')
         .max_pool(2, 2, 2, 2, name='pool1')
         .conv(3, 3, 128, 1, 1, name='conv2_1')
         .conv(3, 3, 128, 1, 1, name='conv2_2')
         .max_pool(2, 2, 2, 2, name='pool2')
         .conv(3, 3, 256, 1, 1, name='conv3_1')
         .conv(3, 3, 256, 1, 1, name='conv3_2')
         .conv(3, 3, 256, 1, 1, name='conv3_3')
         .max_pool(2, 2, 2, 2, name='pool3')
         .conv(3, 3, 512, 1, 1, name='conv4_1')
         .conv(3, 3, 512, 1, 1, name='conv4_2')
         .conv(3, 3, 512, 1, 1, name='conv4_3')
         .max_pool(2, 2, 2, 2, name='pool4')
         .conv(3, 3, 512, 1, 1, name='conv5_1')
         .conv(3, 3, 512, 1, 1, name='conv5_2')
         .conv(3, 3, 512, 1, 1, name='conv5_3')
         .max_pool(2, 2, 2, 2, name='pool5')
         .fc(4096, name='fc6')
         .fc(4096, name='fc7')
         .fc(2622, relu=False, name='fc8')
         .softmax(name='prob'))


class VGG_Face_Extractor():
    def __init__(self):
        self.input_image_size = 224
        self.batch_size = 100
        self.channels = 3
        self.images = tf.placeholder(tf.float32, [None, self.input_image_size, self.input_image_size, self.channels])
        self.model = VGG_FACE_16_Layer({'data': self.images})

        self.feature_layer = self.model.layers['fc7']

        self.out = self.model.layers['prob']
        allow_soft_placement = True
        log_device_placement = False
        session_conf = tf.ConfigProto(
            # device_count={'GPU': gpu_count},
            allow_soft_placement=allow_soft_placement,
            log_device_placement=log_device_placement,
            gpu_options=tf.GPUOptions(allow_growth=True))
        self.sess = tf.Session(config=session_conf)
        self.sess.run(tf.global_variables_initializer())
        print('Loading VGG model...')
        self.model.load('models/vgg_face_model.npy', self.sess)
        print('VGG model loaded...')

    def extract_features(self, data, frames):
        """
        data => (videos, frames, h, w, channels)
        :param data:
        :return:
        """
        features = []

        data = np.transpose(data, [1, 0, 2, 3, 4])

        for frame in range(frames):
            feed = {self.images: data[frame, :, :, :, :]}

            frame_features = self.sess.run([self.feature_layer], feed_dict=feed)
            features.append(frame_features)
        features = np.array(features)
        features = np.transpose(features, [1, 0, 2])
        return features

    def extract_per_video_features(self, data, frames):
        """
        data => (frames, h, w, channels)
        :param data:
        :param frames:
        :return:
        """
        features = []

        for x in range(0, frames, self.batch_size):
            feed = {self.images: data[x:x + self.batch_size]}
            frame_features = self.sess.run([self.feature_layer], feed_dict=feed)
            features.append(frame_features)
        features = np.concatenate(features, axis=0)
        return features

    def extract_per_frame_features(self, data):
        """
        data => (h, w, channels)
        :param data:
        :param frames:
        :return:
        """
        feed = {self.images: data}
        return self.sess.run([self.feature_layer], feed_dict=feed)


class OpenFace_VGG():

    def __init__(self):
        self.vgg_extractor = VGG_Face_Extractor()
        model_file = os.path.join(OPENFACE_DIR, 'models/dlib/shape_predictor_68_face_landmarks.dat')
        self.openface_model = openface.AlignDlib(model_file)

    def extract_video_features(self, video_path, depth, image_dim, color=True, all_frames=True, face_only=True):
        """
        Extract frame level features from the video specified in :param video_path.
        1. Read video and get metadata.
        2. Determine frames to extract.
        3. Extract those frames.
        4. Extract frame features from VGG Face model.
        5. Return video level features.

        :param video_path: the path to the video to extract features from
        :param depth: no of frames to extract.
        :param image_dim: the dimension of the image
        :param color: True if you want the frame extracted to be color, else False
        :param all_frames: True if you want to extract features of all the frames in the video.
        :param face_only: True if only facial features should be extracted. Else false.
        :return: ndarray of shape (depth, vgg_feature_size)
        """
        if face_only:
            return self._extract_openface_vgg_video_features(video_path, depth, image_dim, color=color,
                                                             all_frames=all_frames)
        else:
            raise NotImplementedError('Not implemented general video extractor')

    def _extract_openface_vgg_video_features(self, video_path, depth, image_dim, color=True, all_frames=True):
        if image_dim != self.vgg_extractor.input_image_size:
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
                vid = self.vgg_extractor.extract_per_video_features(video_feat, len(video_feat))
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
