import numpy as np
import tensorflow as tf

from lib.kaffe.tensorflow import Network


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


class VGG_face_model():
    def __init__(self):
        self.input_image_size = 224
        self.batch_size = 128
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
        self.model.load('model.npy', self.sess)
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
        print('features shape', features.shape)
        features = np.transpose(features, [1, 0, 2])
        print('features shape after transpose', features.shape)
        return features

    def extract_per_video_features(self, data, frames):
        """
        data => (frames, h, w, channels)
        :param data:
        :param frames:
        :return:
        """
        features = []

        batch_size = 100
        for x in range(0, frames, batch_size):
            feed = {self.images: data[x:x + batch_size]}
            frame_features = self.sess.run([self.feature_layer], feed_dict=feed)
            # print('frame_features shape', frame_features.shape)
            features.append(frame_features)
        features = np.concatenate(features, axis=0)
        # print('features shape', features.shape)
        # features = np.transpose(features, [1, 0, 2])
        # print('features shape after transpose', features.shape)
        return features

    def extract_per_frame_features(self, data):
        """
        data => (h, w, channels)
        :param data:
        :param frames:
        :return:
        """
        # features = []
        feed = {self.images: data}
        frame_features = self.sess.run([self.feature_layer], feed_dict=feed)
        # print('frame_features shape', frame_features.shape)
        return frame_features
