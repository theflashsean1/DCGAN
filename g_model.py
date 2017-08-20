import tensorflow as tf
import ops


class Generator:
    def __init__(self, is_training, output_img_height, output_img_width, ngf=512, name='generator'):
        self._name = name
        self._is_training = is_training
        self._output_img_height = output_img_height
        self._output_img_width = output_img_width
        self._reuse = False
        self._ngf = ngf
        self._variables = None

    def __call__(self, z):
        with tf.variable_scope(self._name):
            # Four stride=2 deconvolution
            num_deconv = 4
            h_heights = [int(self._output_img_height/(2 ** i)) for i in range(num_deconv, 0, -1)]
            h_widths = [int(self._output_img_width/(2 ** i)) for i in range(num_deconv, 0, -1)]
            print(h_heights)
            print(h_widths)
            z_ = ops.fc(z, self._ngf*h_heights[0]*h_widths[0], reuse=self._reuse, name='g_fc0')
            h0 = tf.nn.relu(
                ops.batch_norm(
                    tf.reshape(z_, [-1, h_heights[0], h_widths[0], self._ngf]),
                    is_training=self._is_training,
                    reuse=self._reuse,
                    name_scope='g_bn0'
                )
            )
            h1 = tf.nn.relu(
                ops.batch_norm(
                    ops.deconv2d(h0, int(self._ngf / 2), reuse=self._reuse, name='g_conv1'),
                    is_training=self._is_training,
                    reuse=self._reuse,
                    name_scope='g_bn1'
                )
            )
            h2 = tf.nn.relu(
                ops.batch_norm(
                    ops.deconv2d(h1, int(self._ngf / 4), reuse=self._reuse, name='g_conv2'),
                    is_training=self._is_training,
                    reuse=self._reuse,
                    name_scope='g_bn2'
                )
            )
            h3 = tf.nn.relu(
                ops.batch_norm(
                    ops.deconv2d(h2, int(self._ngf / 8), reuse=self._reuse, name='g_conv3'),
                    is_training=self._is_training,
                    reuse=self._reuse,
                    name_scope='g_bn3'
                )
            )
            h4 = tf.nn.tanh(
                ops.deconv2d(h3, 1, reuse=self._reuse, name='g_conv4')
            )
        self._reuse = True
        return h4

    @property
    def variables(self):
        if not self._variables:
            self._variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._name)
        return self._variables
