import tensorflow as tf
import ops


class Discriminator:
    def __init__(self, is_training,  input_img_height, input_img_width, ndf=64, name='discriminator'):
        self._name = name
        self._is_training = is_training
        self._input_img_height = input_img_height
        self._input_img_width = input_img_width
        self._reuse = False
        self._ndf = ndf
        self._variables = None

    def __call__(self, input_):
        with tf.variable_scope(self._name):  # (64, 64, 3)
            num_conv = 4
            h_heights = [int(self._input_img_height/(2 ** i)) for i in range(1, num_conv+1, 1)]
            h_widths = [int(self._input_img_width/(2 ** i)) for i in range(1, num_conv+1, 1)]
            print(h_heights)
            print(h_widths)
            h0 = tf.nn.relu(
                ops.conv2d(input_, self._ndf, reuse=self._reuse, name='d_conv0')  # (64, 32, 32, 64)
            )
            h1 = tf.nn.relu(
                ops.conv2d(h0, self._ndf * 2, reuse=self._reuse, name='d_conv1'),  # (64, 16, 16, 128)
            )
            h2 = tf.nn.relu(
                ops.conv2d(h1, self._ndf * 4, reuse=self._reuse, name='d_conv2'),  # (64, 8, 8, 256)
            )
            h3 = tf.nn.relu(
                ops.conv2d(h2, self._ndf * 8, reuse=self._reuse, name='d_conv3'),  # (64, 4, 4, 512)
            )
            """
            h1 = ops.leaky_relu(
                ops.batch_norm(
                    ops.conv2d(h0, self._ndf * 2, reuse=self._reuse, name='d_conv1'),  # (64, 16, 16, 128)
                    is_training=self._is_training,
                    reuse=self._reuse,
                    name_scope='d_bn1'
                )

            )
            h2 = ops.leaky_relu(
                ops.batch_norm(
                    ops.conv2d(h1, self._ndf * 4, reuse=self._reuse, name='d_conv2'),  # (64, 8, 8, 256)
                    is_training=self._is_training,
                    reuse=self._reuse,
                    name_scope='d_bn2'
                )
            )
            h3 = ops.leaky_relu(
                ops.batch_norm(
                    ops.conv2d(h2, self._ndf * 8, reuse=self._reuse, name='d_conv3'),  # (64, 4, 4, 512)
                    is_training=self._is_training,
                    reuse=self._reuse,
                    name_scope='d_bn3'
                )
            )
            """
            h3_shape = h3.get_shape().as_list()
            h3_ = tf.reshape(h3, [h3_shape[0], self._ndf*8*h_heights[-1]*h_widths[-1]])  # (64, 4*4*512)
            h4 = tf.nn.sigmoid(
                ops.fc(h3_, 1, reuse=self._reuse, name='d_fc4')  # (64, 1)
            )
        self._reuse = True
        return h4

    @property
    def variables(self):
        if not self._variables:
            self._variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._name)
            # print(1)
        return self._variables
