import tensorflow as tf
from g_model import Generator
from d_model import Discriminator


class DCGAN(object):
    def __init__(self,
                 batch_size=50,
                 z_dim=100,
                 input_img_height=33,
                 input_img_width=59,
                 output_img_height=33,
                 output_img_width=59,
                 num_channels=3,
                 ngf=512,
                 ndf=64,
                 is_training=True,
                 learning_rate=2e-4,
                 ):
        # Private member vars
        self._batch_size = batch_size
        self._learning_rate = learning_rate

        self._z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z')
        self._input_imgs = tf.placeholder(
            tf.float32,
            [batch_size, input_img_height, input_img_width, num_channels],
            name='ground_truth'
        )
        is_training = tf.placeholder_with_default(True, shape=[], name='is_training')
        self._D = Discriminator(is_training=is_training,
                                input_img_height=input_img_height,
                                input_img_width=input_img_width,
                                ndf=ndf)
        self._G = Generator(is_training=is_training,
                            output_img_height=output_img_height,
                            output_img_width=output_img_width,
                            ngf=ngf,
                            )

        self._g_output = None
        self._d_real_output = None
        self._d_fake_output = None

        self._g_loss = None
        self._d_loss_fake = None
        self._d_loss_real = None

        self._d_optimizer = None
        self._g_optimizer = None

        self._summary_op = None

        self._test_op = None

    """
    Public Interfaces
    """
    @property
    def input_placeholder(self):
        return self._input_imgs

    @property
    def z_placeholder(self):
        return self._z

    @property
    def g_output(self):
        if self._g_output is None:
            self._g_output = self._G(self._z)
        return self._g_output

    @property
    def d_real_output(self):
        if self._d_real_output is None:
            self._d_real_output = self._D(self._input_imgs)
        return self._d_real_output

    @property
    def d_fake_output(self):
        if self._d_fake_output is None:
            self._d_fake_output = self._D(self.g_output)
        return self._d_fake_output

    @property
    def d_out_shape(self):
        return self._batch_size, 1

    @property
    def g_loss(self):
        if self._g_loss is None:
            self._g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.d_fake_output,
                    labels=tf.ones(self.d_out_shape)
                )
            )
        return self._g_loss

    @property
    def d_loss_fake(self):
        if self._d_loss_fake is None:
            self._d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.d_fake_output,
                    labels=tf.zeros(self.d_out_shape)
                )
            )
        return self._d_loss_fake

    @property
    def d_loss_real(self):
        if self._d_loss_real is None:
            self._d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.d_real_output,
                    labels=tf.ones(self.d_out_shape)
                )
            )
        return self._d_loss_real

    @property
    def g_optimizer(self):
        if self._g_optimizer is None:
            self._g_optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(
                loss=self.g_loss,
                var_list=self._G.variables
            )
        return self._g_optimizer

    @property
    def d_optimizer(self):
        if self._d_optimizer is None:
            self._d_optimizer = tf.train.AdamOptimizer(self._learning_rate/150.0).minimize(
                loss=self.d_loss_real + self.d_loss_fake,
                var_list=self._D.variables
            )
        return self._d_optimizer

    @property
    def summary_op(self):
        if self._summary_op is None:
            # TODO
            tf.summary.histogram('D_FAKE', self.d_fake_output)
            tf.summary.histogram('D_REAL', self.d_real_output)

            tf.summary.image('G', self.g_output)

            self._summary_op = tf.summary.merge_all()
        return self._summary_op
