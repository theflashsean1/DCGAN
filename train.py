import tensorflow as tf
from dc_gan_model import DCGAN
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 50, 'batch_size: default:100')
tf.flags.DEFINE_integer('image_size_type', 'small', 'image_size_type: default: small')
tf.flags.DEFINE_integer('image_channel', 3, 'image_channel: default:3')
tf.flags.DEFINE_integer('z_dim', 100, 'z_dim: default:100')
tf.flags.DEFINE_float('learning_rate', 2e-4, 'learning rate: default:2e-4')
tf.flags.DEFINE_integer('ngf', 512, 'number of gen filters in first conv layer')
tf.flags.DEFINE_integer('ndf', 64, 'number of dis filters in first conv layer')
tf.flags.DEFINE_string('data_dir', '/tmp/tensorflow/mnist/input_data', 'Directory for storing input data')
tf.flags.DEFINE_string('load_model', None, 'folder of saved model that you wish to continue training '
                                           '(e.g. 20170602-1936), default=None')
tf.flags.DEFINE_integer('max_num_steps', 2000, 'Number of steps to train')
tf.flags.DEFINE_integer('num_steps_run', 200, 'Number of steps to run per this script call')

IMG_SIZE_MAP = {
    'small': (33, 59),
    'medium': (115, 205),
    'large': (144, 256)
}


def main(_):
    if FLAGS.load_model is not None:
        checkpoints_dir = "checkpoints/" + FLAGS.load_model
    else:
        current_time = datetime.now().strftime("%Y%m%d-%H")
        checkpoints_dir = "checkpoints/{}".format(current_time)
        try:
            os.makedirs(checkpoints_dir)
        except os.error:
            pass

    img_height, img_width = IMG_SIZE_MAP[FLAGS.image_size_type]
    graph = tf.Graph()

