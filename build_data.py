import tensorflow as tf
import numpy as np
import random
import os
from os import scandir
from PIL import Image
import matplotlib.pyplot as plt

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('raw_data_dir', 'dota2heroes/256x144', 'heroes image dir')
tf.flags.DEFINE_string('tfrecord_data_dir', 'dota2_data', 'Output tfrecord files')


def read_images_dir(input_dir, shuffle=True):
    image_paths = [img_f for img_f in os.listdir(input_dir) if (img_f.endswith('.png') or img_f.endswith('.jpg'))]
    if shuffle:
        shuffled_index = list(range(len(image_paths)))
        random.seed(12345)
        random.shuffle(shuffled_index)

        image_paths = [os.path.join(input_dir, image_paths[i]) for i in shuffled_index]
    return image_paths


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def convert_to_tfrecord(image_paths, output_dir, record_name, num_channel):
    record_path = os.path.join(output_dir, record_name + ".tfrecords")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    writer = tf.python_io.TFRecordWriter(record_path)
    for path in image_paths:
        temp = Image.open(path)
        img = np.array(temp)
        height, width, channel = img.shape
        img = img[:, :, :num_channel]
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image': _bytes_feature(img.tostring()),
                }
            )
        )
        writer.write(example.SerializeToString())
    writer.close()
    return record_path


def read_and_decode(filename_queue, batch_size, height, width, channel, min_after_dequeue=100):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized=serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'channel': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
        }
    )
    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image'], tf.uint8)

    image_shape = tf.stack([height, width, channel])
    image = tf.reshape(image, image_shape)
    # image = tf.image.resize_image_with_crop_or_pad(image, 33, 59)
    # image = tf.image.resize_images(image, [32, 64])
    # image_size_const = tf.constant([33, 59, 3], dtype=tf.int32)

    images = tf.train.shuffle_batch(
        [image],
        batch_size=batch_size,
        capacity=min_after_dequeue + 3*batch_size,
        num_threads=2,
        min_after_dequeue=min_after_dequeue
    )
    return images


def main(_):
    tfrecord_path = convert_to_tfrecord(read_images_dir(FLAGS.raw_data_dir),
                                        FLAGS.tfrecord_data_dir, 'heroes_images')
    # tfrecord_path = "dota2_data/heroes_images.tfrecords"
    exit()
    filename_queue = tf.train.string_input_producer(
        string_tensor=[tfrecord_path], num_epochs=10
    )
    images = read_and_decode(filename_queue, batch_size=2, height=32, width=32, channel=3)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(3):
            print(i)
            img = sess.run(images)
            print(img[0, :, :].shape)

            print('current batch')
            plt.imshow(img[0, :, :, :].reshape(32, 64), cmap='Greys_r')
            plt.show()
            plt.imshow(img[1, :, :, :].reshape(32, 64), cmap='Greys_r')
            plt.show()

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    tf.app.run()
