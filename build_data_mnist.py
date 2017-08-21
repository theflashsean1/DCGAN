import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
# from build_data import read_and_decode
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('data_dir', '/tmp/tensorflow/mnist/input_data', 'Directory for storing input data')
tf.flags.DEFINE_string('tfrecord_data_dir', 'mnist_data', 'Output tfrecord files')
tf.flags.DEFINE_string('record_name', 'mnist_images', 'mnist record name')


def read_and_decode(filename_queue, batch_size, height, width, channel, min_after_dequeue=10):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized=serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
        }
    )
    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image'], tf.uint8)

    image_shape = tf.stack([height, width, channel])
    image = tf.reshape(image, image_shape)
    image = tf.image.resize_images(image, size=(32, 32))
    image = _convert2float(image)
    # image.set_shape(image_shape)
    # image = tf.reshape(image, image_shape)
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


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _convert2float(image):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image


def convert_to_tfrecord(flat_images, output_dir, record_name):
    record_path = os.path.join(output_dir, record_name + ".tfrecords")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    writer = tf.python_io.TFRecordWriter(record_path)
    i = 0
    for flat_image in flat_images:
        img = (flat_image.reshape([28, 28, 1])) #.astype(np.uint8)
        # print(img)
        img = np.multiply(img, 255).astype(np.uint8)
        # plt.imshow(img.reshape([28, 28]), cmap='Greys_r')
        # plt.show()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image': _bytes_feature(img.tostring()),
                }
            )
        )
        writer.write(example.SerializeToString())
        i += 1
    writer.close()
    print("{} images written in tfrecord".format(i))
    return record_path


def main(_):
    #mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    #train_imgs = mnist.train.images
    #tfrecord_path = convert_to_tfrecord(train_imgs, FLAGS.tfrecord_data_dir, FLAGS.record_name)
    # print(tfrecord_path)
    tfrecord_path = "mnist_data/mnist_images.tfrecords"
    # exit()
    filename_queue = tf.train.string_input_producer(
        string_tensor=[tfrecord_path]
    )
    images = read_and_decode(filename_queue, batch_size=2, height=28, width=28,
                             channel=1)
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
            # plt.imshow(img[0, :, :, :].reshape(32, 64), cmap='Greys_r')
            plt.imshow(img[0].reshape(32, 32), cmap='Greys_r')
            plt.show()
            # plt.imshow(img[1, :, :, :].reshape(32, 64), cmap='Greys_r')
            plt.imshow(img[1].reshape(32, 32), cmap='Greys_r')
            plt.show()

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run()