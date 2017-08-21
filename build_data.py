import tensorflow as tf
import numpy as np
import random
import os
from os import scandir
from PIL import Image
import matplotlib.pyplot as plt

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('raw_data_dir', 'data/celebA', 'heroes image dir')
tf.flags.DEFINE_string('tfrecord_data_dir', 'celebA_data', 'Output tfrecord files')
tf.flags.DEFINE_integer('max_num_images', 100000, 'maximum number of images to store')
tf.flags.DEFINE_integer('img_height', 64, 'image height')
tf.flags.DEFINE_integer('img_width', 64, 'image_width')
tf.flags.DEFINE_integer('img_channel', 3, 'image_channel')
tf.flags.DEFINE_string('record_name', 'celebA_images', 'celebA record name')


def read_images_dir(input_dir, max_num_images):
    for i, img_f in enumerate(os.listdir(input_dir)):
        if img_f.endswith('.png') or img_f.endswith('.jpg'):
            yield os.path.join(input_dir, img_f)
        if i > max_num_images:
            break


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _convert2float(image):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image - 1.


def convert_to_tfrecord(image_path_generator, output_dir, record_name, img_height, img_width, img_channel):
    record_path = os.path.join(output_dir, record_name + ".tfrecords")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    writer = tf.python_io.TFRecordWriter(record_path)
    i = 0
    for path in image_path_generator:
        temp = Image.open(path).resize([img_height, img_width], Image.ANTIALIAS)
        img = np.array(temp)
        height, width, channel = img.shape
        img = img[:, :, :img_channel]
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


def read_and_decode(filename_queue, batch_size,
                    height, width, channel, min_after_dequeue=10,
                    resize_height=None, resize_width=None):
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
    #if resize_height and resize_width:
    #    image = tf.image.resize_images(image, size=(resize_height, resize_height))
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


def main(_):
    #tfrecord_path = convert_to_tfrecord(read_images_dir(FLAGS.raw_data_dir, max_num_images=FLAGS.max_num_images),
    #                                    FLAGS.tfrecord_data_dir, record_name='celebA_images', img_height=FLAGS.img_height,
    #                                    img_width=FLAGS.img_width, img_channel=FLAGS.img_channel)
    #print(tfrecord_path)
    tfrecord_path = "celebA_data/celebA_images.tfrecords"
    #exit()
    filename_queue = tf.train.string_input_producer(
        string_tensor=[tfrecord_path], num_epochs=1
    )
    images = read_and_decode(filename_queue, batch_size=2, height=FLAGS.img_height, width=FLAGS.img_width,
                             channel=FLAGS.img_channel)
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
            plt.imshow(img[0])
            plt.show()
            # plt.imshow(img[1, :, :, :].reshape(32, 64), cmap='Greys_r')
            plt.imshow(img[1])
            plt.show()

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    tf.app.run()
