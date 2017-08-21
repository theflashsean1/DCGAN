import tensorflow as tf
from dc_gan_model import DCGAN
from build_data import read_and_decode
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

""""""
from glob import glob
from utils import *
""""""

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 64, 'batch_size: default:100')
tf.flags.DEFINE_integer('plot_num_rows', 8, 'number of rows displayed in plot')
tf.flags.DEFINE_string('image_dims', 'celebrity', 'image_size_type: default: large')
tf.flags.DEFINE_integer('z_dim', 100, 'z_dim: default:100')
tf.flags.DEFINE_float('g_learning_rate', 2e-4, 'learning rate: default:2e-4')
tf.flags.DEFINE_float('d_learning_rate', 2e-4, 'learning rate: default:2e-4')
tf.flags.DEFINE_float('beta1', 0.5, "Momentum term of Adam")
tf.flags.DEFINE_integer('ngf', 512, 'number of gen filters in first conv layer')
tf.flags.DEFINE_integer('ndf', 64, 'number of dis filters in first conv layer')
tf.flags.DEFINE_string('data_path', 'celebA_data/celebA_images.tfrecords', 'Directory for storing input data')
tf.flags.DEFINE_string('load_model', None, 'folder of saved model that you wish to continue training '
                                           '(e.g. 20170602-1936), default=None')
tf.flags.DEFINE_integer('sample_interval', 100, 'plot intermediate results')

IMG_SIZE_MAP = {
    'dota_small': (33, 59, 1),
    'dota_medium': (115, 205, 1),
    'dota_large': (144, 256, 1),
    'celebrity': (64, 64, 3),
    'mnist': (28, 28, 1)
}
IMG_RESIZE_MAP={
    'dota_small': (None, None),
    'dota_medium': (None, None),
    'dota_large': (None, None),
    'celebrity': (None, None),
    'mnist': (32, 32)
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
    plot_num_rows, plot_num_cols = FLAGS.plot_num_rows, int(FLAGS.batch_size/FLAGS.plot_num_rows)
    img_height, img_width, img_channel = IMG_SIZE_MAP[FLAGS.image_dims]
    # resize_height, resize_width = IMG_RESIZE_MAP[FLAGS.image_dims]
    graph = tf.Graph()
    with graph.as_default():
        print(FLAGS.data_path)
        print(os.path.exists(FLAGS.data_path))
        filename_queue = tf.train.string_input_producer(
            string_tensor=[FLAGS.data_path]
        )
        # input_images = read_and_decode(filename_queue, batch_size=FLAGS.batch_size,
        #                               height=img_height, width=img_width, channel=img_channel,
        #                               )

        # if resize_height:
        #    img_height = resize_height
        # if resize_width:
        #    img_width = resize_width

        dcgan = DCGAN(
            batch_size=FLAGS.batch_size,
            z_dim=FLAGS.z_dim,
            input_img_height=img_height,
            input_img_width=img_width,
            output_img_height=img_height,
            output_img_width=img_width,
            num_channels=img_channel,
            ngf=FLAGS.ngf,
            ndf=FLAGS.ndf,
            g_learning_rate=FLAGS.g_learning_rate,
            d_learning_rate=FLAGS.d_learning_rate,
            beta1=FLAGS.beta1
        )
        generated_imgs = dcgan.g_output
        d_out_real = dcgan.d_real_output
        d_out_fake = dcgan.d_fake_output
        g_loss, d_loss_real, d_loss_fake = dcgan.g_loss, dcgan.d_loss_real, dcgan.d_loss_fake
        g_optimizer, d_optimizer = dcgan.g_optimizer, dcgan.d_optimizer

        summary_op = dcgan.summary_op
        train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        if FLAGS.load_model is not None:
            checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
            meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
            restore = tf.train.import_meta_graph(meta_graph_path)
            restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
            step = int(meta_graph_path.split("-")[2].split(".")[0])
        else:
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            step = 0
        # saver.save(sess, checkpoints_dir)
        # exit()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        try:
            while not coord.should_stop():
                z = np.random.uniform(-1, 1, [FLAGS.batch_size, FLAGS.z_dim]).astype(np.float32)
                print(z.shape)
                # batch_images = sess.run(input_images)
                """"""
                data = glob(os.path.join(
                    "data", "celebA", "*.jpg"))
                batch_files = data[step * FLAGS.batch_size:(step + 1) * FLAGS.batch_size]
                batch_images = [
                    get_image(batch_file,
                              input_height=108,
                              input_width=108,
                              resize_height=64,
                              resize_width=64,
                              crop=True,
                              grayscale=False) for batch_file in batch_files]
                """"""
                _, _, _, g_loss_val, d_loss_real_val, d_loss_fake_val, d_out_real_val, d_out_fake_val, summary = \
                sess.run(
                    fetches=[d_optimizer, g_optimizer, g_optimizer, g_loss, d_loss_real, d_loss_fake, d_out_real, d_out_fake, summary_op],
                    feed_dict={
                        dcgan.input_placeholder: batch_images,
                        dcgan.z_placeholder: z
                    }
                )
                train_writer.add_summary(summary, step)
                train_writer.flush()
                print('Step {}, d_loss_fake:{}, d_loss_real:{}, g_loss: {}'.format(
                    step, d_loss_fake_val, d_loss_real_val, g_loss_val))

                if step % FLAGS.sample_interval == 0:
                    """
                    fig = _plot(z, 1, FLAGS.z_dim)
                    plt.savefig('{}/z_{}.png'.format(checkpoints_dir, str(step).zfill(3)), bbox_inches='tight')
                    plt.close(fig)
                    """
                    plt.imshow(batch_images[0])
                    plt.show()
                    fig = _plot(batch_images, plot_num_rows, plot_num_cols, img_height, img_width, img_channel)
                    plt.savefig('{}/input_{}.png'.format(checkpoints_dir, str(step).zfill(3)), bbox_inches='tight')
                    plt.close(fig)
                    print('----------Step %d: ----------' % step)
                    print('D_real_out: {}'.format(d_out_real_val))
                    print('D_fake_out: {}'.format(d_out_fake_val))
                    print('G_loss: {}'.format(g_loss_val))
                    print('D(G(Z))_loss: {}'.format(d_loss_fake_val))
                    print('D(x)_LOSS: {}'.format(d_loss_real_val))
                    print('----------Sample img---------')

                    g_z = sess.run(
                        fetches=generated_imgs,
                        feed_dict={
                            dcgan.z_placeholder: z
                        }
                    )
                    plt.imshow(g_z[0])
                    plt.show()
                    fig = _plot(g_z, plot_num_rows, plot_num_cols, img_height, img_width, img_channel)
                    plt.savefig('{}/{}.png'.format(checkpoints_dir, str(step).zfill(3)), bbox_inches='tight')
                    plt.close(fig)
                step += 1
        except KeyboardInterrupt:
            # save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
            # print("Training Interrupted.  Model saved in file: %s" % save_path)
            coord.request_stop()
        finally:
            save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
            print("Model saved in file: %s" % save_path)
            coord.request_stop()
            coord.join(threads)


def _plot(samples, num_rows, num_cols, height, width, channel):
    fig = plt.figure(figsize=(num_rows, num_cols))
    gs = gridspec.GridSpec(num_rows, num_cols)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if channel == 1:
            plt.imshow(sample.reshape([height, width]), cmap='Greys_r')
        elif channel == 3:
            plt.imshow(sample)
        else:
            raise AttributeError("{} channel not allowed".format(channel))
    return fig


if __name__ == '__main__':
    tf.app.run()

