import tensorflow as tf
from dc_gan_model import DCGAN
from build_data import read_and_decode
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 20, 'batch_size: default:100')
tf.flags.DEFINE_string('image_size_type', 'large', 'image_size_type: default: large')
tf.flags.DEFINE_integer('image_channel', 3, 'image_channel: default:3')
tf.flags.DEFINE_integer('z_dim', 100, 'z_dim: default:100')
tf.flags.DEFINE_float('learning_rate', 2e-4, 'learning rate: default:2e-4')
tf.flags.DEFINE_integer('ngf', 512, 'number of gen filters in first conv layer')
tf.flags.DEFINE_integer('ndf', 64, 'number of dis filters in first conv layer')
tf.flags.DEFINE_string('data_path', 'dota2_data/heroes_images.tfrecords', 'Directory for storing input data')
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
    with graph.as_default():
        print(FLAGS.data_path)
        print(os.path.exists(FLAGS.data_path))
        filename_queue = tf.train.string_input_producer(
            string_tensor=[FLAGS.data_path], num_epochs=1000
        )
        input_images = read_and_decode(filename_queue, batch_size=FLAGS.batch_size)

        dcgan = DCGAN(
            batch_size=FLAGS.batch_size,
            z_dim=FLAGS.z_dim,
            input_img_height=img_height,
            input_img_width=img_width,
            output_img_height=img_height,
            output_img_width=img_width,
            num_channels=FLAGS.image_channel,
            ngf=FLAGS.ngf,
            ndf=FLAGS.ndf,
            learning_rate=FLAGS.learning_rate
        )
        generated_imgs = dcgan.g_output
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

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        try:
            while not coord.should_stop():
                z = np.random.uniform(-1, 1, [FLAGS.batch_size, FLAGS.z_dim]).astype(np.float32)
                batch_images = sess.run(input_images)
                _, _, g_loss_val, d_loss_real_val, d_loss_fake_val, summary = sess.run(
                    fetches=[g_optimizer, d_optimizer, g_loss, d_loss_real, d_loss_fake, summary_op],
                    feed_dict={
                        dcgan.input_placeholder: batch_images,
                        dcgan.z_placeholder: z
                    }
                )
                train_writer.add_summary(summary, step)
                train_writer.flush()
                if step % 5 == 0:
                    print('----------Step %d: ----------' % step)
                    print('G_loss: {}'.format(g_loss_val))
                    print('D(G(Z))_loss: {}'.format(d_loss_fake_val))
                    print('D(x)_LOSS: {}'.format(d_loss_real_val))
                    print('----------Sample img---------')
                    """
                    g_z = sess.run(
                        fetches=generated_imgs,
                        feed_dict={
                            dcgan.z_placeholder: z
                        }
                    )
                    fig = _plot(g_z)
                    plt.savefig('{}/{}.png'.format(checkpoints_dir, str(step).zfill(3)), bbox_inches='tight')
                    plt.close(fig)
                    """
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


def _plot(samples):
    fig = plt.figure(figsize=(5, 4))
    gs = gridspec.GridSpec(5, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig


if __name__ == '__main__':
    tf.app.run()

