from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

try:
    import tfplot
except:
    pass

from ops import conv2d, fc
from util import log

from vqa_util import question2str, answer2str


class Model(object):

    def __init__(self, config, debug_information=False, is_train=True):
        self.debug = debug_information

        self.config = config
        self.batch_size = self.config.batch_size
        self.img_num = self.config.data_info[0]
        self.img_size = self.config.data_info[1]
        self.c_dim = self.config.data_info[3]
        self.a_dim = self.config.data_info[4]
        self.meta_dim = self.config.data_info[5]
        self.conv_info = self.config.conv_info

        # create placeholders for the input
        self.img = tf.placeholder(name='img', dtype=tf.float32,
                                  shape=[self.batch_size, self.img_num, self.img_size, self.img_size, self.c_dim])
        self.a = tf.placeholder(name='a', dtype=tf.float32, shape=[self.batch_size, self.a_dim])
        self.meta_tgt = tf.placeholder(name='meta_tgt', dtype=tf.float32, shape=[self.batch_size, self.meta_dim])

        self.is_training = tf.placeholder_with_default(bool(is_train), [], name='is_training')

        self.build(is_train=is_train)

    def get_feed_dict(self, batch_chunk, step=None, is_training=None):
        fd = {
            self.img: batch_chunk['img'],  # [B, h, w, c]
            self.a: batch_chunk['a'],  # [B, m]
            self.meta_tgt: batch_chunk['meta_target'],
        }
        if is_training is not None:
            fd[self.is_training] = is_training

        return fd

    def build(self, is_train=True):
        conv_info = self.conv_info

        # build loss and accuracy {{{
        def build_loss(logits, meta_pred, labels, meta_tgt):
            # Cross-entropy loss
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss_meta = tf.nn.sigmoid_cross_entropy_with_logits(logits=meta_pred, labels=meta_tgt)
            loss += tf.reduce_mean(loss_meta, axis=1) * self.config.beta

            # Classification accuracy
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return tf.reduce_mean(loss), accuracy

        # }}}

        def concat_coor(o, i, d=3):
            coor = tf.tile(tf.expand_dims([float(int(i / d)) / d, (i % d) / d], axis=0), [self.batch_size, 1])
            # print(o.shape, coor.shape)
            o = tf.concat([o, tf.to_float(coor)], axis=1)
            return o

        def g_theta(o_i, o_j, scope='g_theta'):
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
                log.warn(scope.name)
                g_1 = fc(tf.concat([o_i, o_j], axis=1), 512, name='g_1')
                g_2 = fc(g_1, 512, name='g_2')
                g_3 = fc(g_2, 512, name='g_3')
                g_4 = fc(g_3, 512, name='g_4')
                return g_4

        def conv_single(img, scope='conv_single'):
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
                log.warn(scope.name)
                conv_1 = conv2d(img,    conv_info[0], is_train, k_h=3, k_w=3, name='conv_1')
                conv_2 = conv2d(conv_1, conv_info[1], is_train, k_h=3, k_w=3, name='conv_2')
                conv_3 = conv2d(conv_2, conv_info[2], is_train, k_h=3, k_w=3, name='conv_3')
                conv_4 = conv2d(conv_3, conv_info[3], is_train, k_h=3, k_w=3, name='conv_4')
                conv_5 = conv2d(conv_4, conv_info[4], is_train, k_h=3, k_w=3, name='conv_5')
                feat_1 = tf.reshape(conv_5, [self.batch_size, -1])
                # TODO: 254 ? 256
                feat_2 = fc(feat_1, 254, name='feat')
                return feat_2

        def f_phi(g, scope='f_phi'):
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
                log.warn(scope.name)
                fc_1 = fc(g, 256, name='fc_1')
                fc_2 = fc(fc_1, 256, name='fc_2')
                fc_2 = slim.dropout(fc_2, keep_prob=0.5, is_training=is_train, scope='fc_3/')
                fc_3 = fc(fc_2, 13, activation_fn=None, name='fc_3')
                return fc_3

        feature = [concat_coor(conv_single(self.img[:, i]), i if i < 8 else 8) for i in range(16)]

        g_dict = dict()
        score = []
        meta_pred = []
        for choice in range(8, 16):
            panel = list(range(8)) + [choice]
            g_collection = []
            for i in panel:
                for j in panel:
                    if (i, j) not in g_dict:
                        log.info('g(o_%d,o_%d)' % (i, j))
                        g_dict[(i, j)] = g_theta(feature[i], feature[j])
                        g_collection.append(g_dict[(i, j)])
            f = f_phi(tf.reduce_sum(tf.stack(g_collection), axis=0))
            score.append(f[:, :1])
            meta_pred.append(f[:, 1:])

        logits = tf.concat(score, axis=1)
        meta_pred = tf.reduce_sum(tf.stack(meta_pred), axis=0)
        log.warn(logits.shape)
        log.warn(meta_pred.shape)
        self.loss, self.accuracy = build_loss(logits, meta_pred, self.a, self.meta_tgt)
        self.all_preds = tf.nn.softmax(logits)

        # Add summaries
        # TODO: add visualization
        def draw_iqa(img, target_a, pred_a):
            fig, ax = tfplot.subplots(figsize=(6, 6))
            ax.imshow(img)
            ax.set_xlabel(answer2str(target_a) + answer2str(pred_a, 'Predicted'))
            return fig

        try:
            tfplot.summary.plot_many('IQA/',
                                     draw_iqa, [self.img, self.a, self.all_preds],
                                     max_outputs=20,
                                     collections=["plot_summaries"])
        except:
            pass

        tf.summary.scalar("loss/accuracy", self.accuracy)
        tf.summary.scalar("loss/cross_entropy", self.loss)
        log.warn('Successfully loaded the model.')


if __name__ == '__main__':
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model', type=str, default='rn', choices=['rn', 'baseline'])
    parser.add_argument('--prefix', type=str, default='default')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--dataset_path', type=str, default='PGM/interpolation')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lr_weight_decay', action='store_true', default=False)
    config = parser.parse_args()

    path = os.path.join('datasets', config.dataset_path)

    import pgm as dataset

    config.data_info = dataset.get_data_info()
    config.conv_info = dataset.get_conv_info()
    dataset_train, dataset_test = dataset.create_default_splits(path)

    Model(config)
