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
from tqdm import tqdm
import numpy as np

from vqa_util import question2str, answer2str
from logic_ops import rel_and, rel_or, rel_xor, rel_progression, rel_con_union, _moment


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
            # loss_meta = tf.nn.sigmoid_cross_entropy_with_logits(logits=meta_pred, labels=meta_tgt)
            # loss += tf.reduce_mean(loss_meta, axis=1) * self.config.beta

            # Classification accuracy
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return tf.reduce_mean(loss), accuracy

        # }}}

        def conv_single(img, scope='conv_single'):
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
                log.warn(scope.name)
                conv_1 = conv2d(img, conv_info[0], is_train, k_h=3, k_w=3, name='conv_1')
                conv_2 = conv2d(conv_1, conv_info[1], is_train, k_h=3, k_w=3, name='conv_2')
                conv_3 = conv2d(conv_2, conv_info[2], is_train, k_h=3, k_w=3, name='conv_3')
                conv_4 = conv2d(conv_3, conv_info[3], is_train, k_h=3, k_w=3, name='conv_4')
                conv_5 = conv2d(conv_4, conv_info[4], is_train, k_h=3, k_w=3, name='conv_5')
                flat = tf.reshape(conv_5, [self.batch_size, -1])
                fc_1 = fc(flat, 256, name='fc_1')
                # fc_2 = fc(fc_1, 256, name='fc_2')
                # fc_2 = slim.dropout(fc_2, keep_prob=0.5, is_training=is_train, scope='fc_3/')
                # fc_3 = fc(fc_2, 65, name='fc_3', activation_fn=tf.nn.sigmoid)
                fc_2 = fc(fc_1, 65, name='fc_2', activation_fn=tf.nn.sigmoid)
                # 10 + 10 + 10 + 9** + 10 (4 types of triangle) + 10 + 6*
                # return fc_3
                return fc_2

        cut = [0, 10, 20, 30, 39, 49, 59, 65]
        feature16 = [conv_single(self.img[:, i]) for i in range(16)]
        feature16_stack = tf.stack(feature16, axis=1)
        print(feature16_stack)
        self.feature16 = feature16_stack
        feature16 = [[feat[:, cut[i]: cut[i + 1]] for i in range(len(cut) - 1)] for feat in feature16]

        # loss = (_sum(feat[2]) != 1) + (_moment(feat[2]) != _sum(feat[3]))
        loss = tf.reduce_mean((tf.reduce_sum(feature16_stack[:, :, cut[2]: cut[3]], axis=2) - 1) ** 2) \
            + tf.reduce_mean((_moment(feature16_stack[:, :, cut[2]: cut[3]], axis=2) -
                              tf.reduce_sum(feature16_stack[:, :, cut[3]: cut[4]], axis=2)) ** 2)

        score = []
        for choice in tqdm(range(8, 16)):
            sk = []
            for i in range(len(cut) - 1):
                feature9 = [feature16[j][i] for j in list(range(8)) + [choice]]
                # print(feature9[0].shape)
                sk.extend([rel_and(feature9), rel_or(feature9), rel_xor(feature9)])
                if i != 3 and i != 6:
                    sk.append(rel_progression(feature9))
                if i != 3:
                    sk.append(rel_con_union(feature9))
                # return
            score.append(tf.reduce_sum(sk, axis=0))
            print(len(tf.get_default_graph().get_operations()))

        logits = tf.stack(score, axis=1)
        meta_pred = 0  # tf.reduce_sum(tf.stack(meta_pred), axis=0)
        log.warn(logits.shape)
        # log.warn(meta_pred.shape)
        self.loss, self.accuracy = build_loss(logits, meta_pred, self.a, self.meta_tgt)
        self.loss += loss
        self.all_preds = tf.nn.softmax(logits)

        # Add summaries
        def draw_iqa(img, target_a, pred_a, meta, attr):
            fig, axes = tfplot.subplots(nrows=6, ncols=6, figsize=(20, 20))
            cut = [0, 10, 20, 30, 39, 49, 59, 65]
            attr = [[np.pad(attr[i, cut[j]: cut[j + 1]], (0, 10 - cut[j + 1] + cut[j]), 'constant')
                     for j in range(len(cut) - 1)] for i in range(16)]
            for i in range(8):
                axes[i // 3 * 2, i % 3].imshow(img[i, :, :, 0])
                axes[i // 3 * 2 + 1, i % 3].imshow(attr[i], vmin=0, vmax=1)
            for i in range(8, 16):
                axes[(i - 8) // 3 * 2, (i - 8) % 3 + 3].imshow(img[i, :, :, 0])
                axes[(i - 8) // 3 * 2 + 1, (i - 8) % 3 + 3].imshow(attr[i], vmin=0, vmax=1)
            for i in range(6):
                for j in range(6):
                    axes[i, j].axis('off')
            fig.suptitle('target: {}. predicted: {}. meta: {}.'
                         .format(np.argmax(target_a), np.argmax(pred_a), meta.astype(np.int)), size=20)
            return fig

        try:
            tfplot.summary.plot_many('IQA/',
                                     draw_iqa, [self.img, self.a, self.all_preds, self.meta_tgt, self.feature16],
                                     max_outputs=10,
                                     collections=["plot_summaries"])
        except:
            pass

        tf.summary.scalar("loss/accuracy", self.accuracy)
        tf.summary.scalar("loss/cross_entropy", self.loss)
        log.warn('Successfully loaded the model.')


if __name__ == '__main__':
    import argparse, os

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model', type=str, default='X', choices=['rn', 'baseline', 'X'])
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

    '''with open('tmp.txt', 'w') as f:
        print('Number of operators: %d.' % len(tf.get_default_graph().get_operations()), file=f)
        for op in tf.get_default_graph().get_operations():
            print(op.name, op.values(), sep='\n\t', file=f)'''
