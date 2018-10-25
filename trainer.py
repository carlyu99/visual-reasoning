from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from pprint import pprint

import tensorflow as tf
from six.moves import xrange

from input_ops import create_input_ops
from util import log


class Trainer(object):

    @staticmethod
    def get_model_class(model_name):
        if model_name == 'baseline':
            from model_baseline import Model
        elif model_name == 'rn':
            from model_rn import Model
        elif model_name == 'X':
            from model_X import Model
        elif model_name == 'conv':
            from model_conv import Model
        else:
            raise ValueError(model_name)
        return Model

    def __init__(self, config, dataset, dataset_test):
        self.config = config
        hyper_parameter_str = config.dataset_path + '_lr_' + str(config.learning_rate) + '_b_' + str(config.beta)
        self.train_dir = './train_dir/%s-%s-%s-%s' % (
            config.model,
            config.prefix,
            hyper_parameter_str,
            time.strftime("%Y%m%d-%H%M%S")
        )

        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)
        log.infov("Train Dir: %s", self.train_dir)

        # --- input ops ---
        self.batch_size = config.batch_size

        _, self.batch_train = create_input_ops(dataset, self.batch_size, is_training=True)
        _, self.batch_test = create_input_ops(dataset_test, self.batch_size, is_training=False)

        # --- create model ---
        Model = self.get_model_class(config.model)
        log.infov("Using Model class : %s", Model)
        self.model = Model(config)

        # --- optimizer ---
        # TODO: (delete this) self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)
        self.global_step = tf.train.get_or_create_global_step(graph=None)
        self.learning_rate = config.learning_rate
        if config.lr_weight_decay:
            self.learning_rate = tf.train.exponential_decay(
                self.learning_rate,
                global_step=self.global_step,
                decay_steps=1000,
                decay_rate=0.5,
                staircase=True,
                name='decaying_learning_rate'
            )

        self.check_op = tf.no_op()

        self.optimizer = tf.contrib.layers.optimize_loss(
            loss=self.model.loss,
            global_step=self.global_step,
            learning_rate=self.learning_rate,
            optimizer=tf.train.AdamOptimizer,
            clip_gradients=20.0,
            name='optimizer_loss'
        )

        self.summary_op = tf.summary.merge_all()
        try:
            # TODO
            # raise IOError
            import tfplot
            self.plot_summary_op = tf.summary.merge_all(key='plot_summaries')
        except:
            pass

        self.saver = tf.train.Saver(max_to_keep=1000)
        self.summary_writer = tf.summary.FileWriter(self.train_dir)

        self.checkpoint_secs = 600  # 10 min

        self.supervisor = tf.train.Supervisor(
            logdir=self.train_dir,
            is_chief=True,
            saver=None,
            summary_op=None,
            summary_writer=self.summary_writer,
            save_summaries_secs=300,
            save_model_secs=self.checkpoint_secs,
            global_step=self.global_step,
        )

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            # intra_op_parallelism_threads=1,
            # inter_op_parallelism_threads=1,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1},
        )
        self.session = self.supervisor.prepare_or_wait_for_session(config=session_config)

        '''variable_names = [v.name for v in tf.trainable_variables()]
        values = self.session.run(variable_names)
        for k, v in zip(variable_names, values):
            print("Variable: ", k)
            print("Shape: ", v.shape)
            # print(v)'''
        for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            print(variable)
        # exit()

        self.ckpt_path = config.checkpoint
        if self.ckpt_path is not None:
            log.info("Checkpoint path: %s", self.ckpt_path)
            self.saver.restore(self.session, self.ckpt_path)
            log.info("Loaded the pretrain parameters from the provided checkpoint path")

    def train(self):
        log.infov("Training Starts!")
        pprint(self.batch_train)

        max_steps = 10000000

        output_save_step = 2000

        for s in xrange(max_steps):
            step, accuracy, summary, loss, step_time = \
                self.run_single_step(self.batch_train, step=s, is_train=True)

            if s % 10 == 0:
                # periodic inference
                accuracy_test, loss_test = self.run_test(self.batch_test, is_train=False)

                self.log_step_message(step, accuracy, accuracy_test, loss, loss_test, step_time)

            self.summary_writer.add_summary(summary, global_step=step)

            if s % output_save_step == 0:
                log.infov("Saved checkpoint at %d", s)
                save_path = self.saver.save(self.session,
                                            os.path.join(self.train_dir, 'model'),
                                            global_step=step)

    def run_single_step(self, batch, step=None, is_train=True):
        _start_time = time.time()

        batch_chunk = self.session.run(batch)

        fetch = [self.global_step, self.model.accuracy, self.summary_op,
                 self.model.loss, self.check_op, self.optimizer]

        try:
            if step is not None and (step % 200 == 0):
                fetch += [self.plot_summary_op]
        except:
            pass

        fetch_values = self.session.run(fetch, feed_dict=self.model.get_feed_dict(batch_chunk, step=step))
        [step, accuracy, summary, loss] = fetch_values[:4]

        try:
            if self.plot_summary_op in fetch:
                summary += fetch_values[-1]
        except:
            pass

        _end_time = time.time()

        return step, accuracy, summary, loss, (_end_time - _start_time)

    def run_test(self, batch, is_train=False, repeat_times=8):

        batch_chunk = self.session.run(batch)

        accuracy_test, loss_test = self.session.run(
            [self.model.accuracy, self.model.loss], feed_dict=self.model.get_feed_dict(batch_chunk, is_training=False)
        )

        return accuracy_test, loss_test

    def log_step_message(self, step, accuracy, accuracy_test, loss, loss_test, step_time, is_train=True):
        if step_time == 0:
            step_time = 0.001
        log_fn = (is_train and log.info or log.infov)
        log_fn((" [{split_mode:5s} step {step:4d}] " +
                "Train loss: {loss:.5f} " +
                "Acc.: {accuracy:.2f} " +
                "Test loss: {loss_test:.5f} "
                "Acc.: {accuracy_test:.2f} " +
                "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) "
                ).format(split_mode=(is_train and 'train' or 'val'),
                         step=step,
                         loss=loss,
                         loss_test=loss_test,
                         accuracy=accuracy * 100,
                         accuracy_test=accuracy_test * 100,
                         sec_per_batch=step_time,
                         instance_per_sec=self.batch_size / step_time
                         )
               )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model', type=str, default='X', choices=['rn', 'conv', 'X'])
    parser.add_argument('--prefix', type=str, default='default')
    parser.add_argument('--checkpoint', type=str, default=None)
    # parser.add_argument('--dataset_path', type=str, default='PGM/interpolation')
    parser.add_argument('--dataset_path', type=str, default='PGM/neutral')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lr_weight_decay', action='store_true', default=False)
    parser.add_argument('--beta', type=float, default=0.)
    config = parser.parse_args()

    path = os.path.join('datasets', config.dataset_path)

    import pgm as dataset

    config.data_info = dataset.get_data_info()
    config.conv_info = dataset.get_conv_info()
    dataset_train, dataset_test = dataset.create_default_splits(path)

    trainer = Trainer(config, dataset_train, dataset_test)

    log.warning("dataset: %s, learning_rate: %f", config.dataset_path, config.learning_rate)
    trainer.train()


if __name__ == '__main__':
    main()
