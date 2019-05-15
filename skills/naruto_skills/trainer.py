"""
Last updated: 11/03/2019
"""
import logging
import os
from datetime import datetime
import time

import tensorflow as tf
from sklearn import metrics

from naruto_skills import graph_utils
from naruto_skills.predictor import Predictor


class Trainer:
    def __init__(self):
        self.init_step = Trainer._init_step
        self.save_meta_data = Trainer._save_meta_data
        self.create_writers = Trainer._create_writers
        self.add_summary_nodes_to_graph = Trainer._add_summary_nodes_to_graph
        self.run_train = Trainer._run_train
        self.run_eval = Trainer._run_eval
        self.do_final_step = None
        self.best_score = -1e8

    @staticmethod
    def _init_step(sess):
        logging.info('Init global and local variables')
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

    @staticmethod
    def _save_meta_data(model, path_to_save_stuff, experiment_name):
        path_for_vocab = os.path.join(path_to_save_stuff, 'output', 'saved_models', model.__class__.__name__,
                                      experiment_name)
        model.save_vocab(path_for_vocab)

        path_for_tf_name = os.path.join(path_to_save_stuff, 'output', 'saved_models',
                                        model.__class__.__name__,
                                        experiment_name, 'tensor_name.pkl')
        model.save_tf_name(path_for_tf_name)
        return path_for_tf_name

    @staticmethod
    def _create_writers():
        # path_to_graph_train = os.path.join(path_to_save_stuff, 'output', 'summary', 'train_' + experiment_name)
        # path_to_graph_eval = os.path.join(path_to_save_stuff, 'output', 'summary', 'eval_' + experiment_name)
        # writer_train = tf.summary.FileWriter(logdir=path_to_graph_train, graph=model.graph)
        # writer_eval = tf.summary.FileWriter(logdir=path_to_graph_eval, graph=model.graph)
        # writer_eval.flush()
        # writer_train.flush()
        # logging.info('Saved graph to %s', path_to_graph_train)
        return None, None

    @staticmethod
    def _add_summary_nodes_to_graph(model):
        tf_streaming_loss, tf_streaming_loss_op = tf.metrics.mean(values=model.tf_loss)
        tf_streaming_loss_summary = tf.summary.scalar('streaming_loss', tf_streaming_loss)
        all_summary_op = tf.summary.merge_all()

    @staticmethod
    def _run_train(model, sess, X_train, y_train, pred_logging_func):
        start = time.time()
        X, y = X_train, y_train
        train_feed_dict = model.create_train_feed_dict(X, y)
        _, global_step = sess.run([model.tf_optimizing_op, model.tf_global_step], feed_dict=train_feed_dict)
        duration = time.time() - start
        if global_step % 10 == 0:
            train_loss = sess.run(model.tf_loss, feed_dict=train_feed_dict)
            logging.info('Step: %s\tTrain loss: %.5f\tDuration: %.3f s/step', global_step, train_loss, duration)

        if global_step % 500 == 0:
            if pred_logging_func is not None:
                sample_size = 3
                train_feed_dict = model.create_train_feed_dict(X[:sample_size], y[:sample_size])
                pred = sess.run(model.tf_predict, feed_dict=train_feed_dict)
                logging.info('Step: %s - Predict samples from train', global_step)
                pred_logging_func(X[:sample_size], pred, y[:sample_size])
        return global_step

    @staticmethod
    def _run_eval(global_step, model, sess, eval_iterator, scoring_func, pred_logging_func, sample_size=3):
        tf_streaming_loss_value, tf_streaming_loss_updater = model.tf_streaming_loss

        y_true = []
        y_pred = []
        # Reset streaming metrics
        sess.run(tf.local_variables_initializer())
        sample_X, sample_pred, sample_y = None, None, None
        for X_eval, y_eval in eval_iterator:
            eval_feed_dict = model.create_train_feed_dict(X_eval, y_eval)
            _, batch_pred = sess.run([tf_streaming_loss_updater, model.tf_predict], feed_dict=eval_feed_dict)
            y_pred.extend(batch_pred)
            y_true.extend(y_eval)

            sample_X = X_eval
            sample_pred = batch_pred
            sample_y = y_eval

        eval_loss = sess.run(tf_streaming_loss_value)

        eval_score = -eval_loss if scoring_func is None else scoring_func(y_true=y_true, y_pred=y_pred)

        logging.info('Step: %s - Eval loss %s', global_step, eval_loss)
        logging.info('Step: %s - Eval score: %s', global_step, eval_score)

        if pred_logging_func is not None:
            logging.info('Step: %s - Predict samples from eval', global_step)
            pred_logging_func(sample_X[:sample_size], sample_pred[:sample_size], sample_y[:sample_size])
        return eval_loss, eval_score

    @staticmethod
    def classification_report_test(predictor, test_iterator):
        logging.info('\n')
        logging.info('Predict on data test')

        y_true = []
        y_pred = []
        for X, y in test_iterator:
            y_true.extend(y)
            y_pred.extend(predictor._predict(X))
        logging.info(metrics.classification_report(y_true=y_true, y_pred=y_pred))

    def go(self, model, my_dataset, batch_size, num_epochs, eval_interval, path_to_save_stuff,
           train_phrase_func=None, eval_phrase_func=None, pred_logging_func=None, scoring_func=None,
           gpu_fraction=0.3):
        """

        :param model:
        :param dataset_path:
        :param batch_size:
        :param num_epochs:
        :param eval_interval:
        :param path_to_save_stuff: directory to save some stuff during training. You might want to set to current directory.
        TODO refactor this param because it is really messup
        :param pred_logging_func: A callable function receives 3 params (X, pred, y) and takes responsibility of logging the prediction
        :param scoring_func: The larger score is, the better model_building is. This function receive 2 params: y_true, y_pred
        :param gpu_fraction:
        :return:
        """
        if train_phrase_func is None:
            train_phrase_func = self.run_train
        if eval_phrase_func is None:
            eval_phrase_func = self.run_eval

        experiment_name = datetime.strftime(datetime.now(), '%Y-%m-%dT%H:%M:%S')
        path_to_params = self.save_meta_data(model, path_to_save_stuff, experiment_name)

        my_dataset.show_info()
        train_iterator = my_dataset.data_train.get_data_iterator(batch_size=batch_size, num_epochs=num_epochs,
                                                                 is_strictly_equal=True)
        with model.graph.as_default():
            saver = tf.train.Saver(max_to_keep=5)

            logging.info('Model contains %s parameters', graph_utils.count_trainable_variables())
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
            with tf.Session(graph=model.graph,
                            config=tf.ConfigProto(allow_soft_placement=False,
                                                  gpu_options=gpu_options)).as_default() as sess:
                self.init_step(sess)

                for X_train, y_train in train_iterator:
                    global_step = train_phrase_func(model, sess, X_train, y_train, pred_logging_func)
                    if global_step % eval_interval == 0 or global_step == 1:
                        eval_iterator = my_dataset.data_eval.get_data_iterator(batch_size=batch_size, num_epochs=1,
                                                                               is_strictly_equal=True)  # crap
                        eval_loss, eval_score = eval_phrase_func(global_step, model, sess, eval_iterator, scoring_func,
                                                                 pred_logging_func)

                        if eval_score > self.best_score:
                            self.best_score = eval_score

                            path_to_model = os.path.join(path_to_save_stuff, 'output', 'saved_models',
                                                         model.__class__.__name__,
                                                         experiment_name)
                            saved_step = global_step
                            saver.save(sess, save_path=path_to_model,
                                       global_step=saved_step,
                                       write_meta_graph=True)
                            logging.info('Gained better score on eval, saved model_building to %s', path_to_model)

                        logging.info('Best score on eval: %s', self.best_score)
                        logging.info('\n')

        if self.do_final_step is not None:
            test_iterator = my_dataset.data_test.get_data_iterator(batch_size=batch_size, num_epochs=1,
                                                                   is_strictly_equal=True)  # crap
            predictor = Predictor(path_to_params=path_to_params, path_to_model=path_to_model + '-' + str(saved_step),
                                  input_transform_func=None, preprocess_func=None)
            self.do_final_step(predictor, test_iterator)
