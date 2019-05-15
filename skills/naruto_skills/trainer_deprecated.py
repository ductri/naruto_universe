import logging
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from naruto_skills.dataset import Dataset
from naruto_skills.text_transformer import TextTransformer
from naruto_skills import graph_utils


def __save_meta_data(model, path_to_save_stuff, experiment_name):
    path_for_vocab = os.path.join(path_to_save_stuff, 'output', 'saved_models', model.__class__.__name__,
                                  experiment_name)
    model.save_vocab(path_for_vocab)

    path_for_tf_name = os.path.join(path_to_save_stuff, 'output', 'saved_models',
                                    model.__class__.__name__,
                                    experiment_name, 'tensor_name.pkl')
    model.save_tf_name(path_for_tf_name)


def __create_writers():
    # path_to_graph_train = os.path.join(path_to_save_stuff, 'output', 'summary', 'train_' + experiment_name)
    # path_to_graph_eval = os.path.join(path_to_save_stuff, 'output', 'summary', 'eval_' + experiment_name)
    # writer_train = tf.summary.FileWriter(logdir=path_to_graph_train, graph=model.graph)
    # writer_eval = tf.summary.FileWriter(logdir=path_to_graph_eval, graph=model.graph)
    # writer_eval.flush()
    # writer_train.flush()
    # logging.info('Saved graph to %s', path_to_graph_train)
    return None, None


def __add_some_node_to_graph(model):
    tf_streaming_loss, tf_streaming_loss_op = tf.metrics.mean(values=model.tf_loss)
    tf_streaming_loss_summary = tf.summary.scalar('streaming_loss', tf_streaming_loss)
    all_summary_op = tf.summary.merge_all()


def __run_train(model, sess, X_train, y_train):
    X, y = X_train, y_train
    train_feed_dict = model.create_train_feed_dict(X, y)
    _, global_step = sess.run([model.tf_optimizer, model.tf_global_step], feed_dict=train_feed_dict)
    if global_step % 10 == 0:
        train_loss = sess.run(model.tf_loss, feed_dict=train_feed_dict)
        logging.info('Step: %s - Train loss: %s', global_step, train_loss)
    return global_step


def __run_eval(global_step, model, sess, eval_iterator, tf_streaming_loss, scoring_func):
    tf_streaming_loss_value, tf_streaming_loss_updater = tf_streaming_loss

    y_true = []
    y_pred = []

    # Reset streaming metrics
    sess.run(tf.local_variables_initializer())
    for X_eval, y_eval in eval_iterator:
        eval_feed_dict = model.create_train_feed_dict(X_eval, y_eval)
        _, batch_pred = sess.run([tf_streaming_loss_updater, model.tf_predict], feed_dict=eval_feed_dict)
        y_pred.extend(batch_pred)
        y_true.extend(y_eval)

    eval_loss = sess.run(tf_streaming_loss_value)

    eval_score = None if scoring_func is None else scoring_func(y_true=y_true, y_pred=y_pred)

    logging.info('Step: %s - Eval loss %s', global_step, eval_loss)
    logging.info('Step: %s - Eval score: %s', global_step, eval_score)
    return eval_loss, eval_score


def run_train(model, dataset_path, batch_size, num_epochs, eval_interval, path_to_save_stuff,
              train_phrase_func=None, eval_phrase_func=None, transform_pred_=None, scoring_func=None, gpu_fraction=0.3):
    """

    :param model:
    :param dataset_path:
    :param batch_size:
    :param num_epochs:
    :param eval_interval:
    :param path_to_save_stuff: directory to save some stuff during training
    :param transform_pred_: transform from index-based to text-based prediction
    :param scoring_func: The larger score is, the better model is. This function receive 2 params: y_true, y_pred
    :param gpu_fraction:
    :return:
    """
    if train_phrase_func is None:
        train_phrase_func = __run_train
    if eval_phrase_func is None:
        eval_phrase_func = __run_eval

    experiment_name = datetime.strftime(datetime.now(), '%Y-%m-%dT%H:%M:%S')
    __save_meta_data(model, path_to_save_stuff, experiment_name)

    my_dataset = Dataset.create_from_npz(dataset_path)
    my_dataset.show_info()
    train_iterator = my_dataset.data_train.get_data_iterator(batch_size=batch_size, num_epochs=num_epochs,
                                                             is_strictly_equal=True)
    with model.graph.as_default():
        tuple_streaming_loss = tf.metrics.mean(values=model.tf_loss)

        saver = tf.train.Saver(max_to_keep=5)

        logging.info('Model contains %s parameters', graph_utils.count_trainable_variables())
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        with tf.Session(graph=model.graph,
                        config=tf.ConfigProto(allow_soft_placement=False,
                                              gpu_options=gpu_options)).as_default() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            best_score = -100

            for X_train, y_train in train_iterator:
                global_step = train_phrase_func(model, sess, X_train, y_train)
                if global_step % eval_interval == 0 or global_step == 1:
                    eval_iterator = my_dataset.data_eval.get_data_iterator(batch_size=batch_size, num_epochs=1,
                                                                           is_strictly_equal=False)
                    eval_loss, eval_score = eval_phrase_func(global_step, model, sess, eval_iterator, tuple_streaming_loss,
                                                       scoring_func)

                    if eval_score > best_score:
                        best_score = eval_score

                        path_to_model = os.path.join(path_to_save_stuff, 'output', 'saved_models',
                                                     model.__class__.__name__,
                                                     experiment_name)
                        saver.save(sess, save_path=path_to_model,
                                   global_step=global_step,
                                   write_meta_graph=True)
                        logging.info('Gained better score on eval, saved model to %s', path_to_model)

                    logging.info('Best score on eval: %s', best_score)
                    logging.info('\n')
