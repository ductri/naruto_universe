import tensorflow as tf
import logging
import pickle
import numpy as np


class Predictor:
    def __init__(self, path_to_params, path_to_model, input_transform_func, preprocess_func, output_transform_func=None):
        with open(path_to_params, 'rb') as input_file:
            self.params_dict = pickle.load(input_file)

        self.input_transform_func = input_transform_func
        self.output_transform_func = output_transform_func if output_transform_func is not None else lambda x: x
        self.preprocess_func = preprocess_func

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                saver = tf.train.import_meta_graph('{}.meta'.format(path_to_model))
                saver.restore(self.sess, path_to_model)
        logging.info('Restored saved model at %s', path_to_model)

    def predict(self, input_values):
        docs_index = self.__cvt_input_format(input_values)
        return self._predict(docs_index)

    def predict_prob(self, input_values):
        docs_index = self.__cvt_input_format(input_values)
        return self._predict_prob(docs_index)

    def _predict(self, input_values):
        tf_predict = self.graph.get_tensor_by_name(self.params_dict['tf_predict'])
        feed_dict = self.__build_feed_dict(input_values)
        return self.output_transform_func(np.squeeze(self.sess.run(tf_predict, feed_dict=feed_dict)))

    def _predict_prob(self, input_values):
        tf_predict_prob = self.graph.get_tensor_by_name(self.params_dict['tf_predict_prob'])
        feed_dict = self.__build_feed_dict(input_values)
        return np.squeeze(self.sess.run(tf_predict_prob, feed_dict=feed_dict))

    def __cvt_input_format(self, input_values):
        input_values = self.preprocess_func(input_values)
        docs_index = self.input_transform_func(input_values)
        return docs_index

    def __build_feed_dict(self, input_values):
        tf_inputs = [self.graph.get_tensor_by_name(input_name) for input_name in self.params_dict['tf_inputs']]
        feed_dict_func = self.params_dict['feed_dict_for_infer_func']
        feed_dict = feed_dict_func(tf_inputs, input_values)
        return feed_dict
