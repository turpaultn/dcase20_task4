# Lint as: python3
"""Inference for trained DCASE 2020 task 4 separation model."""

import numpy as np
import tensorflow as tf


class SeparationModel(object):
  """Tensorflow audio separation model."""

  def __init__(self, checkpoint_path, metagraph_path):
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)
    with self.graph.as_default():
      new_saver = tf.train.import_meta_graph(metagraph_path)
      new_saver.restore(self.sess, checkpoint_path)
    self.input_placeholder = self.graph.get_tensor_by_name(
        'input_audio/receiver_audio:0')
    self.output_tensor = self.graph.get_tensor_by_name('denoised_waveforms:0')

  def separate(self, mixture_waveform):
    """Separates an mixture waveform into sources.

    Args:
      mixture_waveform: numpy.ndarray of shape (num_samples,).

    Returns:
      numpy.ndarray of (num_sources, num_samples).
    """
    mixture_waveform_input = np.reshape(mixture_waveform, (1, 1, -1))
    separated_waveforms = self.sess.run(
        self.output_tensor,
        feed_dict={self.input_placeholder: mixture_waveform_input})[0]
    return separated_waveforms
