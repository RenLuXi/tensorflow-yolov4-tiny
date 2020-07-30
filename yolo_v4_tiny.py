# -*- coding: utf-8 -*-
from builtins import *

import numpy as np
import tensorflow as tf
from yolo_v3 import _conv2d_fixed_padding, _fixed_padding, _get_size, \
    _detection_layer, _upsample

slim = tf.contrib.slim

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1

_ANCHORS = [(10, 14),  (23, 27),  (37, 58),
            (81, 82),  (135, 169),  (344, 319)]

def route_group(input_layer, groups, group_id):
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]
def yolo_v4_tiny(inputs, num_classes, is_training=False, data_format='NCHW', reuse=False):
    """
    Creates YOLO v4 tiny model.

    :param inputs: a 4-D tensor of size [batch_size, height, width, channels].
        Dimension batch_size may be undefined. The channel order is RGB.
    :param num_classes: number of predicted classes.
    :param is_training: whether is training or not.
    :param data_format: data format NCHW or NHWC.
    :param reuse: whether or not the network and its variables should be reused.
    :return:
    """
    # it will be needed later on
    img_size = inputs.get_shape().as_list()[1:3]

    # transpose the inputs to NCHW
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    # normalize values to range [0..1]
    inputs = inputs / 255

    # set batch norm params
    batch_norm_params = {
        'decay': _BATCH_NORM_DECAY,
        'epsilon': _BATCH_NORM_EPSILON,
        'scale': True,
        'is_training': is_training,
        'fused': None,  # Use fused batch norm if possible.
    }

    # Set activation_fn and parameters for conv2d, batch_norm.
    with slim.arg_scope([slim.conv2d, slim.batch_norm, _fixed_padding, slim.max_pool2d], data_format=data_format):
        with slim.arg_scope([slim.conv2d, slim.batch_norm, _fixed_padding], reuse=reuse):
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=_LEAKY_RELU)):

                with tf.variable_scope('yolo-v4-tiny'):
                    for i in range(2):
                        inputs = _conv2d_fixed_padding(
                            inputs, 16 * pow(2, i+1), 3, 2)
                    inputs = _conv2d_fixed_padding(
                        inputs, 64, 3)
                    inputs_2 = inputs
                    inputs = route_group(inputs, 2, 1)
                    inputs = _conv2d_fixed_padding(
                        inputs, 32, 3)
                    inputs_4 = inputs
                    inputs = _conv2d_fixed_padding(inputs, 32, 3)
                    inputs = tf.concat([inputs, inputs_4],
                                       axis=1 if data_format == 'NCHW' else 3)
                    inputs = _conv2d_fixed_padding(inputs, 64, 1)
                    inputs = tf.concat([inputs, inputs_2],
                                       axis=1 if data_format == 'NCHW' else 3)
                    inputs = slim.max_pool2d(
                        inputs, [2, 2], scope='pool2')


                    inputs = _conv2d_fixed_padding(
                        inputs, 128, 3)
                    inputs_10 = inputs
                    inputs = route_group(inputs, 2, 1)
                    inputs = _conv2d_fixed_padding(
                        inputs, 64, 3)
                    inputs_12 = inputs
                    inputs = _conv2d_fixed_padding(inputs, 64, 3)
                    inputs = tf.concat([inputs, inputs_12],
                                       axis=1 if data_format == 'NCHW' else 3)
                    inputs = _conv2d_fixed_padding(inputs, 128, 1)
                    inputs = tf.concat([inputs, inputs_10],
                                       axis=1 if data_format == 'NCHW' else 3)
                    inputs = slim.max_pool2d(
                        inputs, [2, 2], scope='pool2')


                    inputs = _conv2d_fixed_padding(
                        inputs, 256, 3)
                    inputs_18 = inputs
                    inputs = route_group(inputs, 2, 1)
                    inputs = _conv2d_fixed_padding(
                        inputs, 128, 3)
                    inputs_20 = inputs
                    inputs = _conv2d_fixed_padding(inputs, 128, 3)
                    inputs = tf.concat([inputs, inputs_20],
                                       axis=1 if data_format == 'NCHW' else 3)
                    inputs = _conv2d_fixed_padding(inputs, 256, 1)
                    inputs_23 = inputs
                    inputs = tf.concat([inputs, inputs_18],
                                       axis=1 if data_format == 'NCHW' else 3)
                    inputs = slim.max_pool2d(
                        inputs, [2, 2], scope='pool2')

                    inputs = _conv2d_fixed_padding(inputs, 512, 3)
                    inputs = _conv2d_fixed_padding(inputs, 256, 1)
                    inputs_27 = inputs
                    inputs = _conv2d_fixed_padding(inputs, 512, 3)
                    detect_1 = _detection_layer(
                        inputs, num_classes, _ANCHORS[3:6], img_size, data_format)
                    detect_1 = tf.identity(detect_1, name='detect_1')
                    inputs = inputs_27
                    inputs = _conv2d_fixed_padding(inputs, 128, 1)
                    upsample_size = inputs.get_shape().as_list()
                    upsample_size[1] = upsample_size[1]*2
                    upsample_size[2] = upsample_size[2] * 2
                    inputs = _upsample(inputs, upsample_size, data_format)
                    inputs = tf.concat([inputs, inputs_23],
                                       axis=1 if data_format == 'NCHW' else 3)
                    inputs = _conv2d_fixed_padding(inputs, 256, 3)

                    detect_2 = _detection_layer(
                        inputs, num_classes, _ANCHORS[1:4], img_size, data_format)
                    detect_2 = tf.identity(detect_2, name='detect_2')

                    detections = tf.concat([detect_1, detect_2], axis=1)
                    detections = tf.identity(detections, name='detections')
                    return detections
