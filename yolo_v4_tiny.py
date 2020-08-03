# -*- coding: utf-8 -*-
from builtins import *

import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1

_ANCHORS = [(10, 14),  (23, 27),  (37, 58),
            (81, 82),  (135, 169),  (344, 319)]


def route_group(input_layer, in_channels, data_format):
    # convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    # return convs[group_id]
    split = input_layer[:, in_channels//2:, :, :] if data_format == "NCHW" else input_layer[:, :, :, in_channels // 2:]
    return split


def _get_size(shape, data_format):
    if len(shape) == 4:
        shape = shape[1:]
    return shape[1:3] if data_format == 'NCHW' else shape[0:2]


def _conv2d_fixed_padding(inputs, filters, kernel_size, strides=1):
    if strides > 1:
        inputs = _fixed_padding(inputs, kernel_size)
    inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides,
                         padding=('SAME' if strides == 1 else 'VALID'))
    return inputs


def _detection_layer(inputs, num_classes, anchors, img_size, data_format):
    num_anchors = len(anchors)
    predictions = slim.conv2d(inputs, num_anchors * (5 + num_classes), 1,
                              stride=1, normalizer_fn=None,
                              activation_fn=None,
                              biases_initializer=tf.zeros_initializer())

    shape = predictions.get_shape().as_list()
    grid_size = _get_size(shape, data_format)
    dim = grid_size[0] * grid_size[1]
    bbox_attrs = 5 + num_classes

    if data_format == 'NCHW':
        predictions = tf.reshape(
            predictions, [-1, num_anchors * bbox_attrs, dim])
        predictions = tf.transpose(predictions, [0, 2, 1])

    predictions = tf.reshape(predictions, [-1, num_anchors * dim, bbox_attrs])

    stride = (img_size[0] // grid_size[0], img_size[1] // grid_size[1])

    anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in anchors]

    box_centers, box_sizes, confidence, classes = tf.split(
        predictions, [2, 2, 1, num_classes], axis=-1)

    box_centers = tf.nn.sigmoid(box_centers)
    confidence = tf.nn.sigmoid(confidence)

    grid_x = tf.range(grid_size[0], dtype=tf.float32)
    grid_y = tf.range(grid_size[1], dtype=tf.float32)
    a, b = tf.meshgrid(grid_x, grid_y)

    x_offset = tf.reshape(a, (-1, 1))
    y_offset = tf.reshape(b, (-1, 1))

    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, num_anchors]), [1, -1, 2])

    box_centers = box_centers + x_y_offset
    box_centers = box_centers * stride

    anchors = tf.tile(anchors, [dim, 1])
    box_sizes = tf.exp(box_sizes) * anchors
    box_sizes = box_sizes * stride

    detections = tf.concat([box_centers, box_sizes, confidence], axis=-1)

    classes = tf.nn.sigmoid(classes)
    predictions = tf.concat([detections, classes], axis=-1)
    return predictions


def _upsample(inputs, out_shape, data_format='NCHW'):
    # tf.image.resize_nearest_neighbor accepts input in format NHWC
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])

    if data_format == 'NCHW':
        new_height = out_shape[3]
        new_width = out_shape[2]
    else:
        new_height = out_shape[2]
        new_width = out_shape[1]

    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))

    # back to NCHW if needed
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    inputs = tf.identity(inputs, name='upsampled')
    return inputs


@tf.contrib.framework.add_arg_scope
def _fixed_padding(inputs, kernel_size, *args, mode='CONSTANT', **kwargs):
    """
    Pads the input along the spatial dimensions independently of input size.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      data_format: The input format ('NHWC' or 'NCHW').
      mode: The mode for tf.pad.

    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if kwargs['data_format'] == 'NCHW':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end],
                                        [pad_beg, pad_end]],
                               mode=mode)
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]], mode=mode)
    return padded_inputs


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
                        #cspdarknet
                    inputs = _conv2d_fixed_padding(
                        inputs, 64, 3)
                    inputs_2 = inputs
                    inputs = route_group(inputs, 64, data_format)
                    inputs = _conv2d_fixed_padding(
                        inputs, 32, 3)
                    inputs_4 = inputs
                    inputs = _conv2d_fixed_padding(inputs, 32, 3)
                    inputs = tf.concat([inputs, inputs_4],
                                       axis=1 if data_format == 'NCHW' else 3)
                    inputs = _conv2d_fixed_padding(inputs, 64, 1)
                    inputs = tf.concat([inputs_2, inputs],
                                       axis=1 if data_format == 'NCHW' else 3)
                    inputs = slim.max_pool2d(
                        inputs, [2, 2], scope='pool2')


                    inputs = _conv2d_fixed_padding(
                        inputs, 128, 3)
                    inputs_10 = inputs
                    inputs = route_group(inputs, 128, data_format)
                    inputs = _conv2d_fixed_padding(
                        inputs, 64, 3)
                    inputs_12 = inputs
                    inputs = _conv2d_fixed_padding(inputs, 64, 3)
                    inputs = tf.concat([inputs, inputs_12],
                                       axis=1 if data_format == 'NCHW' else 3)
                    inputs = _conv2d_fixed_padding(inputs, 128, 1)
                    inputs = tf.concat([inputs_10, inputs],
                                       axis=1 if data_format == 'NCHW' else 3)
                    inputs = slim.max_pool2d(
                        inputs, [2, 2], scope='pool2')


                    inputs = _conv2d_fixed_padding(
                        inputs, 256, 3)
                    inputs_18 = inputs
                    inputs = route_group(inputs, 256, data_format)
                    inputs = _conv2d_fixed_padding(
                        inputs, 128, 3)
                    inputs_20 = inputs
                    inputs = _conv2d_fixed_padding(inputs, 128, 3)
                    inputs = tf.concat([inputs, inputs_20],
                                       axis=1 if data_format == 'NCHW' else 3)
                    inputs = _conv2d_fixed_padding(inputs, 256, 1)
                    inputs_23 = inputs
                    inputs = tf.concat([inputs_18, inputs],
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
                    upsample_size[1] = upsample_size[1] * 2
                    upsample_size[2] = upsample_size[2] * 2
                    inputs = _upsample(inputs, upsample_size, data_format)
                    inputs = tf.concat([inputs_23, inputs],
                                       axis=1 if data_format == 'NCHW' else 3)
                    inputs = _conv2d_fixed_padding(inputs, 256, 3)

                    detect_2 = _detection_layer(
                        inputs, num_classes, _ANCHORS[1:4], img_size, data_format)
                    detect_2 = tf.identity(detect_2, name='detect_2')

                    detections = tf.concat([detect_1, detect_2], axis=1)
                    detections = tf.identity(detections, name='detections')
                    return detections
