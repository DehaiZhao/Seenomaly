# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the flowers dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/research/slim/datasets/download_and_convert_flowers.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

slim = tf.contrib.slim

_FILE_PATTERN = 'gif_%s_*.tfrecord'

SPLITS_TO_SIZES = {'train': 88239, 'validation': 1000}

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.'
}

_NUM_CLASSES = 50


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded_0': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/encoded_1': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/encoded_2': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/encoded_3': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/encoded_4': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/encoded_5': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/encoded_6': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/encoded_7': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
      'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
      'image/box': tf.FixedLenFeature([4], tf.int64, default_value=tf.zeros([4], dtype=tf.int64)),
      'image/height': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
      'image/width': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
  }

  items_to_handlers = {
      'image_0': slim.tfexample_decoder.Image('image/encoded_0'),
      'image_1': slim.tfexample_decoder.Image('image/encoded_1'),
      'image_2': slim.tfexample_decoder.Image('image/encoded_2'),
      'image_3': slim.tfexample_decoder.Image('image/encoded_3'),
      'image_4': slim.tfexample_decoder.Image('image/encoded_4'),
      'image_5': slim.tfexample_decoder.Image('image/encoded_5'),
      'image_6': slim.tfexample_decoder.Image('image/encoded_6'),
      'image_7': slim.tfexample_decoder.Image('image/encoded_7'),
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
      'box': slim.tfexample_decoder.Tensor('image/box'),
      'height': slim.tfexample_decoder.Tensor('image/height'),
      'width': slim.tfexample_decoder.Tensor('image/width'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,)
