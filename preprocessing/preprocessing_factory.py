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
"""Contains a factory for building various models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from preprocessing import vae_preprocessing
from preprocessing import fbn_preprocessing


slim = tf.contrib.slim


def get_preprocessing(name, is_training=False):

  preprocessing_fn_map = {
      'vae': vae_preprocessing,
      'gan': vae_preprocessing,
      'c3d': vae_preprocessing,
      'fbn': fbn_preprocessing,
      'vaegan': vae_preprocessing,
      'aernn': vae_preprocessing,

  }

  if name not in preprocessing_fn_map:
    raise ValueError('Preprocessing name [%s] was not recognized' % name)

  def preprocessing_fn(image, box, height, width, output_height, output_width, **kwargs):
    return preprocessing_fn_map[name].preprocess_image(
        image, box, height, width, output_height, output_width, is_training=is_training, **kwargs)

  return preprocessing_fn
