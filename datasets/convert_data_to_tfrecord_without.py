from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import glob
import cv2
import scipy
import numpy as np

import tensorflow as tf
from constants import ROOT_PATH


_NUM_VALIDATION = 1000
_RANDOM_SEED = 0
_NUM_SHARDS = 20
_STRIDE = 8
dataset_name = 'rico_gif'

dataset_dir = f'{ROOT_PATH}/Seenomaly/Rico_Data'
output_dir = os.path.join(dataset_dir, 'tf_record', dataset_name)

def bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def _image_to_tfexample(image_data):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded_0': bytes_feature(image_data[0]),
      'image/encoded_1': bytes_feature(image_data[1]),
      'image/encoded_2': bytes_feature(image_data[2]),
      'image/encoded_3': bytes_feature(image_data[3]),
      'image/encoded_4': bytes_feature(image_data[4]),
      'image/encoded_5': bytes_feature(image_data[5]),
      'image/encoded_6': bytes_feature(image_data[6]),
      'image/encoded_7': bytes_feature(image_data[7]),
  }))

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_filenames_and_classes():
  with open(os.path.join(dataset_dir, 'file_list.txt')) as list_file:
    file_list = list_file.readlines()
  return file_list


def _get_dataset_filename(split_name, shard_id):
  output_filename = 'gif_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(output_dir, output_filename)

def _concat_image(file_name):
  file_list = os.listdir(file_name)
  image_list = []
  if len(file_list) in range(int(_STRIDE / 2) + 1, _STRIDE):
    sample_list = file_list + random.sample(file_list, _STRIDE - len(file_list))
    sample_list.sort()
  else:  
    file_list = file_list * math.ceil(_STRIDE/len(file_list))
    file_list.sort()
    sample_list = random.sample(file_list, _STRIDE)
    sample_list.sort()
  for sample in sample_list:
    image = tf.gfile.FastGFile(os.path.join(file_name, sample), 'rb').read()
    image_list.append(image)
  return image_list

def _convert_dataset(split_name, filenames):

  assert split_name in ['train', 'validation']

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  img_files = []

  for filename in filenames:
    img_files.append(filename.strip())

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filenames), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_data = _concat_image(img_files[i])
            example = _image_to_tfexample(image_data)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()

def main(_):

  file_list = _get_filenames_and_classes()

  random.seed(_RANDOM_SEED)
  random.shuffle(file_list)
  training_filenames = file_list[_NUM_VALIDATION:]
  validation_filenames = file_list[:_NUM_VALIDATION]

  # First, convert the training and validation sets.
  _convert_dataset('train', training_filenames)
  _convert_dataset('validation', validation_filenames)

  print('\nFinished converting the Flowers dataset!')

if __name__ == '__main__':
  tf.app.run()
