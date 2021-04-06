# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
from nets import nets_factory
import tf_slim as slim
# import tf_slim as slim
import os
import numpy as np
# from scipy.misc import imsave
from scipy import misc
from sklearn.decomposition import PCA
from scipy.spatial import distance
from tqdm import tqdm
import pickle
import json
import random
import math

from constants import ROOT_PATH

'''
gan:gan/generator/encoder/fc6  ROOT_PATH/Seenomaly/models/gan/model.ckpt-29471
vae:vae/encoder/fc6  ROOT_PATH/Seenomaly/models/vae/model.ckpt-54717
vaegan:vaegan/generator/encoder/fc6  ROOT_PATH/Seenomaly/models/vaegan/model.ckpt-121858
aernn:aernnfc  ROOT_PATH/Seenomaly/models/aernn/model.ckpt-52198

ROOT_PATH/Seenomaly/Rico_Data/synthetic_data/label.txt
ROOT_PATH/Seenomaly/Rico_Data/test_data/images/label.txt

'''

net_name = 'vae'
logits_name = 'vae/encoder/fc6'
# ck_path = f'{ROOT_PATH}/Seenomaly/models/vae/model.ckpt-54717'
ck_path = f'{ROOT_PATH}/Seenomaly/models/vae/model.ckpt-29471'


batch_size = 1
image_size = 224
_STRIDE = 8
num_classes = 50 
max_num_images = 225
dataset_dir = f'{ROOT_PATH}/Seenomaly/Rico_Data'
abnormal_label = f'{ROOT_PATH}/Seenomaly/Rico_Data/test_data/images/label.txt'
normal_label = f'{ROOT_PATH}/Seenomaly/Rico_Data/label.txt'

save_dir = os.path.join(dataset_dir, 'features', 'real', net_name)


class FeatureExtractor(object):

  def __init__(self, network_name, checkpoint_path, batch_size, image_size=None):
    self._network_name = network_name
    self._checkpoint_path = checkpoint_path
    self._batch_size = batch_size
    self._image_size = image_size
    self._layer = {}

    self._global_step = tf.train.get_or_create_global_step()

    # Retrieve the function that returns logits and endpoints
    self._network_fn = nets_factory.get_network_fn(self._network_name, num_classes = num_classes, is_training=False)

    # Retrieve the model scope from network factory
    self._model_scope = nets_factory.arg_scopes_map[self._network_name]

    # Fetch the default image size
    self._image_size = self._network_fn.default_image_size
    self._filename_queue = tf.FIFOQueue(100000, [tf.string], shapes=[[]], name="filename_queue")
    self._pl_image_files = tf.placeholder(tf.string, shape=[None], name="image_file_list")
    self._enqueue_op = self._filename_queue.enqueue_many([self._pl_image_files])
    self._num_in_queue = self._filename_queue.size()

    self._batch_from_queue, self._batch_filenames = self._preproc_image_batch(self._batch_size, num_threads=4)

    #self._image_batch = tf.placeholder_with_default(
    #        self._batch_from_queue, shape=[self._batch_size, _STRIDE, self._image_size, self._image_size, 3])
    self._image_batch = tf.placeholder(tf.float32, [batch_size, _STRIDE, image_size, image_size, 3])

    # Retrieve the logits and network endpoints (for extracting activations)
    # Note: endpoints is a dictionary with endpoints[name] = tf.Tensor
    self._logits, self._endpoints = self._network_fn(self._image_batch)

    # Find the checkpoint file
    checkpoint_path = self._checkpoint_path
    if tf.gfile.IsDirectory(self._checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(self._checkpoint_path)

    # Load pre-trained weights into the model
    variables_to_restore = slim.get_variables_to_restore()
    restore_fn = slim.assign_from_checkpoint_fn(self._checkpoint_path, variables_to_restore)

    # Start the session and load the pre-trained weights
    self._sess = tf.Session()
    restore_fn(self._sess)

    # Local variables initializer, needed for queues etc.
    self._sess.run(tf.local_variables_initializer())

    # Managing the queues and threads
    self._coord = tf.train.Coordinator()
    self._threads = tf.train.start_queue_runners(coord=self._coord, sess=self._sess)

  def print_network_summary(self):
    for name, tensor in self._endpoints.items():
      print("{} has shape {}".format(name, tensor.shape))
      #print(self._inputs.eval(session = self._sess))

  def forward(self, layer_names, image=None, fetch_images=False):
    fetches = {}
    available_layers = self.layer_names()
    for layer_name in layer_names:
      if layer_name not in available_layers:
        raise ValueError("Unable to extract features for layer: {}".format(layer_name))
      fetches[layer_name] = self._endpoints[layer_name]
    feed_dict = None
    if image is not None:
      feed_dict = {self._image_batch: image}
    if fetch_images:
      fetches["image"] = self._image_batch

    # Fetch how many examples left in queue
    fetches["examples_in_queue"] = self._num_in_queue
    return self._sess.run(fetches, feed_dict = feed_dict)

  def enqueue_image_files(self, image_files):
    self._sess.run(self._enqueue_op, feed_dict={self._pl_image_files: image_files})

  def _preproc_image_batch(self, batch_size, num_threads=1):

    # Read image file from disk and decode JPEG
    reader = tf.WholeFileReader()
    image_filename, image_raw = reader.read(self._filename_queue)
    image_data = tf.image.decode_jpeg(image_raw, channels=3)
    # Image preprocessing
    image_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)
    image_data = tf.expand_dims(image_data, 0)
    image_data = tf.image.resize_bilinear(image_data, [image_size, image_size], align_corners=False)
    image_data = tf.squeeze(image_data)
    image_data = image_data/255.0
        
    # Read a batch of preprocessing images from queue
    image_batch = tf.train.batch(
        [image_data, image_filename], batch_size, num_threads=num_threads, allow_smaller_final_batch=True)
    return image_batch

  def layer_names(self):
    return self._endpoints.keys()

  @property
  def image_size(self):
    return self._image_size

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def num_preproc_threads(self):
    return self._num_preproc_threads

def _get_filenames_and_classes(label_dir):
  with open(label_dir) as list_file:
    file_list = list_file.readlines()
  return file_list

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
    image = misc.imread(os.path.join(file_name, sample))
    image = misc.imresize(image, (image_size, image_size))
    image = image / 255.0
    image_list.append(image)
  return image_list

def main(_):
  feature_extractor = FeatureExtractor(
    network_name=net_name,
    checkpoint_path=ck_path,
    batch_size=batch_size)
  feature_extractor.print_network_summary()

  batch_image = np.zeros([batch_size, _STRIDE, image_size, image_size, 3], dtype=np.float32)

  normal_list = _get_filenames_and_classes(normal_label)
  abnormal_list = _get_filenames_and_classes(abnormal_label)

  if max_num_images < len(normal_list):
    normal_list = [normal_list[i] for i in sorted(random.sample(range(len(normal_list)), max_num_images))]

  if max_num_images < len(abnormal_list):
    abnormal_list = [abnormal_list[i] for i in sorted(random.sample(range(len(abnormal_list)), max_num_images))]

  gif_files = []
  box_list = []
  labels = []
  for filename in abnormal_list:
    gif_files.append(filename.strip().split()[0])
    labels.append(filename.strip().split()[1])
  for filename in normal_list:
    gif_files.append(filename.strip().split()[0])
    labels.append(0)

  print("keeping %d images to analyze" % (len(normal_list) + len(abnormal_list)))

  features = []
  for i in tqdm(range(len(gif_files))):
    image_data = _concat_image(gif_files[i]) 
    batch_image[0] = image_data

    feat = feature_extractor.forward([logits_name], batch_image, fetch_images=True)
    feat = feat[logits_name]
    feat = np.squeeze(feat)
    features.append(feat)

  features = np.array(features)
  pca = PCA(n_components=200)
  pca.fit(features)
  pca_features = pca.transform(features)

  if not os.path.exists(save_dir):
    os.makedirs(save_dir) 

  pickle.dump([gif_files, pca_features, labels], open(os.path.join(save_dir, 'features.p'), 'wb'))

if __name__ == '__main__':
  tf.app.run()



