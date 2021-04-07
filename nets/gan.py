from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tf_slim as slim


def gan_arg_scope(is_training = True, weight_decay=0.0005):
  """Defines the VAE arg scope.

  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv3d_transpose, slim.conv3d],
                      activation_fn=tf.nn.relu,
                      weights_regularizer = slim.l2_regularizer(weight_decay),
                      #biases_initializer=tf.zeros_initializer(),
                      normalizer_fn=slim.batch_norm,
                      normalizer_params={'scale': True,}
                      )as arg_sc:
    return arg_sc

def encoder(inputs, is_training=True, dropout_keep_prob=0.5):
  with tf.variable_scope('encoder') as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv3d], kernel_size = 3, padding = 'same', activation_fn = tf.nn.leaky_relu, outputs_collections=end_points_collection):
      print (inputs.shape, 'inputs')
      net = slim.repeat(inputs, 1, slim.conv3d, 32, scope='conv1')
      net = slim.max_pool3d(net, [1, 2, 2],[1, 2, 2], scope='pool1')
      print (net.shape, 'pool1')
      net = slim.repeat(net, 1, slim.conv3d, 64, scope='conv2')
      net = slim.max_pool3d(net, [1, 2, 2],[1, 2, 2], scope='pool2')
      print (net.shape, 'pool2')
      net = slim.repeat(net, 1, slim.conv3d, 128, scope='conv3')
      net = slim.max_pool3d(net, [2, 2, 2],[2, 2, 2], scope='pool3')
      print (net.shape, 'pool3')
      net = slim.repeat(net, 1, slim.conv3d, 256, scope='conv4')
      net = slim.max_pool3d(net, [2, 2, 2],[2, 2, 2], scope='pool4')
      print (net.shape, 'pool4')
      net = slim.repeat(net, 1, slim.conv3d, 512, scope='conv5')
      net = slim.max_pool3d(net, [2, 2, 2],[2, 2, 2], scope='pool5')
      print (net.shape, 'pool5')
      net = slim.conv3d(net, 2048, kernel_size = [1, 7, 7], activation_fn = None, normalizer_fn = None, padding = 'valid', scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')
      print (net.shape)  
      net = tf.squeeze(net, [1, 2, 3])
      print (net.shape)       
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
    return net, end_points

def decoder(inputs, is_training=True):
  with tf.variable_scope('decoder') as sc:
    end_points_collection = sc.original_name_scope + '_end_points'

    with slim.arg_scope([slim.conv3d_transpose], kernel_size = 3, padding='same', outputs_collections=end_points_collection):
      net = inputs
      print (net.shape, 'input')
      net = tf.expand_dims(tf.expand_dims(tf.expand_dims(net, 1), 1), 1)
      print (net.shape, 'expand')
      net = slim.repeat(net, 1, slim.conv3d_transpose, 512, kernel_size = [1, 7, 7], stride = 1, padding = 'valid', scope='tran_conv1')
      print (net.shape)
      net = slim.repeat(net, 1, slim.conv3d_transpose, 256, stride = [1, 2, 2], scope='tran_conv2')
      print (net.shape)
      net = slim.repeat(net, 1, slim.conv3d_transpose, 128, stride = [1, 2, 2], scope='tran_conv3')
      print (net.shape)
      net = slim.repeat(net, 1, slim.conv3d_transpose, 64, stride = [2, 2, 2], scope='tran_conv4')
      print (net.shape)
      net = slim.repeat(net, 1, slim.conv3d_transpose, 32, stride = [2, 2, 2], scope='tran_conv5')  
      print (net.shape)
      net = slim.repeat(net, 1, slim.conv3d_transpose, 3, stride = [2, 2, 2], activation_fn = tf.nn.tanh, normalizer_fn = None, scope='tran_conv6')
      print (net.shape) 
      end_points = slim.utils.convert_collection_to_dict(end_points_collection) 
    return net, end_points 

def generator(inputs, is_training = True):
  with tf.variable_scope('generator'):
    z, _ = encoder(inputs, is_training = is_training)
    x_, end_points_d = decoder(z, is_training = is_training)
    z_, end_points = encoder(x_, is_training = is_training)
    end_points.update(end_points_d)
  return z, x_, z_, end_points

def discriminator(inputs, is_training = True):
  with tf.variable_scope('discriminator') as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv3d], kernel_size = 4, padding = 'same', activation_fn = tf.nn.leaky_relu, outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 1, slim.conv3d, 32, stride = [1, 2, 2], scope='conv1')
      print (net.shape)
      net = slim.repeat(net, 1, slim.conv3d, 64, stride = [1, 2, 2], scope='conv2')
      print (net.shape)
      net = slim.repeat(net, 1, slim.conv3d, 128, stride = [2, 2, 2], scope='conv3')
      print (net.shape)
      net = slim.repeat(net, 1, slim.conv3d, 256, stride = [2, 2, 2], scope='conv4')
      print (net.shape)
      net = slim.repeat(net, 1, slim.conv3d, 512, stride = [2, 2, 2], scope='conv5')
      print (net.shape)
      net = slim.conv3d(net, 2048, kernel_size = [1, 7, 7], padding = 'valid', scope='fc6')
      print (net.shape)  
      net = tf.squeeze(net, 1)
      print (net.shape) 
      classifier = slim.conv2d(net, 1, [1, 1], activation_fn = None, scope='fc7')
      classifier = tf.squeeze(classifier)
      print (classifier.shape, 'classifier')  
      net = tf.squeeze(net, [1, 2])   
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      print (net.shape) 
    return net, classifier, end_points


def gan(inputs, label, num_classes=50, is_training=True):
  with tf.variable_scope('gan', reuse=tf.AUTO_REUSE) as sc:
    if is_training:
      z, x_, z_, end_points = generator(inputs, is_training = is_training)
      feature_fake, label_fake, _ = discriminator(x_, is_training = is_training)
      feature_real, label_real, end_points_d = discriminator(inputs, is_training = is_training)
      end_points.update(end_points_d)
      
      context_loss = tf.losses.absolute_difference(inputs, x_, weights = 10.0)
      encoder_loss = tf.losses.mean_squared_error(z, z_, weights = 10)
      real_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(label_real), label_real)
      fake_loss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(label_fake), label_fake)
      
      return z_, end_points
    else:
      z, x_, z_, end_points = generator(inputs, is_training = is_training)
      return z_, end_points

gan.default_image_size = 224


