from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def vae_arg_scope(is_training = True, weight_decay=0.0005):
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
    with slim.arg_scope([slim.conv3d], kernel_size = 3, padding = 'same', outputs_collections=end_points_collection):
      print (inputs.shape)
      net = slim.repeat(inputs, 1, slim.conv3d, 32,  scope='conv1')
      net = slim.max_pool3d(net, [1, 2, 2],[1, 2, 2], scope='pool1')
      print (net.shape)
      net = slim.repeat(net, 1, slim.conv3d, 64, scope='conv2')
      net = slim.max_pool3d(net, [1, 2, 2],[1, 2, 2], scope='pool2')
      print (net.shape)
      net = slim.repeat(net, 1, slim.conv3d, 128, scope='conv3')
      net = slim.max_pool3d(net, [2, 2, 2],[2, 2, 2], scope='pool3')
      print (net.shape)
      net = slim.repeat(net, 1, slim.conv3d, 256, scope='conv4')
      net = slim.max_pool3d(net, [2, 2, 2],[2, 2, 2], scope='pool4')
      print (net.shape)
      net = slim.repeat(net, 1, slim.conv3d, 512, scope='conv5')
      net = slim.max_pool3d(net, [2, 2, 2],[2, 2, 2], scope='pool5')
      print (net.shape)
      net = slim.conv3d(net, 4096, kernel_size = [1, 7, 7], normalizer_fn = None, padding = 'valid', scope='fc6')
      #net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')
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
      net = slim.repeat(net, 1, slim.conv3d_transpose, 32, stride = [2, 2, 2],scope='tran_conv5')  
      print (net.shape)
      if is_training:
        net = slim.repeat(net, 1, slim.conv3d_transpose, 3, stride = [2, 2, 2], activation_fn = None, normalizer_fn = None, scope='tran_conv6')
      else:
        net = slim.repeat(net, 1, slim.conv3d_transpose, 3, stride = [1, 2, 2], activation_fn = tf.nn.sigmoid, normalizer_fn = None, scope='tran_conv6')
      print (net.shape) 
      end_points = slim.utils.convert_collection_to_dict(end_points_collection) 
    return net, end_points 

def vae(inputs, label, num_classes=50, is_training=True):
  with tf.variable_scope('vae') as sc:
    end_points = {}
    if is_training:
      net, end_points = encoder(inputs, is_training = is_training)
      mu = tf.split(value = net, axis = 1, num_or_size_splits = 2)[0]
      std = tf.split(value = net, axis = 1, num_or_size_splits = 2)[1]
      eps = tf.random_normal(std.shape)
      z = eps * std + mu

      net_decoder, end_points_decoder = decoder(z, is_training = is_training)
      end_points.update(end_points_decoder)
      # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
      kl_loss = 0.5 * tf.reduce_sum(tf.square(mu) + std - tf.log(std + 1e-8) - 1)
      rec_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = inputs, logits = net_decoder,  reduction=tf.losses.Reduction.SUM)

      slim.losses.add_loss(kl_loss)     
      return net_decoder, end_points
    else:
      net, end_points = encoder(inputs, is_training = is_training)
      net = tf.split(value = net, axis = 1, num_or_size_splits = 2)[0]
      return net, end_points

vae.default_image_size = 224


