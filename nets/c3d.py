from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import tf_slim as slim


def c3d_arg_scope(is_training = True, weight_decay=0.0005):
  """Defines the VAE arg scope.

  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv3d, slim.conv2d],
                      activation_fn=tf.nn.relu,
                      weights_regularizer = slim.l2_regularizer(weight_decay),
                      #biases_initializer=tf.zeros_initializer(),
                      normalizer_fn=slim.batch_norm,
                      normalizer_params={'scale': True,}
                      )as arg_sc:
    return arg_sc

def c3d_base(inputs, is_training=True):
  with tf.variable_scope('c3d_base') as sc:
    with slim.arg_scope([slim.conv3d], kernel_size = 3, padding = 'same'):
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
           
    return net


def c3d(inputs, label = None, num_classes=50, dropout_keep_prob=0.9, is_training=True):
  with tf.variable_scope('c3d') as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d], outputs_collections=end_points_collection):
      net = c3d_base(inputs, is_training = is_training)
      net = tf.squeeze(net, 1)
      print (net.shape)
      net = slim.conv2d(net, 4096, [7, 7], padding = 'valid', scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')
      print (net.shape)
      net = slim.conv2d(net, 2048, [1, 1], padding = 'valid', scope='fc7')  
      print (net.shape)    
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        print (net.shape)
        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      if is_training:
        loss = slim.losses.softmax_cross_entropy(net, label)
        return net, end_points
      else:
        return net, end_points

c3d.default_image_size = 224


