from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import tf_slim as slim


def fbn_arg_scope(is_training = True, weight_decay=0.0005):
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
      net = slim.repeat(inputs, 1, slim.conv3d, 32,  scope='conv3d_1')
      net = slim.max_pool3d(net, [1, 2, 2],[1, 2, 2], scope='pool3d_1')
      print (net.shape)
      net = slim.repeat(net, 1, slim.conv3d, 64, scope='conv3d_2')
      net = slim.max_pool3d(net, [1, 2, 2],[1, 2, 2], scope='pool3d_2')
      print (net.shape)
      net = slim.repeat(net, 1, slim.conv3d, 128, scope='conv3d_3')
      net = slim.max_pool3d(net, [2, 2, 2],[2, 2, 2], scope='pool3d_3')
      print (net.shape)
      net = slim.repeat(net, 1, slim.conv3d, 256, scope='conv3d_4')
      net = slim.max_pool3d(net, [2, 2, 2],[2, 2, 2], scope='pool3d_4')
      print (net.shape)
      net = slim.repeat(net, 1, slim.conv3d, 512, scope='conv3d_5')
      net = slim.max_pool3d(net, [2, 2, 2],[2, 2, 2], scope='pool3d_5')
      print (net.shape)
           
    return net

def c2d_base(inputs, is_training=True):
  with tf.variable_scope('c2d_base') as sc:
    with slim.arg_scope([slim.conv2d], kernel_size = 3):
      print (inputs.shape, '2d input')
      net = slim.repeat(inputs, 1, slim.conv2d, 32, scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      print (net.shape)
      net = slim.repeat(net, 1, slim.conv2d, 64, scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      print (net.shape)
      net = slim.repeat(net, 1, slim.conv2d, 128, scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      print (net.shape)
      net = slim.repeat(net, 1, slim.conv2d, 256, scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      print (net.shape)
      net = slim.repeat(net, 1, slim.conv2d, 512, scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      print (net.shape)
    return net


def fbn(inputs, label = None, num_classes=50, dropout_keep_prob=0.9, is_training=True):
  with tf.variable_scope('fbn') as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d], outputs_collections=end_points_collection):
      fg = tf.slice(inputs, [0, 1, 0, 0, 0], [-1, 7, 224, 224, -1])
      padding = [[0, 0], [1, 0], [0, 0], [0, 0], [0, 0]]
      fg = tf.pad(fg, padding)
      bg = tf.slice(inputs, [0, 0, 0, 0, 0], [-1, 1, 224, 224, -1])
      print (bg.shape, 'bg')
      fg = c3d_base(fg, is_training = is_training)
      bg = c2d_base(tf.squeeze(bg, 1), is_training = is_training)

      fg = tf.squeeze(fg, 1)
      print (fg.shape)
      fg = slim.conv2d(fg, 4096, [7, 7], padding = 'valid', scope='fc_fg_6')
      fg = slim.dropout(fg, dropout_keep_prob, is_training=is_training, scope='dropout_fg_6')
      print (fg.shape)
      fg = slim.conv2d(fg, 2048, [1, 1], padding = 'valid', activation_fn=tf.nn.tanh, scope='fc_fg_7')  
      print (fg.shape, 'fg') 
      mask = slim.conv2d(fg, 2048, [1, 1], padding = 'valid', activation_fn=tf.nn.sigmoid, scope='fc_mask_7')  

      bg = slim.conv2d(bg, 4096, [7, 7], padding = 'valid', scope='fc_bg_6')
      bg = slim.dropout(bg, dropout_keep_prob, is_training=is_training, scope='dropout_bg_6')
      print (bg.shape)
      bg = slim.conv2d(bg, 2048, [1, 1], padding = 'valid', activation_fn=tf.nn.tanh, scope='fc_bg_7')  

      net = mask * fg + (1 - mask) * bg  
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)

      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        print (net.shape, 'fc8')
        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      if is_training:
        loss = slim.losses.softmax_cross_entropy(net, label)
        return net, end_points
      else:
        return net, end_points

fbn.default_image_size = 224


