# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os, scipy.io
from os.path import basename

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import rawpy
import glob
from PIL import Image


input_image_path = "/content/drive/My Drive/Colab Notebooks/LearningToSeeInDark/Learning-to-See-in-the-Dark-master/MyOwn/TestImages/20208_09_0.04s.ARW"
checkpoint_dir = '../checkpoint/Sony/'
result_dir = 'TestResults/'
def lrelu(x):
    return tf.maximum(x * 0.2, x)

def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output

def network(input):
    conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
    conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
    pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

    conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
    conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
    pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

    conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
    conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
    pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

    conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
    conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
    pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

    conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
    conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')

    up6 = upsample_and_concat(conv5, conv4, 256, 512)
    conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
    conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

    up7 = upsample_and_concat(conv6, conv3, 128, 256)
    conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
    conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

    up8 = upsample_and_concat(conv7, conv2, 64, 128)
    conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
    conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')

    up9 = upsample_and_concat(conv8, conv1, 32, 64)
    conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
    conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')

    conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    out = tf.depth_to_space(conv10, 2)
    return out

def pack_raw(raw):
    # pack Bayer image to 4 channels
    black_level =  np.amin(raw.black_level_per_channel) # assume they're all the same
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - black_level, 0) / (2**16-1 - black_level)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]
    #if bayer patern = [[G, B]; [R, G]]
    out = np.concatenate((im[1:H:2, 0:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[0:H:2, 0:W:2, :]), axis=2)
    # #if bayer patern = [[R, G]; [G, B]]
    # out = np.concatenate((im[0:H:2, 0:W:2, :],
    #                       im[0:H:2, 1:W:2, :],
    #                       im[1:H:2, 1:W:2, :],
    #                       im[1:H:2, 0:W:2, :]), axis=2)
    return out
def pack_png(png):
    # pack png image to 4 channels RGBG, currently PNG is RGB
    H = png.shape[0]
    W = png.shape[1]
    #extract three chanels
    R = np.expand_dims(png[:,:,0],axis=2)
    G = np.expand_dims(png[:,:,1],axis=2)
    B = np.expand_dims(png[:,:,2],axis=2)
    arr = np.concatenate((R,G,B,G),axis=2)
    arr = arr*62.23 + 512

    black_level =  512
    im = arr.astype(np.float32)
    im = np.maximum(im - black_level, 0) / (16383 - black_level)  # subtract the black level
    return im

sess = tf.Session()
in_image = tf.placeholder(tf.float32, [None, None, None, 4])
out_image = network(in_image)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
    pass

#raw
ratio = np.random.randint(100,300,1)[0]   #ampilfication ratio (get ramdomly)
raw = rawpy.imread(input_image_path)

input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio

input_full = np.minimum(input_full, 1.0)

output = sess.run(out_image, feed_dict={in_image: input_full})
output = np.minimum(np.maximum(output, 0), 1)

output = output[0, :, :, :]


input_image_basename = basename(input_image_path)
input_image_basename = input_image_basename[:-4]

scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(result_dir + input_image_basename+"_result.png")

# #png
# png = Image.open(input_image_path)
# ratio = 5#np.random.randint(100,300,1)[0]   #ampilfication ratio (get ramdomly)
# png_image_full = np.array(png)
# #resize image to avoid OOM 
# H = png_image_full.shape[0]
# W = png_image_full.shape[1]
# png_image_full = png_image_full[0:H:2,0:W:2,:]
# png_image_full = np.expand_dims(pack_png(png_image_full),axis=0)*ratio
# png_image_full = np.minimum(png_image_full,1.0)

# output = sess.run(out_image, feed_dict={in_image: png_image_full})
# output = np.minimum(np.maximum(output, 0), 1)

# output = output[0, :, :, :]


# input_image_basename = basename(input_image_path)
# input_image_basename = input_image_basename[:-4]

# scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(result_dir + input_image_basename+"_result.png")

