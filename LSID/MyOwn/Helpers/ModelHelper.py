from __future__ import division
import numpy as np
import rawpy
import scipy
import tensorflow as tf
import tensorflow.contrib.slim as slim
from PIL import Image

class SIDModel:
    def __init__(self, checkpoint_dir):
        self.sess = tf.Session()
        self.in_image = tf.placeholder(tf.float32, [None, None, None, 4])
        self.out_image = self.__network(self.in_image)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            pass

    def run_png(self, input_path, save_path):
        png = Image.open(input_path)
        ratio = 30  # from 5 - 300 depend on intensity of your image    #ampilfication ratio
        png_image_full = np.array(png)
        ## resize image size if need to avoid OOM
        # H = png_image_full.shape[0]
        # W = png_image_full.shape[1]
        # png_image_full = png_image_full[0:H:2, 0:W:2, :]

        # pack the image to 4 channels of colors (RGBR),
        png_image_full = np.expand_dims(self.__pack_png(png_image_full), axis=0) * ratio
        png_image_full = np.minimum(png_image_full, 1.0)

        self.__run_on_numpy_array(png_image_full, save_path)

    def run_raw(self, input_path, save_path):
        ratio = np.random.randint(100, 300, 1)[0]  # ampilfication ratio (get ramdomly)
        raw = rawpy.imread(input_path)

        input_full = np.expand_dims(self.__pack_raw(raw), axis=0) * ratio
        input_full = np.minimum(input_full, 1.0)

        self.__run_on_numpy_array(input_full, save_path)

    def run_dng(self, input_path, save_path):
        ratio = 15# ampilfication ratio (get ramdomly)
        raw = rawpy.imread(input_path)

        input_full = np.expand_dims(self.__pack_dng(raw), axis=0) * ratio
        input_full = np.minimum(input_full, 1.0)

        self.__run_on_numpy_array(input_full, save_path)

    def __run_on_numpy_array(self, color_array, save_path):
        output = self.sess.run(self.out_image, feed_dict={self.in_image: color_array})
        output = np.minimum(np.maximum(output, 0), 1)
        output = output[0, :, :, :]
        print(output*255)
        scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(
            save_path)

    def run_multi_images(self):
        # implement later
        pass

    def __pack_raw(self, raw):
        # pack Bayer image to 4 channels
        black_level = np.amin(raw.black_level_per_channel)  # assume they're all the same
        im = raw.raw_image_visible.astype(np.float32)
        im = np.maximum(im - black_level, 0) / (16383 - black_level)  # subtract the black level

        im = np.expand_dims(im, axis=2)
        img_shape = im.shape
        H = img_shape[0]
        W = img_shape[1]
        # if bayer patern = [[G, B]; [R, G]]
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

    def __pack_png(self, png):
        # pack png image to 4 channels RGBG, currently PNG is RGB
        H = png.shape[0]
        W = png.shape[1]
        if len(png.shape)==2:
            layer = np.expand_dims(png,axis=2)
            png = np.concatenate((layer,layer,layer,layer),axis=2)
            
        # extract to three channels
        G = np.expand_dims(png[:, :, 0], axis=2)
        R = np.expand_dims(png[:, :, 1], axis=2)
        B = np.expand_dims(png[:, :, 2], axis=2)
        arr = np.concatenate((R, G, B, (R+G*2+B)/4), axis=2)
        arr = arr * 60.2352941176 + 512  # make value of the image in range [2^9 to 2^14-1]
        
        shape = arr.shape
        H = shape[0]
        W = shape[1]
        arr = arr[0:H:2,0:W:2,:]

        black_level = 512
        im = arr.astype(np.float32)
        im = np.maximum(im - black_level, 0) / (16383 - black_level)  # subtract the black level
        return im

    def __upsample_and_concat(self, x1, x2, output_channels, in_channels):
        pool_size = 2
        deconv_filter = tf.Variable(
            tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
        deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

        deconv_output = tf.concat([deconv, x2], 3)
        deconv_output.set_shape([None, None, None, output_channels * 2])

        return deconv_output

    def __network(self, input):
        conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=self.__lrelu, scope='g_conv1_1')
        conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=self.__lrelu, scope='g_conv1_2')
        pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

        conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=self.__lrelu, scope='g_conv2_1')
        conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=self.__lrelu, scope='g_conv2_2')
        pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

        conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=self.__lrelu, scope='g_conv3_1')
        conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=self.__lrelu, scope='g_conv3_2')
        pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

        conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=self.__lrelu, scope='g_conv4_1')
        conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=self.__lrelu, scope='g_conv4_2')
        pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

        conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=self.__lrelu, scope='g_conv5_1')
        conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=self.__lrelu, scope='g_conv5_2')

        up6 = self.__upsample_and_concat(conv5, conv4, 256, 512)
        conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=self.__lrelu, scope='g_conv6_1')
        conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=self.__lrelu, scope='g_conv6_2')

        up7 = self.__upsample_and_concat(conv6, conv3, 128, 256)
        conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=self.__lrelu, scope='g_conv7_1')
        conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=self.__lrelu, scope='g_conv7_2')

        up8 = self.__upsample_and_concat(conv7, conv2, 64, 128)
        conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=self.__lrelu, scope='g_conv8_1')
        conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=self.__lrelu, scope='g_conv8_2')

        up9 = self.__upsample_and_concat(conv8, conv1, 32, 64)
        conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=self.__lrelu, scope='g_conv9_1')
        conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=self.__lrelu, scope='g_conv9_2')

        conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
        out = tf.depth_to_space(conv10, 2)
        return out

    def __lrelu(self,x):
        return tf.maximum(x * 0.2, x)

    def __pack_dng(self, raw):
        black_level = 512  # assume they're all the same
        im = raw.raw_image_visible.astype(np.float32)
        im = (im - 64) * 16.55 + 512

        im = np.maximum(im - black_level, 0) / (16383 - black_level)  # subtract the black level

        im = np.expand_dims(im, axis=2)
        img_shape = im.shape
        H = img_shape[0]
        W = img_shape[1]
        # # if bayer patern = [[G, B]; [R, G]]
        # out = np.concatenate((im[1:H:2, 0:W:2, :],
        #                       im[1:H:2, 1:W:2, :],
        #                       im[0:H:2, 1:W:2, :],
        #                       im[0:H:2, 0:W:2, :]), axis=2)
        # #if bayer patern = [[R, G]; [G, B]]
        # out = np.concatenate((im[0:H:2, 0:W:2, :],
        #                       im[0:H:2, 1:W:2, :],
        #                       im[1:H:2, 1:W:2, :],
        #                       im[1:H:2, 0:W:2, :]), axis=2)
        # if bayer patern = [[B, G]; [G, R]]
        out = np.concatenate((im[1:H:2, 1:W:2, :],
                              im[0:H:2, 1:W:2, :],
                              im[0:H:2, 0:W:2, :],
                              im[1:H:2, 0:W:2, :]), axis=2)
        return out
