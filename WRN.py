
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from scipy import misc
import numpy
import numpy.random
import tensorflow as tf
import  numpy as np
weight_decay = 5e-4
from keras import layers

# import keras
# from keras.models import Model
# from keras.layers import Dense, Conv2D, BatchNormalization, Activation
# from keras.layers import Input, Add, GlobalAveragePooling2D, Dropout
# from keras import regularizers

weight_decay = 5e-4


class ShakeShake(tf.keras.layers.Layer):
    """ Shake-Shake-Image Layer """

    def __init__(self, **kwargs):
        super(ShakeShake, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ShakeShake, self).build(input_shape)

    def call(self, x):
        # unpack x1 and x2
        assert isinstance(x, list)
        x1, x2 = x
        # create alpha and beta
        batch_size = tf.keras.backend.shape(x1)[0]
        alpha = tf.keras.backend.random_uniform((batch_size, 1, 1, 1))
        beta = tf.keras.backend.random_uniform((batch_size, 1, 1, 1))
        # shake-shake during training phase
        def x_shake():
            return beta * x1 + (1 - beta) * x2 + tf.keras.backend.stop_gradient((alpha - beta) * x1 + (beta - alpha) * x2)
        # even-even during testing phase
        def x_even():
            return 0.5 * x1 + 0.5 * x2
        return tf.keras.backend.in_train_phase(x_shake, x_even)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape[0]


def swish(x):
    return tf.keras.backend.sigmoid(x) * x


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser


def random_crop_image(image):
      height, width = image.shape[:2]
      random_array = numpy.random.random(size=4)
      w = int((width*0.5)*(1+random_array[0]*0.5))
      h = int((height*0.5)*(1+random_array[1]*0.5))
      x = int(random_array[2]*(width-w))
      y = int(random_array[3]*(height-h))

      image_crop = image[y:h+y,x:w+x,0:3]



class Swish(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, training=None):
        return tf.nn.swish(inputs)

class DropConnect(tf.keras.layers.Layer):

    def __init__(self, drop_connect_rate=0., **kwargs):
        super(DropConnect, self).__init__(**kwargs)
        self.drop_connect_rate = float(drop_connect_rate)

    def call(self, inputs, training=None):

        def drop_connect():
            keep_prob = 1.0 - self.drop_connect_rate

            # Compute drop_connect tensor
            batch_size = tf.shape(inputs)[0]
            random_tensor = keep_prob
            random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
            binary_tensor = tf.floor(random_tensor)
            output = (inputs / keep_prob) * binary_tensor
            return output

        return tf.keras.backend.in_train_phase(drop_connect, inputs, training=training)

    def get_config(self):
        config = {
            'drop_connect_rate': self.drop_connect_rate,
        }
        base_config = super(DropConnect, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



def SEBlock(input_filters, se_ratio, expand_ratio, data_format=None):
    if data_format is None:
        data_format = tf.keras.backend.image_data_format()

    num_reduced_filters = max(
        1, int(input_filters * se_ratio))
    filters = input_filters * expand_ratio

    if data_format == 'channels_first':
        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = -1
        spatial_dims = [1, 2]

    def block(inputs):
        x = inputs
        x = tf.keras.layers.Lambda(lambda a: tf.keras.backend.mean(a, axis=spatial_dims, keepdims=True))(x)
        x = tf.keras.layers.Conv2D(
            num_reduced_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer='he_normal',
            padding='same',
            use_bias=True)(x)
        x = Swish()(x)
        # Excite
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer='he_normal',
            padding='same',
            use_bias=True)(x)
        x = tf.keras.layers.Activation('sigmoid')(x)
        out = tf.keras.layers.Multiply()([x, inputs])
        return out

    return block



def MBConvBlock(input_filters, output_filters,
                kernel_size, strides,
                expand_ratio, se_ratio,
                id_skip, drop_connect_rate,
                batch_norm_momentum=0.99,
                batch_norm_epsilon=1e-3,
                data_format=None):

    if data_format is None:
        data_format = tf.keras.backend.image_data_format()

    if data_format == 'channels_first':
        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = -1
        spatial_dims = [1, 2]

    has_se = (se_ratio is not None) and (se_ratio > 0) and (se_ratio <= 1)
    filters = input_filters * expand_ratio

    def block(inputs):

        if expand_ratio != 1:
            x = tf.keras.layers.Conv2D(
                filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer='he_normal',
                padding='same',
                use_bias=False)(inputs)
            x = tf.keras.layers.BatchNormalization(
                axis=channel_axis,
                momentum=batch_norm_momentum,
                epsilon=batch_norm_epsilon)(x)
            x = Swish()(x)
        else:
            x = inputs

        x = tf.keras.layers.DepthwiseConv2D(
            [kernel_size, kernel_size],
            strides=strides,
            depthwise_initializer='he_normal',
            padding='same',
            use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)(x)
        x = Swish()(x)

        if has_se:
            x = SEBlock(input_filters, se_ratio, expand_ratio,
                        data_format)(x)

        # output phase

        x = tf.keras.layers.Conv2D(
            output_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer='he_normal',
            padding='same',
            use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)(x)

        if id_skip:
            if all(s == 1 for s in strides) and (
                    input_filters == output_filters):

                # only apply drop_connect if skip presents.
                if drop_connect_rate:
                    x = DropConnect(drop_connect_rate)(x)

                # x = tf.keras.layers.Add()([x, inputs])
                x = ShakeShake()([x, inputs])
        return x

    return block

def conv3x3(input, out_planes, stride=1):
    """3x3 convolution with padding"""
    return tf.keras.layers.Conv2D(out_planes, kernel_size=3, strides=stride,
                    padding='same', use_bias=False, kernel_initializer='he_normal',
                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)


def conv1x1(input, out_planes, stride=1):
    """1x1 convolution"""
    return tf.keras.layers.Conv2D(out_planes, kernel_size=1, strides=stride,
                    padding='same', use_bias=False, kernel_initializer='he_normal',
                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)


def BasicBlock(input, planes, dropout, stride=1):
    inplanes = input.shape[3]

    out = tf.keras.layers.BatchNormalization()(input)
    # out = SwitchableNormalization()(input)

    # out = Activation('elu')(out)
    # out = swish(out)

    # out = tf.keras.layers.Activation(swish)(out)
    out = Swish()(out)
    out = conv3x3(out, planes, stride)
    out = tf.keras.layers.BatchNormalization()(out)
    # out = SwitchableNormalization()(out)
    # out = Activation('elu')(out)
    # out = tf.keras.layers.Activation(swish)(out)
    out = Swish()(out)
    out = tf.keras.layers.Dropout(dropout)(out)
    out = conv3x3(out, planes)

    if stride != 1 or inplanes != planes:
        shortcut = conv1x1(input, planes, stride)
    else:
        shortcut = out

    # out = tf.keras.layers.Add()([out, shortcut])
    out = ShakeShake()([out, shortcut])
    return out


def WideResNet(depth=40, width=10,num_classes=10, dropout=0.3):
    layer = (depth - 4) // 6

    input = tf.keras.layers.Input(shape=(28, 28, 3))
    x = tf.keras.layers.BatchNormalization()(input)
    x = conv3x3(x, 16)
    for _ in range(layer):
        # x = BasicBlock(x, 16*width, dropout)
        x = MBConvBlock(16,16*width,kernel_size=3, strides=(1, 1), se_ratio=0.25, expand_ratio=6,id_skip=True,drop_connect_rate=0.2)(x)
    x = BasicBlock(x, 32*width, dropout, 2)
    for _ in range(layer-1):
        # x = BasicBlock(x, 32*width, dropout)
        x = MBConvBlock(32,32*depth,kernel_size=3, strides=(2,2), se_ratio=0.25, expand_ratio=6,id_skip=True,drop_connect_rate=0.2)(x)
    #
    x = BasicBlock(x, 64*width, dropout, 2)
    for _ in range(layer-1):
        # x = BasicBlock(x, 64*width, dropout)
        x = MBConvBlock(64, 64 * depth, kernel_size=3, strides=(1, 1), se_ratio=0.25, expand_ratio=6, id_skip=True,
                        drop_connect_rate=0.2)(x)

    x = tf.keras.layers.BatchNormalization()(x)
    # x = SwitchableNormalization()(x)
    # x = Activation('elu')(x)
    x = Swish()(x)
    print(x.shape)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    output = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)

    model = tf.keras.Model(input, output)
    model.summary()

    return model