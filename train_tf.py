import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib
from collections import OrderedDict
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import keras
from keras.optimizers import SGD,Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10

from keras.layers import Input, Dense, Dropout, Lambda,Reshape
from utils import *
from wide_resnet import *
from cosine_annealing import *
from dataset import Cifar10ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,EarlyStopping
from shake_shake_model import create_shakeshake_cifar
from keras.applications.mobilenet import MobileNet
from SE_resnext import SEResNeXt
from keras.applications.densenet import DenseNet121
from scipy import misc
from keras_efficientnets import EfficientNetB5
from auto_augment import cutout, apply_policy
from keras import backend as K
import tensorflow as tf
from keras.utils import np_utils
from keras.layers import Layer
from keras_mixnets import MixNetLarge,MixNetSmall
from lookahead import Lookahead
from keras_SGDR import SGDRScheduler
# Compatible with tensorflow backend
from keras.layers import Input, Dense, Dropout, Lambda,Reshape,BatchNormalization, Lambda, Embedding
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import keras
from keras.utils import np_utils
from keras_efficientnets import EfficientNetB7,EfficientNetB5
from EfficientNet_tf.model import EfficientNetB7
from WRN import WideResNet

# Compatible with tensorflow backend

def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -tf.keras.backend.mean(alpha * tf.keras.backend.pow(1. - pt_1, gamma) * tf.keras.backend.log(pt_1+tf.keras.backend.epsilon())) - \
               tf.keras.backend.mean((1 - alpha) * tf.keras.backend.pow(pt_0, gamma) * tf.keras.backend.log(1. - pt_0+tf.keras.backend.epsilon()))
	return focal_loss_fixed

def triplet_center_loss(y_true, y_pred, n_classes= 10, alpha=0.4):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    print('y_pred.shape = ', y_pred)

    total_lenght = y_pred.shape.as_list()[-1]
    #     print('total_lenght=',  total_lenght)
    #     total_lenght =12

    # repeat y_true for n_classes and == np.arange(n_classes)
    # repeat also y_pred and apply mask
    # obtain min for each column min vector for each class

    classes = tf.range(0, n_classes,dtype=tf.float32)
    y_pred_r = tf.reshape(y_pred, (tf.shape(y_pred)[0], 1))
    y_pred_r = tf.keras.backend.repeat(y_pred_r, n_classes)

    y_true_r = tf.reshape(y_true, (tf.shape(y_true)[0], 1))
    y_true_r = tf.keras.backend.repeat(y_true_r, n_classes)

    mask = tf.equal(y_true_r[:, :, 0], classes)

    #mask2 = tf.ones((tf.shape(y_true_r)[0], tf.shape(y_true_r)[1]))  # todo inf

    # use tf.where(tf.equal(masked, 0.0), np.inf*tf.ones_like(masked), masked)

    masked = y_pred_r[:, :, 0] * tf.cast(mask, tf.float32) #+ (mask2 * tf.cast(tf.logical_not(mask), tf.float32))*tf.constant(float(2**10))
    masked = tf.where(tf.equal(masked, 0.0), np.inf*tf.ones_like(masked), masked)

    minimums = tf.math.reduce_min(masked, axis=1)

    loss = K.max(y_pred - minimums +alpha ,0)

    # obtain a mask for each pred


    return loss
# def focal_loss(gamma=2., alpha=.25):
# 	def focal_loss_fixed(y_true, y_pred):
# 		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
# 		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
# 		return (K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0++K.epsilon())))*0.4+keras.losses.categorical_crossentropy(y_true, y_pred)*0.6
# 	return focal_loss_fixed



def dice_loss(smooth=1e-5, thresh=0.5):
    def focal_loss_fixed(y_true, y_pred,gamma=2., alpha=.25):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -tf.keras.backend.mean(
            alpha * tf.keras.backend.pow(1. - pt_1, gamma) * tf.keras.backend.log(pt_1 + tf.keras.backend.epsilon())) - \
               tf.keras.backend.mean((1 - alpha) * tf.keras.backend.pow(pt_0, gamma) * tf.keras.backend.log(
                   1. - pt_0 + tf.keras.backend.epsilon()))

    def dice_coef(y_true, y_pred, smooth=1e-5, thresh=0.5):
        a = focal_loss_fixed(y_true,y_pred)
        y_pred = y_pred > thresh
        return tf.keras.losses.categorical_crossentropy(y_pred,y_true)+a

    def dice(y_true, y_pred):
        return dice_coef(y_true, y_pred, smooth, thresh)
    return dice







def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--depth', default=20, type=int)
    parser.add_argument('--width', default=10, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--cutout', default=True, type=str2bool)
    parser.add_argument('--auto-augment', default=True, type=str2bool)

    args = parser.parse_args()

    return args
from keras import backend as K
from keras.optimizers import Optimizer


# Ported from https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam.py
class RectifiedAdam(Optimizer):
    """RectifiedAdam optimizer.

    Default parameters follow those provided in the original paper.

    # Arguments
        lr: float >= 0. Learning rate.
        final_lr: float >= 0. Final learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        gamma: float >= 0. Convergence speed of the bound function.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        weight_decay: Weight decay weight.
        amsbound: boolean. Whether to apply the AMSBound variant of this
            algorithm.

    # References
        - [On the Variance of the Adaptive Learning Rate and Beyond]
          (https://arxiv.org/abs/1908.03265)
        - [Adam - A Method for Stochastic Optimization]
          (https://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond]
          (https://openreview.net/forum?id=ryQu7f-RZ)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., weight_decay=0.0, **kwargs):
        super(RectifiedAdam, self).__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')

        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay

        self.weight_decay = float(weight_decay)

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)

            beta2_t = self.beta_2 ** t
            N_sma_max = 2 / (1 - self.beta_2) - 1
            N_sma = N_sma_max - 2 * t * beta2_t / (1 - beta2_t)

            # apply weight decay
            if self.weight_decay != 0.:
                p_wd = p - self.weight_decay * lr * p
            else:
                p_wd = None

            if p_wd is None:
                p_ = p
            else:
                p_ = p_wd

            def gt_path():
                step_size = lr * K.sqrt(
                    (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max /
                    (N_sma_max - 2)) / (1 - self.beta_1 ** t)

                denom = K.sqrt(v_t) + self.epsilon
                p_t = p_ - step_size * (m_t / denom)

                return p_t

            def lt_path():
                step_size = lr / (1 - self.beta_1 ** t)
                p_t = p_ - step_size * m_t

                return p_t

            p_t = K.switch(N_sma > 5, gt_path, lt_path)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'weight_decay': self.weight_decay}
        base_config = super(RectifiedAdam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def en_train(model,x_train,y_train,x_test,y_test,weights_file='',name=''):
    args = parse_args()
    args.name = 'WideResNet'+name
    # if args.name is None:
    #
    #     if args.cutout:
    #         args.name += '_wCutout'
    #     if args.auto_augment:
    #         args.name += '_wAutoAugment'
    #
    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('./models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, './models/%s/args.pkl' %args.name)

    # create model
    # input_layer = Input(shape=(28,28,1))
    # input_image_ = Lambda(lambda x: K.repeat_elements(K.expand_dims(x, 3), 3, 3))(input_layer)
    # # model = ResNext(img_dim, depth=depth, cardinality=cardinality, width=width, weights=None, classes=nb_classes)
    # se_resnext= SEResNeXt()
    # model1=se_resnext.build_model(inputs=input_layer,num_classes=10,include_top=False) #wrn
    # model1 = Dense(10)(model1)
    # # model1.load_weights("",by_name=True)
    # # model1 = NASNetMobile(input_tensor=input_image_, include_top=False, pooling='avg', weights="./model/nasnet.hdf5")
    #
    #
    # model2 = create_shakeshake_cifar(n_classes=10,include_top=False,x_in=input_layer)
    # model2 = Dense(10)(model2)
    #


    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()


    datagen = Cifar10ImageDataGenerator(args)

    x_test = datagen.standardize(x_test)

    # y_train = keras.utils.to_categorical(y_train, 11)
    y_test_onehot = keras.utils.to_categorical(y_test, 10)
    # weights_file = "models/merge.h5"
    if os.path.exists(weights_file):
        model.load_weights(weights_file,by_name=True)
        # model = tf.keras.models.load_model(weights_file)
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9,  # 当标准评估停止提升时，降低学习速率。
                                   cooldown=0, patience=20, min_lr=1e-8)

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(weights_file, monitor="val_acc", save_best_only=True, mode='auto')
    earlyStopping =tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=20, verbose=1, mode='auto')
    callbacks = [
        # ModelCheckpoint('models/%s/model.hdf5'%args.name, verbose=1, save_best_only=True),
        lr_reducer,
        earlyStopping,
        model_checkpoint,
        CSVLogger('models/%s/log.csv'%args.name),
        CosineAnnealingScheduler(T_max=args.epochs, eta_max=0.05, eta_min=4e-4)
    ]

    model.fit_generator(datagen.flow(x_train, y_train, batch_size=args.batch_size),
                        steps_per_epoch=len(x_train)//args.batch_size,
                        validation_data=(x_test,y_test),
                        epochs=args.epochs, verbose=1,
                        callbacks=callbacks)
    return model








    # os.system('shutdown -s -f -t 59')

if __name__ == '__main__':
    opt = tf.train.AdamOptimizer(learning_rate=3e-4)
    # input =Input(shape=(56,56,1))
    # input_image_ = Lambda(lambda x: K.repeat_elements(x, 3, 3))(input)
    # print(input_image_.shape)
    # base_model = EfficientNetB5(input_tensor=input_image_, include_top=False, pooling='avg', weights='imagenet')
    # # x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    # # base_model = MobileNet(input_tensor=input_image_, alpha=1.0)
    # x = Dropout(0.5)(base_model.output)
    # # predict = Dense(10, activation='softmax')(output)
    #
    # # labels = Input((10,))
    # # x = Dropout(0.5)(x)
    # real = Dense(10, kernel_initializer='he_normal',
    #              kernel_regularizer=keras.regularizers.l2(1e-4), name='real_output')(x)
    # x = BatchNormalization(name='fcbn1')(real)
    # x = keras.layers.ReLU()(x)
    # predict = Dense(10, activation='softmax', name='output')(x)
    #
    # target_input = Input((1,))
    # center = Embedding(10, 10)(target_input)
    # l2_loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss')(
    #     [x, center])
    # x = ArcFace(n_classes=10, regularizer=keras.regularizers.l2(1e-4), name='output')([x, labels])
    # model = keras.models.Model([input_image, labels], x)
    # model.compile(optimizer=keras.optimizers.SGD(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # model = Model(inputs=[input, target_input], outputs=[predict, l2_loss])

    model  = WideResNet(depth=28,width=2)
    # model = EfficientNetB7(classes=10)
    model.compile(optimizer=tf.keras.optimizers.SGD(0.01), loss=[dice_loss()],#'categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()


    mnist = read_data_sets('./data/fashion', reshape=False, validation_size=0,
                           source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
    x_train = mnist.train.images
    y_train = mnist.train.labels
    x_test = mnist.test.images
    y_test = mnist.test.labels
    height, width=28,28
    # x_train = x_train.reshape((-1, 28, 28))
    # x_train = np.array([misc.imresize(x, (height, width)).astype(float) for x in tqdm(iter(x_train))]) / 255.
    x_train = x_train.reshape((-1, height, width,1))
    x_train = np.concatenate([x_train,x_train,x_train],axis=-1)
    # x_test = x_test.reshape((-1, 28, 28))
    # x_test = np.array([misc.imresize(x, (height, width)).astype(float) for x in tqdm(iter(x_test))]) / 255.
    x_test = x_test.reshape((-1, height, width,1))
    x_test = np.concatenate([x_test,x_test,x_test],axis=-1)
    x_train = np.uint8(x_train*255)
    x_test = np.uint8(x_test*255)

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    # main()
    en_train(model, x_train, y_train, x_test, y_test, weights_file='models/wrn.h5', name='aaa')
