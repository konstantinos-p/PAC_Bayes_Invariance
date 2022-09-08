import numpy as np
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.datasets import cifar10
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import EarlyStopping
K.set_image_dim_ordering('th')
from keras import regularizers
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from utils_spectral_norm import convParsevalReg
from utils_spectral_norm import denseParsevalReg

# !!! Limit memory consumption
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
# !!! Limit memory consumption

"""
In this script we train a 3 simple 3-layer neural networks. We use 2 convolutional layers and 1 linear layer. 
"""


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

#Norm regularizes
l2_model1_conv1 = regularizers.l2(0)
l2_model1_conv2 = regularizers.l2(0)
l2_model1_linear = regularizers.l2(0)

#Parseval regularizers
pars_con1 = convParsevalReg('conv2d_1',0.01)
pars_con2 = convParsevalReg('conv2d_2',0.01)
pars_den1 = denseParsevalReg('dense_1',0.01,5000)


# Create the model
model1 = Sequential()
model1.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation='relu', padding='valid',kernel_regularizer=l2_model1_conv1))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(64, (3, 3), activation='relu', padding='valid',kernel_regularizer=l2_model1_conv2))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Flatten())
model1.add(Dense(num_classes, activation='softmax',kernel_regularizer = l2_model1_linear))

# Optimisation Parameters
epochs1 = 1
lrate1 = 0.01
decay1 = lrate1/epochs1
sgd1 = SGD(lr=lrate1, momentum=0.9, decay=decay1, nesterov=False)

# Compile models
model1.compile(loss='categorical_crossentropy', optimizer=sgd1, metrics=['accuracy'])
print(model1.summary())


indexes = np.random.permutation(X_train.shape[0])
batch_size = 10000
batch_num = 1


earlyStopping = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=4,
                              verbose=0, mode='auto')

#Train the model
model1.fit(X_train[indexes[0:batch_size], :, :, :], y_train[indexes[0:batch_size], :], validation_data=(X_test, y_test), epochs=epochs1, batch_size=32,callbacks=[pars_con1,pars_con2,pars_den1])

#Model1 Norms
tmp_layer = model1.get_layer('conv2d_1')
tmp_weights = tmp_layer.get_weights()[0]
model1_conv1 = 30*30*np.linalg.norm(tmp_weights)*np.linalg.norm(tmp_weights)

tmp_layer = model1.get_layer('conv2d_2')
tmp_weights = tmp_layer.get_weights()[0]
model1_conv2 = 13*13*np.linalg.norm(tmp_weights)*np.linalg.norm(tmp_weights)

tmp_layer = model1.get_layer('dense_1')
tmp_weights = tmp_layer.get_weights()[0]
model1_dense = np.linalg.norm(tmp_weights)*np.linalg.norm(tmp_weights)

print('Convolutional layer 1 ||.||_F^2: ',model1_conv1)
print('Convolutional layer 2 ||.||_F^2: ',model1_conv2)
print('Dense layer           ||.||_F^2: ',model1_dense)
