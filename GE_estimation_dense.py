import numpy as np

from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LocallyConnected2D
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
from utils_spectral_norm import denseParsevalReg

# !!! Limit memory consumption
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
# !!! Limit memory consumption

# !!!!!!
"""
Here we train 3 simple 3-layer neural networks. We use 2 non-linear layers and 1 linear layer. We alternate between
convolutional, locally connected and dense non-linear layers.
"""
# !!!!!!


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

#Norm regularizers
l2_model3_full1 = regularizers.l2(0)
l2_model3_full2 = regularizers.l2(0)
l2_model3_linear = regularizers.l2(0)

#Parseval regularizers
pars_den1 = denseParsevalReg('dense_1',0.01,5000)
pars_den2 = denseParsevalReg('dense_2',0.01,5000)
pars_den3 = denseParsevalReg('dense_3',0.01,5000)


# Create the model
model3 = Sequential()
model3.add(Flatten(input_shape=(3, 32, 32)))
model3.add(Dense(28800, activation='relu',kernel_regularizer = l2_model3_full1))
model3.add(Dense(2400, activation='relu',kernel_regularizer = l2_model3_full2))
model3.add(Dense(num_classes, activation='softmax',kernel_regularizer = l2_model3_linear))

# Optimisation Parameters
epochs3 = 1
lrate3 = 0.01
decay3 = lrate3/epochs3
sgd3 = SGD(lr=lrate3, momentum=0.9, decay=decay3, nesterov=False)

# Compile models
model3.compile(loss='categorical_crossentropy', optimizer=sgd3, metrics=['accuracy'])
print(model3.summary())

indexes = np.random.permutation(X_train.shape[0])
batch_size = 10000
batch_num = 1

earlyStopping = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=4,
                              verbose=0, mode='auto')

model3.fit(X_train[indexes[0:batch_size], :, :, :], y_train[indexes[0:batch_size], :],validation_data=(X_test, y_test), epochs=epochs3, batch_size=32,callbacks=[pars_den1,pars_den2,pars_den3])

#Model3 Norms
tmp_layer = model3.get_layer('dense_1')
tmp_weights = tmp_layer.get_weights()[0]
model3_dense1 = np.linalg.norm(tmp_weights)*np.linalg.norm(tmp_weights)

tmp_layer = model3.get_layer('dense_2')
tmp_weights = tmp_layer.get_weights()[0]
model3_dense2 = np.linalg.norm(tmp_weights)*np.linalg.norm(tmp_weights)

tmp_layer = model3.get_layer('dense_3')
tmp_weights = tmp_layer.get_weights()[0]
model3_dense3 = np.linalg.norm(tmp_weights)*np.linalg.norm(tmp_weights)

print('Dense layer 1 ||.||_F^2: ',model3_dense1)
print('Dense layer 2 ||.||_F^2: ',model3_dense2)
print('Dense layer 3 ||.||_F^2: ',model3_dense3)

end  = 1
