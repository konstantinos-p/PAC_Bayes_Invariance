import numpy as np

from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LocallyConnected2D
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
from utils_spectral_norm import localParsevalReg
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
l2_model2_local1 = regularizers.l2(0)
l2_model2_local2 = regularizers.l2(0)
l2_model2_linear = regularizers.l2(0)

#Parseval regularizers
pars_loc1 = localParsevalReg('locally_connected2d_1',0.01)
pars_loc2 = localParsevalReg('locally_connected2d_2',0.01)
pars_den1 = denseParsevalReg('dense_1',0.01,5000)

# Create the model
model2 = Sequential()
model2.add(LocallyConnected2D(32, (3, 3), input_shape=(3, 32, 32), activation='relu', padding='valid',kernel_regularizer=l2_model2_local1))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(LocallyConnected2D(64, (3, 3), activation='relu', padding='valid',kernel_regularizer=l2_model2_local2))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Flatten())
model2.add(Dense(num_classes, activation='softmax',kernel_regularizer = l2_model2_linear))

# Optimisation Parameters
epochs2 = 1
lrate2 = 0.01
decay2 = lrate2/epochs2
sgd2 = SGD(lr=lrate2, momentum=0.9, decay=decay2, nesterov=False)


# Compile models
model2.compile(loss='categorical_crossentropy', optimizer=sgd2, metrics=['accuracy'])
print(model2.summary())



indexes = np.random.permutation(X_train.shape[0])
batch_size = 10000
batch_num = 1


earlyStopping = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=4,
                              verbose=0, mode='auto')

#model2.fit(X_train[indexes[0:batch_size], :, :, :], y_train[indexes[0:batch_size], :],validation_data=(X_test, y_test), epochs=epochs2, batch_size=32)
model2.fit(X_train[indexes[0:batch_size], :, :, :], y_train[indexes[0:batch_size], :],validation_data=(X_test, y_test), epochs=epochs2, batch_size=32,callbacks=[pars_den1,pars_loc1,pars_loc2])


#Model2 Norms

tmp_layer = model2.get_layer('locally_connected2d_1')
tmp_weights = tmp_layer.get_weights()[0]
model2_local1 = np.linalg.norm(tmp_weights)*np.linalg.norm(tmp_weights)

tmp_layer = model2.get_layer('locally_connected2d_2')
tmp_weights = tmp_layer.get_weights()[0]
model2_local2 = np.linalg.norm(tmp_weights)*np.linalg.norm(tmp_weights)

tmp_layer = model2.get_layer('dense_1')
tmp_weights = tmp_layer.get_weights()[0]
model2_dense = np.linalg.norm(tmp_weights)*np.linalg.norm(tmp_weights)

print('Locally Connected layer 1 ||.||_F^2: ',model2_local1)
print('Locally Connected layer 2 ||.||_F^2: ',model2_local2)
print('Dense layer               ||.||_F^2: ',model2_dense)

end  = 1
