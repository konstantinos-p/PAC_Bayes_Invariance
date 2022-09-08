"""
In this script define functions that calculate the spectral norm for different keras layers.
Specifically we can calculate the spectral norm of
1) convolutional
2) locally connected layers
This is a difficult problem as the full operators are large sparse matrices.
"""

import numpy as np
from keras.callbacks import Callback

def compute_spectral_norm_of_conv(weights,featuremap_dim):
    '''
    The dimensions of the weight matrix are
    trainable_weights[0] = (filter_d1,filter_d2,input_channels,output_channels) This is the kernel.
    trainable_weights[1] = (output_channels) This is the bias.
    '''
    dim1 = weights.shape[0]
    dim2 = weights.shape[1]
    dim3 = weights.shape[2]
    dim4 = weights.shape[3]

    fourier_weights = np.empty((featuremap_dim*featuremap_dim,dim3,dim4),dtype='cfloat')
    tmp_featuremap = np.zeros((featuremap_dim,featuremap_dim))


    for i in range(0,dim3):
        for j in range(0,dim4):
            tmp_featuremap[0:dim1,0:dim2] = weights[:,:,i,j]
            fourier_weights[:,i,j] = np.reshape(np.fft.fft2(tmp_featuremap),-1)

    max_l2 = 0

    for i in range(0,featuremap_dim*featuremap_dim):
        if np.linalg.norm(fourier_weights[i,:,:],2)>max_l2:
            max_l2 = np.linalg.norm(fourier_weights[i,:,:],2)
    return max_l2

def compute_spectral_norm_of_locally_connected(weights,featuremap_dim,epsilon,max_it,hta):
    '''
    The dimensions of the weight matrix are
    trainable_weights[0] = (filter_d1,filter_d2,input_channels,output_channels) This is the kernel.
    trainable_weights[1] = (output_channels) This is the bias.
    '''
    dim1 = weights.shape[0]
    dim2 = weights.shape[1]
    dim3 = weights.shape[2]
    dim4 = weights.shape[3]

    flat_maps = np.zeros((featuremap_dim*dim3*dim4,1))

    w = np.random.normal(0,1,size=(featuremap_dim*dim3*dim4,1))
    w  = w/np.linalg.norm(w)

    singular_value_current = 0
    singular_value = np.zeros((max_it,1))

    for i in range(0,max_it):

        # Create a convolutional filter map
        featuremaps = np.zeros((featuremap_dim, featuremap_dim, dim3, dim4))
        filter_loc = np.random.randint(1,featuremap_dim-1,size=(2))
        featuremaps[filter_loc[0]-1:filter_loc[0]+1,filter_loc[1]-1:filter_loc[1]+1,:,:] = weights
        flat_maps = featuremaps[:]

        w = w + hta*flat_maps*(flat_maps.T@w)
        w = w / np.linalg.norm(w)

        singular_value[i] = np.power(flat_maps.T@w,2)

        if np.abs(singular_value[i]-singular_value_current)>epsilon:
            singular_value_current = singular_value[i]
        else:
            break
    return singular_value

#Create Parseval Regularisation
class convParsevalReg(Callback):
    '''
    This is the parseval regularization class for a convolutional layer.
    '''
    def __init__(self,layer_name,beta_par):
        self.validation_data = None
        self.model = None
        self.layername = layer_name
        self.beta = beta_par

    def on_batch_end(self, batch, logs={}):

        #This is an implementation of Parseval regularisation for convolutional layers

        # The dimensions of the weight matrix are
        # trainable_weights[0] = (filter_d1,filter_d2,input_channels,output_channels) This is the kernel.
        # trainable_weights[1] = (output_channels) This is the bias.

        beta_param = self.beta
        conv_layer = self.model.get_layer(self.layername)
        weights = conv_layer.get_weights()[0]
        bias = conv_layer.get_weights()[1]

        weights_swapped = np.transpose(weights,[0,1,3,2])

        dim1 = weights_swapped.shape[0]
        dim2 = weights_swapped.shape[1]
        dim3 = weights_swapped.shape[2]
        dim4 = weights_swapped.shape[3]

        vector_weights = np.reshape(weights_swapped,(-1,weights_swapped.shape[3]))
        vector_weights = (1+beta_param)*vector_weights-beta_param*vector_weights@vector_weights.T@vector_weights

        """
        print('\n')
        print('The spectral norm of the unfolded weight matrix is: ', np.linalg.norm(vector_weights,2))

        print('\n')
        print('The Frobenius norm of the unfolded weight matrix is: ', np.linalg.norm(vector_weights,'fro'))
        """

        weights_swapped = np.reshape(vector_weights,(dim1,dim2,dim3,dim4))
        weights = np.transpose(weights_swapped,[0,1,3,2])

        conv_layer.set_weights([weights,bias])

class denseParsevalReg(Callback):
    '''
    This is the parseval regularization class for a dense layer.
    '''
    def __init__(self,layer_name,beta_par,subsampling):
        self.validation_data = None
        self.model = None
        self.layername = layer_name
        self.beta = beta_par
        self.sub = subsampling

    def on_batch_end(self, batch, logs={}):

        #This is an implementation of Parseval regularisation for dense layers

        beta_param = self.beta
        subsampling_size = self.sub
        dense_layer = self.model.get_layer(self.layername)
        dense_weights = dense_layer.get_weights()[0]
        bias = dense_layer.get_weights()[1]

        rand_rows = np.random.permutation(dense_weights.shape[0])

        dense_weights_subsampled = dense_weights[rand_rows[0:subsampling_size],:]

        dense_weights_subsampled = (1+beta_param)*dense_weights_subsampled-beta_param*dense_weights_subsampled@dense_weights_subsampled.T@dense_weights_subsampled

        dense_weights[rand_rows[0:subsampling_size], :] = dense_weights_subsampled

        dense_layer.set_weights([dense_weights,bias])

class localParsevalReg(Callback):
    '''
    This is the parseval regularization class for a locally connected layer.
    '''
    def __init__(self,layer_name,beta_par):
        self.validation_data = None
        self.model = None
        self.layername = layer_name
        self.beta = beta_par

    def on_batch_end(self, batch, logs={}):

        #This is an implementation of Parseval regularisation for convolutional layers

        # The dimensions of the weight matrix are
        # trainable_weights[0] = (filter_d1,filter_d2,input_channels,output_channels) This is the kernel.
        # trainable_weights[1] = (output_channels) This is the bias.

        beta_param = self.beta
        local_layer = self.model.get_layer(self.layername)
        weights = local_layer.get_weights()[0]
        bias = local_layer.get_weights()[1]

        dim1 = weights.shape[0]
        dim2 = weights.shape[1]
        dim3 = weights.shape[2]

        for i in range(0,dim1):
            vector_weights = weights[i,:,:]
            vector_weights = (1+beta_param)*vector_weights-beta_param*vector_weights@vector_weights.T@vector_weights
            weights[i, :, :] = vector_weights


        #print('\n')
        #print('The spectral norm of the unfolded weight matrix is: ', np.linalg.norm(vector_weights,2))


        local_layer.set_weights([weights,bias])
