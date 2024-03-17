import os
import pickle

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense, Reshape, Conv2DTranspose, Activation, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import tensorflow as tf


tf.compat.v1.disable_eager_execution()


class VAE:
    """
    VAE represents a Deep Convolutional variational autoencoder architecture with
    mirrored encoder and decoder components.
    """
 
    def __init__(self,
                 input_shape,
                 conv_filters, 
                 conv_kernels,
                 conv_strides,
                 latent_space_dim):
        self.input_shape = input_shape # [28, 28, 1] [width, height, num_channels] -> 1 channel for grey scale images.
        self.conv_filters = conv_filters # [2, 4, 8] 1st conv layer, 2 filters, 2nd conv layer 4 filters etc.
        self.conv_kernels = conv_kernels # [3, 5, 3] 1st conv layer, 3x3 kernel size, 2nd 5x5 etc.
        self.conv_strides = conv_strides # [1, 2, 2] 1st conv layer, stride = 1, then = 2. Stride 2 = downsampling the data
        self.latent_space_dim = latent_space_dim # 2 (integer, bottleneck will have 2 dimensions for example)

        self.encoder = None
        self.decoder = None
        self.model = None
        
        # private attributes
        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        # _build method to instanciate an Autoencoder model
        self._build()

    def summary(self):
        """Prints on console architecture information"""
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        mse_loss = MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=mse_loss) # compile method native in keras api

    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(x_train, # input is training set
                       x_train, # output is input (expect data reconstruction)
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=True)
        
    def save(self, save_folder="."):
        """Save (default is working dir ".")
        1. Ensures save folder exists, if not creates it.
        2. Saves params
        3. Saves weights
        """
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder) # constructor-passed params (need to store to recreate)
        self._save_weights(save_folder) # trained weights also need to be stored

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path) # .load_weights is keras model method

    def reconstruct(self, images):
        """Method reconstructs images it is passed."""
        # Use encoder to create latent space representation of the images
        # Images -> Encoder -> Latent Representations
        latent_representations =  self.encoder.predict(images) 
        # Input latent space representations into the decoder to reconstruct images
        # Latent representations -> Decoder -> Reconstructed Images
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations

    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = VAE(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder

    def _create_folder_if_it_doesnt_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5") # h5 is a keras format for storing weights
        self.model.save_weights(save_path)

    def _build(self):
        """
        3 steps: build encoder, build decoder, build autoencoder
        to build all parts on the architecture.
        """
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_autoencoder(self):
        model_input = self._model_input
        # apply encoder to input, then decoder to output of encoder
        # this is output of entire autoencoder
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="autoencoder")

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer) # reshape dense layer into 3d array
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name="decoder_input")
    
    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck) # [1, 2, 4] -> want flattened version of this -> 8 (np.prod)
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input) # apply dense layer to decoder input
        return dense_layer
    
    def _add_reshape_layer(self, dense_layer):
        """Go back to 3D array from flattened layer"""
        reshape_layer = Reshape(self._shape_before_bottleneck)(dense_layer) # pass in target shape
        return reshape_layer
    
    def _add_conv_transpose_layers(self, x): # pass generic graph of layers, 'x'
        """Add conv transpose blocks."""
        # loop through all conv layers in reverse order and stop at the first layer
        # don't want to apply conv transpose block to first layer
        for layer_index in reversed(range(1, self._num_conv_layers)):
            # [0, 1, 2] say we have 3 conv layers
            # want the reverse: [2, 1, 0], also want to drop 0th (first conv layer)
            # [1, 2] -> [2, 1] is new loop
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    def _add_conv_transpose_layer(self, layer_index, x):
            layer_num = self._num_conv_layers - layer_index
            conv_transpose_layer = Conv2DTranspose(
                filters=self.conv_filters[layer_index],
                kernel_size=self.conv_kernels[layer_index],
                strides=self.conv_strides[layer_index],
                padding="same",
                name=f"decoder_conv_transpose_layer_{layer_num}"
            )
            x = conv_transpose_layer(x)
            x = ReLU(name=f"decoder_relu_{layer_num}")(x)
            x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
            return x
    
    def _add_decoder_output(self, x):
        """Add final conv transpose layer, but don't want relu & bn
        Instead sigmoid activation function"""
        conv_transpose_layer = Conv2DTranspose(
                filters=1, # [24, 24, 1] [height, width, channel]
                kernel_size=self.conv_kernels[0], # get first conv input shape kernel nums
                strides=self.conv_strides[0],
                padding="same",
                name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
            )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder") # input, output, name

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")
    
    def _add_conv_layers(self, encoder_input):
        """Creates all convolutional blocks in encoder."""
        x = encoder_input
        # go through all conv layers, and add to graph/layers/network each conv layer
        for layer_index in range(self._num_conv_layers):
            # defer actual addition of new conv block to a method
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        """Adds a convolutional block to a graph of layers, consisting of
        conv 2d + ReLU + batch normalization.
        """
        layer_number = layer_index + 1
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_number}"
        )  
        # apply keras conv layer to incoming graph of layers, x
        x = conv_layer(x)
        x = ReLU(name=f"encoder_relu_{layer_number}")(x) # instantiate ReLU, and apply it to x
        x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        return x
    
    def _add_bottleneck(self, x):
        """Flatten data and add bottleneck with Gaussian Sampling (Dense Layer)."""
        # need to store info on shape of data before it is flattened
        # for decoder, we must mirror the process, i.e go from flatten -> 3d arrays
        self._shape_before_bottleneck = K.int_shape(x)[1:] # [2, 7, 7, 32] 4d array, [batch size, width, height, num_channels]
        # batch size dimension ignored, [1:]

        x = Flatten()(x) # flattened data
        self.mu = Dense(self.latent_space_dim, name="mu")(x) # represents mean vector
        self.log_variance = Dense(self.latent_space_dim,
                                   name="log_variance")(x) # represents log variance vector
        # no longer a sequential graph: i.e we take the graph built so far and
        # branch it out into two dense layers x-->mu & x-->log_variance
        # (two separate dense layers have been applied to the graph x)

        def sample_point_from_normal_distribution(args):
            "z = mu + sigma*epsilon (sigma = e^(log_variance/2))"
            mu, log_variance = args
            # sample a point from the random normal dist.
            epsilon = K.random_normal(shape=K.shape(self.mu), mean=0.,
                                       stddev=1.)
            sampled_point = mu + K.exp(log_variance / 2) * epsilon
            return sampled_point

        # next: sample a data point from the gaussian distribution that is parameterised by mu and log_variance dense layers
        # this can be done using keras' lambda layer: this layer can wrap functions within our graph
        x = Lambda(sample_point_from_normal_distribution,
                    name="encoder_output")([self.mu, self.log_variance])

        return x

#############################################################################

class Autoencoder:
    """
    Autoencoder represents a Deep Convolutional autoencoder architecture with
    mirrored encoder and decoder components.
    """
 
    def __init__(self,
                 input_shape,
                 conv_filters, 
                 conv_kernels,
                 conv_strides,
                 latent_space_dim):
        self.input_shape = input_shape # [28, 28, 1] [width, height, num_channels] -> 1 channel for grey scale images.
        self.conv_filters = conv_filters # [2, 4, 8] 1st conv layer, 2 filters, 2nd conv layer 4 filters etc.
        self.conv_kernels = conv_kernels # [3, 5, 3] 1st conv layer, 3x3 kernel size, 2nd 5x5 etc.
        self.conv_strides = conv_strides # [1, 2, 2] 1st conv layer, stride = 1, then = 2. Stride 2 = downsampling the data
        self.latent_space_dim = latent_space_dim # 2 (integer, bottleneck will have 2 dimensions for example)

        self.encoder = None
        self.decoder = None
        self.model = None
        
        # private attributes
        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        # _build method to instanciate an Autoencoder model
        self._build()

    def summary(self):
        """Prints on console architecture information"""
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        mse_loss = MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=mse_loss) # compile method native in keras api

    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(x_train, # input is training set
                       x_train, # output is input (expect data reconstruction)
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=True)
        
    def save(self, save_folder="."):
        """Save (default is working dir ".")
        1. Ensures save folder exists, if not creates it.
        2. Saves params
        3. Saves weights
        """
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder) # constructor-passed params (need to store to recreate)
        self._save_weights(save_folder) # trained weights also need to be stored

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path) # .load_weights is keras model method

    def reconstruct(self, images):
        """Method reconstructs images it is passed."""
        # Use encoder to create latent space representation of the images
        # Images -> Encoder -> Latent Representations
        latent_representations =  self.encoder.predict(images) 
        # Input latent space representations into the decoder to reconstruct images
        # Latent representations -> Decoder -> Reconstructed Images
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations

    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = Autoencoder(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder

    def _create_folder_if_it_doesnt_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5") # h5 is a keras format for storing weights
        self.model.save_weights(save_path)

    def _build(self):
        """
        3 steps: build encoder, build decoder, build autoencoder
        to build all parts on the architecture.
        """
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_autoencoder(self):
        model_input = self._model_input
        # apply encoder to input, then decoder to output of encoder
        # this is output of entire autoencoder
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="autoencoder")

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer) # reshape dense layer into 3d array
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name="decoder_input")
    
    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck) # [1, 2, 4] -> want flattened version of this -> 8 (np.prod)
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input) # apply dense layer to decoder input
        return dense_layer
    
    def _add_reshape_layer(self, dense_layer):
        """Go back to 3D array from flattened layer"""
        reshape_layer = Reshape(self._shape_before_bottleneck)(dense_layer) # pass in target shape
        return reshape_layer
    
    def _add_conv_transpose_layers(self, x): # pass generic graph of layers, 'x'
        """Add conv transpose blocks."""
        # loop through all conv layers in reverse order and stop at the first layer
        # don't want to apply conv transpose block to first layer
        for layer_index in reversed(range(1, self._num_conv_layers)):
            # [0, 1, 2] say we have 3 conv layers
            # want the reverse: [2, 1, 0], also want to drop 0th (first conv layer)
            # [1, 2] -> [2, 1] is new loop
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    def _add_conv_transpose_layer(self, layer_index, x):
            layer_num = self._num_conv_layers - layer_index
            conv_transpose_layer = Conv2DTranspose(
                filters=self.conv_filters[layer_index],
                kernel_size=self.conv_kernels[layer_index],
                strides=self.conv_strides[layer_index],
                padding="same",
                name=f"decoder_conv_transpose_layer_{layer_num}"
            )
            x = conv_transpose_layer(x)
            x = ReLU(name=f"decoder_relu_{layer_num}")(x)
            x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
            return x
    
    def _add_decoder_output(self, x):
        """Add final conv transpose layer, but don't want relu & bn
        Instead sigmoid activation function"""
        conv_transpose_layer = Conv2DTranspose(
                filters=1, # [24, 24, 1] [height, width, channel]
                kernel_size=self.conv_kernels[0], # get first conv input shape kernel nums
                strides=self.conv_strides[0],
                padding="same",
                name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
            )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder") # input, output, name

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")
    
    def _add_conv_layers(self, encoder_input):
        """Creates all convolutional blocks in encoder."""
        x = encoder_input
        # go through all conv layers, and add to graph/layers/network each conv layer
        for layer_index in range(self._num_conv_layers):
            # defer actual addition of new conv block to a method
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        """Adds a convolutional block to a graph of layers, consisting of
        conv 2d + ReLU + batch normalization.
        """
        layer_number = layer_index + 1
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_number}"
        )  
        # apply keras conv layer to incoming graph of layers, x
        x = conv_layer(x)
        x = ReLU(name=f"encoder_relu_{layer_number}")(x) # instantiate ReLU, and apply it to x
        x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        return x
    
    def _add_bottleneck(self, x):
        """Flatten data and add bottleneck (Dense Layer)."""
        # need to store info on shape of data before it is flattened
        # for decoder, we must mirror the process, i.e go from flatten -> 3d arrays
        self._shape_before_bottleneck = K.int_shape(x)[1:] # [2, 7, 7, 32] 4d array, [batch size, width, height, num_channels]
        # batch size dimension ignored, [1:]

        x = Flatten()(x) # flattened data
        x = Dense(self.latent_space_dim, name="encoder_output")(x)
        return x
    
#############################################################################

if __name__ == "__main__":
    autoencoder = VAE(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
        )
    autoencoder.summary()




