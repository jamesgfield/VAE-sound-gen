from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense
from tensorflow.keras import backend as K



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

        # _build method to instanciate an Autoencoder model
        self._build()

    def summary(self):
        """Prints on console architecture information"""
        self.encoder.summary()

    def _build(self):
        """
        3 steps: build encoder, build decoder, build autoencoder
        to build all parts on the architecture.
        """
        self._build_encoder()
        # self._build_decoder()
        # self._build_autoencoder()

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
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
    
if __name__ == "__main__":
    autoencoder = Autoencoder(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
        )
    autoencoder.summary()




