
import tensorflow as tf
import math

class SpatioConv(tf.keras.layers.Layer):
    """
    Applies a factored 3D convolution over an input signal composed of several input
        planes with distinct spatial and time axes, by performing a 2D convolution over the
        spatial axes to an intermediate subspace, followed by a 1D convolution over the time
        axis to produce the final output.
        Args:
            in_channels (int): Number of channels in the input tensor
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
            bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding="same", bias=True):
        super(SpatioConv, self).__init__()
        # if ints are entered, convert them to iterables, 1 -> [1, 1]
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 2
        if isinstance(stride, int):
            stride = [stride] * 2

        # the spatial conv is effectively a 2D conv due to the
        # spatial_kernel_size, followed by batch_norm and ReLU
        self.spatial_conv = tf.keras.layers.Conv2D(filters=out_channels,
                                                   kernel_size=kernel_size,
                                                   strides=stride,
                                                   padding=padding,
                                                   use_bias=bias,
                                                   data_format="channels_last")

    def call(self, x):
        x = self.spatial_conv(x)
        return x


class SpatioResBlock(tf.keras.layers.Layer):
    """Single block for the ResNet network. Uses SpatioTemporalConv in
        the standard ResNet block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output produced by the block.
        kernel_size (int or tuple): Size of the convolving kernels.
        downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
    """

    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super(SpatioResBlock, self).__init__()

        # If downsample == True, the first conv of the layer has stride = 2
        # to halve the residual output size, and the input x is passed
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.downsample = downsample

        # to allow for SAME padding
        padding = "same"

        if self.downsample:
            # downsample with stride =2 the input x
            self.downsampleconv = SpatioConv(in_channels, out_channels, kernel_size=1, stride=2)
            self.downsamplebn = tf.keras.layers.BatchNormalization(axis=-1)

            # downsample with stride = 2when producing the residual
            self.conv1 = SpatioConv(in_channels, out_channels, kernel_size, padding=padding, stride=2)
        else:
            self.conv1 = SpatioConv(in_channels, out_channels, kernel_size, padding=padding)

        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu1 = tf.keras.layers.ReLU()

        # standard conv->batchnorm->ReLU
        self.conv2 = SpatioConv(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.outrelu = tf.keras.layers.ReLU()

    def call(self, x):
        res = self.relu1(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))

        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        return self.outrelu(x + res)


class SpatioResLayer(tf.keras.layers.Layer):
    r"""Forms a single layer of the ResNet network, with a number of repeating
    blocks of same output size stacked on top of each other

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the layer.
            kernel_size (int or tuple): Size of the convolving kernels.
            layer_size (int): Number of blocks to be stacked to form the layer
            block_type (Module, optional): Type of block that is to be used to form the layer. Default: SpatioTemporalResBlock.
            downsample (bool, optional): If ``True``, the first block in layer will implement downsampling. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, layer_size, block_type=SpatioResBlock,
                 downsample=False):
        super(SpatioResLayer, self).__init__()
        self.block1 = block_type(in_channels, out_channels, kernel_size, downsample)
        self.blocks = []
        for i in range(layer_size - 1):
            # all these blocks are identical, and have downsample = False by default
            self.blocks += [block_type(out_channels, out_channels, kernel_size)]

    def call(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)
        return x
