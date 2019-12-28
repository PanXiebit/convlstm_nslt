
import tensorflow as tf
import math

class SpatioTemporalConv(tf.keras.layers.Layer):
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
        super(SpatioTemporalConv, self).__init__()
        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 3
        if isinstance(stride, int):
            stride = [stride] * 3

        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid unintentional
        # behavior such as padding being added twice
        spatial_kernel_size = [1, kernel_size[1], kernel_size[2]]
        spatial_stride = [1, stride[1], stride[2]]


        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride = [stride[0], 1, 1]

        # compute the number of intermediary channels (M) using formula
        # from the paper section 3.5
        intermed_channels = int(
            math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels) / \
                       (kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

        # the spatial conv is effectively a 2D conv due to the
        # spatial_kernel_size, followed by batch_norm and ReLU
        self.spatial_conv = tf.keras.layers.Conv3D(filters=intermed_channels,
                                                   kernel_size=spatial_kernel_size,
                                                   strides=spatial_stride,
                                                   padding=padding,
                                                   use_bias=bias,
                                                   data_format="channels_last")
        self.bn = tf.keras.layers.BatchNormalization(axis=-1)   # 3D ???
        # self.bn = tf.nn.batch_normalization()
        self.relu = tf.keras.layers.ReLU()

        self.temporal_conv = tf.keras.layers.Conv3D(filters=out_channels,
                                                    kernel_size=temporal_kernel_size,
                                                    strides=temporal_stride,
                                                    padding=padding,
                                                    use_bias=bias,
                                                    data_format="channels_last")

    def call(self, x):
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.temporal_conv(x)  # no bn or relu later, in the residual block add bn_relu..
        return x


class SpatioTemporalResBlock(tf.keras.layers.Layer):
    """Single block for the ResNet network. Uses SpatioTemporalConv in
        the standard ResNet block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output produced by the block.
        kernel_size (int or tuple): Size of the convolving kernels.
        downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
    """

    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super(SpatioTemporalResBlock, self).__init__()

        # If downsample == True, the first conv of the layer has stride = 2
        # to halve the residual output size, and the input x is passed
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.downsample = downsample

        # to allow for SAME padding
        padding = "same"

        if self.downsample:
            # downsample with stride =2 the input x
            self.downsampleconv = SpatioTemporalConv(in_channels, out_channels, kernel_size=1, stride=2)
            self.downsamplebn = tf.keras.layers.BatchNormalization(axis=-1)

            # downsample with stride = 2when producing the residual
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, stride=2)
        else:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding)

        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu1 = tf.keras.layers.ReLU()

        # standard conv->batchnorm->ReLU
        self.conv2 = SpatioTemporalConv(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.outrelu = tf.keras.layers.ReLU()

    def call(self, x):
        res = self.relu1(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))

        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        return self.outrelu(x + res)


class SpatioTemporalResLayer(tf.keras.layers.Layer):
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

    def __init__(self, in_channels, out_channels, kernel_size, layer_size, block_type=SpatioTemporalResBlock,
                 downsample=False):
        super(SpatioTemporalResLayer, self).__init__()
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
