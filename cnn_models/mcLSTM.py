"""
reference:  https://github.com/irhum/R2Plus1D-PyTorch/blob/master/network.py
"""

import tensorflow as tf
from cnn_models.modules.SpatioTemporalModule import SpatioTemporalResLayer, SpatioTemporalResBlock, SpatioTemporalConv
from cnn_models.modules.SpatioModule import SpatioResLayer, SpatioResBlock

class Mclstm(tf.keras.Model):
    r"""Forms the overall ResNet feature extractor by initializng 5 layers, with the number of blocks in
    each layer set by layer_sizes, and by performing a global average pool at the end producing a
    512-dimensional vector for each element in the batch.

        Args:
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
        """

    def __init__(self, res_n, init_kernel=64, st_block_type=SpatioTemporalResBlock, sp_block_type=SpatioResBlock):
        block_num = {18: (2, 2, 2, 2),
                     34: (3, 4, 6, 3),
                     50: (3, 4, 6, 3),
                     101: (3, 4, 23, 3),
                     152: (3, 4, 36, 3)}
        layer_sizes = block_num[res_n]
        super(Mclstm, self).__init__()
        self.init_kernel = init_kernel
        # first conv, with stride 1x2x2 and kernel size 3x7x7
        self.st_conv1 = SpatioTemporalConv(3, init_kernel, kernel_size=(7, 7, 7), stride=[1, 2, 2], padding="same")
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        # self.st_conv2 = SpatioTemporalResLayer(init_kernel, init_kernel, 3, layer_sizes[0], block_type=st_block_type, downsample=False)  # resnet18 is False
        self.st_conv2 = SpatioTemporalResLayer(init_kernel, init_kernel, 3, layer_sizes[0], block_type=st_block_type, downsample=True)
        # each of the final three layers doubles num_channels, while performing downsampling
        # inside the first block
        self.st_conv3 = SpatioTemporalResLayer(init_kernel, init_kernel * 2, 3, layer_sizes[1], block_type=st_block_type, downsample=True)
        self.convlstm = tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=3, return_sequences=True, padding="same")
        self.sp_conv4 = SpatioResLayer(init_kernel * 2, init_kernel * 4, 3, layer_sizes[2], block_type=sp_block_type, downsample=True)
        self.sp_conv5 = SpatioResLayer(init_kernel * 4, init_kernel * 8, 3, layer_sizes[3], block_type=sp_block_type, downsample=True)

        self.pool = tf.keras.layers.GlobalAvgPool2D(data_format="channels_last")

    def call(self, x, training=True):
        # print("inputs: out_shape = ", x.shape)

        x = self.st_conv1(x)
        # print("conv1: out_shape = ", x.shape)

        x = self.st_conv2(x)
        # print("conv2: out_shape = ", x.shape)

        x = self.st_conv3(x)
        # print("conv3: out_shape = ", x.shape)

        x = self.convlstm(x)
        # print("convlstm: out_shape = ", x.shape)

        bs, t, h, w, c = x.shape
        x = tf.reshape(x, (-1, h, w, c))
        # print("reshape: out_shape = ", x.shape)

        x = self.sp_conv4(x)
        # print("conv4: out_shape = ", x.shape)

        x = self.sp_conv5(x)
        # print("conv5: out_shape = ", x.shape)

        x = self.pool(x)
        # print("pooling: out_shape = ", x.shape)

        return tf.reshape(x, (bs, t, self.init_kernel * 8)), t   # donnot support model.summary
        # return tf.reshape(x, (-1, self.init_kernel * 8)), t

if __name__ == "__main__":
    model = Mclstm(res_n=18)
    model.build(input_shape=(None, 50, 224, 224, 3))
    model.summary()



