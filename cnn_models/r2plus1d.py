import tensorflow as tf
import math
from cnn_models.modules.SpatioTemporalModule import SpatioTemporalResLayer, SpatioTemporalResBlock, SpatioTemporalConv

class R2Plus1DNet(tf.keras.Model):
    r"""Forms the overall ResNet feature extractor by initializng 5 layers, with the number of blocks in
    each layer set by layer_sizes, and by performing a global average pool at the end producing a
    512-dimensional vector for each element in the batch.

        Args:
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
        """

    def __init__(self, res_n, block_type=SpatioTemporalResBlock):
        block_num = {18: (2, 2, 2, 2),
                     34: (3, 4, 6, 3),
                     50: (3, 4, 6, 3),
                     101: (3, 4, 23, 3),
                     152: (3, 4, 36, 3)}
        layer_sizes = block_num[res_n]
        super(R2Plus1DNet, self).__init__()
        # first conv, with stride 1x2x2 and kernel size 3x7x7
        self.conv1 = SpatioTemporalConv(3, 64, kernel_size=(7, 7, 7), stride=[1, 2, 2], padding="same")
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.conv2 = SpatioTemporalResLayer(64, 64, 3, layer_sizes[0], block_type=block_type, downsample=False)
        # each of the final three layers doubles num_channels, while performing downsampling
        # inside the first block
        self.conv3 = SpatioTemporalResLayer(64, 128, 3, layer_sizes[1], block_type=block_type, downsample=True)
        self.conv4 = SpatioTemporalResLayer(128, 256, 3, layer_sizes[2], block_type=block_type, downsample=True)
        self.conv5 = SpatioTemporalResLayer(256, 512, 3, layer_sizes[3], block_type=block_type, downsample=True)

        self.pool = tf.keras.layers.GlobalAvgPool3D(data_format="channels_last")

    def call(self, x, training=True):
        print("inputs: out_shape = ", x.shape)
        x = self.conv1(x)
        print("conv1: out_shape = ", x.shape)
        x = self.conv2(x)
        print("conv2: out_shape = ", x.shape)
        x = self.conv3(x)
        print("conv3: out_shape = ", x.shape)
        x = self.conv4(x)
        print("conv4: out_shape = ", x.shape)
        x = self.conv5(x)
        print("conv5: out_shape = ", x.shape)

        x = self.pool(x)
        print("pooling: out_shape = ", x.shape)

        return tf.reshape(x, (-1, 512))

if __name__ == "__main__":
    model = R2Plus1DNet(res_n=18)
    model.build(input_shape=(None, 200, 224, 224, 3))
    # model.summary()
    for param in model.trainable_variables:
        print(param.name, param.shape)


