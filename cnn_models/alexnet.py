import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
import numpy as np


class AlexNet(tf.keras.Model):
    def __init__(self, dropout_rate, weights_path):
        super(AlexNet, self).__init__()
        self.dropout_rate = dropout_rate
        self.weights_path = weights_path

        self.conv1 = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4),
                            padding="valid", name="conv1")
        # self.norm1 = tf.nn.local_response_normalization()
        self.pool1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                               padding="valid", name="pool1")

        self.conv2_1 = Conv2D(filters=int(256 / 2), kernel_size=(5, 5), strides=(1, 1),
                            padding="same", name="conv2_1")
        self.conv2_2 = Conv2D(filters=int(256 / 2), kernel_size=(5, 5), strides=(1, 1),
                              padding="same", name="conv2_2")

        self.pool2 = MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                               padding="valid", name="pool2")

        self.conv3 = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1),
                            padding="same", name="conv3")

        self.conv4_1 = Conv2D(filters=int(384 /2), kernel_size=(3, 3), strides=(1, 1),
                              padding="same", name="conv4_1")
        self.conv4_2 = Conv2D(filters=int(384 / 2), kernel_size=(3, 3), strides=(1, 1),
                              padding="same", name="conv4_2")

        self.conv5_1 = Conv2D(filters=int(256 /2), kernel_size=(3, 3), strides=(1, 1),
                              padding="same", name="conv5_1")
        self.conv5_2 = Conv2D(filters=int(256 / 2), kernel_size=(3, 3), strides=(1, 1),
                              padding="same", name="conv5_2")
        self.pool5 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding="valid", name="pool5")


        self.flat6 = Flatten()
        self.fc6 = Dense(units=4096, activation=tf.nn.relu, use_bias=True, name="fc6")
        self.dropout6 = Dropout(rate=self.dropout_rate, name="dropout6")

        self.fc7 = Dense(units=4096, activation=tf.nn.relu, use_bias=True, name="fc7")
        self.dropout7 =  Dropout(rate=self.dropout_rate, name="dropout7")

    def load_weights(self):
        weights_dict = np.load(self.weights_path, encoding='bytes').item()
        
        assert "conv1" in weights_dict and len(weights_dict["conv1"]) == 2
        self.conv1.set_weights(weights_dict["conv1"])

        conv2_weights = np.split(weights_dict["conv2"][0], indices_or_sections=2, axis=-1)   # [5, 5, 48, 128] * 2
        conv2_bias = np.split(weights_dict["conv2"][1], indices_or_sections=2, axis=-1)
        self.conv2_1.set_weights([conv2_weights[0], conv2_bias[0]])
        self.conv2_2.set_weights([conv2_weights[1], conv2_bias[1]])

        self.conv3.set_weights(weights_dict["conv3"])

        conv4_weights = np.split(weights_dict["conv4"][0], indices_or_sections=2, axis=-1)  # [3, 3, 192, 192] * 2
        conv4_bias = np.split(weights_dict["conv4"][1], indices_or_sections=2, axis=-1)
        self.conv4_1.set_weights([conv4_weights[0], conv4_bias[0]])
        self.conv4_2.set_weights([conv4_weights[1], conv4_bias[1]])

        conv5_weights = np.split(weights_dict["conv5"][0], indices_or_sections=2, axis=-1)  # [5, 5, 192, 128] * 2
        conv5_bias = np.split(weights_dict["conv5"][1], indices_or_sections=2, axis=-1)
        self.conv5_1.set_weights([conv5_weights[0], conv5_bias[0]])
        self.conv5_2.set_weights([conv5_weights[1], conv5_bias[1]])

        assert "fc6" in weights_dict and len(weights_dict["fc6"]) == 2
        self.fc6.set_weights(weights_dict["fc6"])

        assert "fc7" in weights_dict and len(weights_dict["fc7"]) == 2
        self.fc7.set_weights(weights_dict["fc7"])

        

    def call(self, x, training, mask=None):
        # conv1
        x = self.conv1(x)
        x = tf.nn.local_response_normalization(x, depth_radius=2, alpha=2e-05, beta=0.02,
                                               bias=1, name="norm1")
        x = self.pool1(x)
        # conv2
        x = tf.split(x, num_or_size_splits=2, axis=-1)
        x = tf.concat([self.conv2_1(x[0]), self.conv2_2(x[1])], axis=-1)
        x = tf.nn.local_response_normalization(x, depth_radius=2, alpha=2e-05, beta=0.02,
                                               bias=1, name="norm2")
        x = self.pool2(x)

        # conv3
        x = self.conv3(x)

        # conv4
        x = tf.split(x, num_or_size_splits=2, axis=-1)
        x = tf.concat([self.conv4_1(x[0]), self.conv4_2(x[1])], axis=-1)

        # conv5
        x = tf.split(x, num_or_size_splits=2, axis=-1)
        x = tf.concat([self.conv5_1(x[0]), self.conv5_2(x[1])], axis=-1)
        x = self.pool5(x)
        # fc6
        x = self.flat6(x)
        x = self.fc6(x)
        x = self.dropout6(x, training=training)

        # fc7
        x = self.fc7(x)
        x = self.dropout7(x, training=training)

        return x


if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    gpus = tf.config.experimental.list_physical_devices('GPU')
    # print(gpus)
    # exit()
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    # model.build((None, 224,224,3))
    # model.summary()
    # exit()
    model = AlexNet(keep_prob=0.2, weights_path="/home/panxie/Documents/sign-language/nslt/BaseModel/bvlc_alexnet.npy")
    model.build((None, 227, 227, 3))
    model.load_weights()
    for i in range(1000):
        inputs = tf.random.normal((300, 227, 227 ,3))
        out = model(inputs)
        print(out.shape)

