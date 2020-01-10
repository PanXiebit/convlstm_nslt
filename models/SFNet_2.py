# -*-encoding=""utf8 -*-

"""
[SF-Net](https://arxiv.org/abs/1908.01341)
"""

import tensorflow as tf
from cnn_models import resnet, alexnet
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, BatchNormalization, LayerNormalization
import math
from tensorflow.keras import backend as K

class SFNet(tf.keras.Model):
    def __init__(self, input_shape, cnn_arch, cnn_model_path, rnn_units, tgt_vocab_size, dropout=0.2):
        super(SFNet, self).__init__()
        # frame level
        self.tgt_vocab_size = tgt_vocab_size
        if cnn_arch == "resnet":
            self.cnn_model = resnet.ResNet(layer_num=18, include_top=True)
            self.cnn_model.build((None,) + input_shape + (3,))
            self.cnn_model.load_weights(cnn_model_path)
        elif cnn_arch == "alexnet":
            self.cnn_model = alexnet.AlexNet(dropout_rate=dropout)
            self.cnn_model.build((None,) + input_shape + (3,))
            self.cnn_model.load_weights(cnn_model_path)
        else:
            raise ValueError("CNN model architexture isn't existed!")

        # sentence level
        self.sent_rnn = Bidirectional(LSTM(units=rnn_units, return_sequences=True, return_state=False))
        self.sent_ln = LayerNormalization()
        self.sent_fc = Dense(units=tgt_vocab_size)

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs: [batch, num_frames, hight, width, channel]
        :param training:
        :param mask:
        :return:
        """
        if training:
            src_inputs, tgt_input_ids, tgt_output_ids, src_path, src_len, tgt_len = inputs
        else:
            src_inputs, tgt_input_ids, src_len, tgt_len = inputs
        bs, t, h, w, c = src_inputs.shape
        src_inputs = tf.reshape(src_inputs, (bs * t, h, w, c))
        cnn_out = self.cnn_model(src_inputs, training=training)
        # print("cnn_out: ", cnn_out.shape)
        cnn_out = tf.reshape(cnn_out, (bs, t, -1))  # [bs, t, K], K is the last channels of CNN


        sent_out = self.sent_rnn(cnn_out)  # [bs, F, 2H]
        sent_out = self.sent_ln(sent_out)
        logits = self.sent_fc(sent_out)  # [bs, F, tgt_vocab_size]
        if training:
            tgt_input = tgt_input_ids[:, 1:]
            tgt_len = tgt_len - 1               # remove eos token
            ctc_loss = tf.nn.ctc_loss(labels=tgt_input,
                                      logits=logits,
                                      label_length=tgt_len,
                                      logit_length=src_len,
                                      logits_time_major=False,
                                      blank_index=-1)
            ctc_loss = tf.reduce_mean(ctc_loss)
            reg_loss = 0
            return ctc_loss, reg_loss
        else:
            logits = tf.transpose(logits, (1, 0, 2))
            decoded, decoded_prob = tf.nn.ctc_beam_search_decoder(inputs=logits,
                                                                  sequence_length=src_len,
                                                                  beam_width=10,
                                                                  top_paths=10)
            return tf.sparse.to_dense(decoded[0]), decoded_prob


if __name__ == "__main__":
    import os

    gpus = tf.config.experimental.list_physical_devices('GPU')
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
    from utils.vocab_utils import create_tgt_vocab_table, check_vocab, UNK
    from utils.dataset import get_train_dataset, get_infer_dataset

    base_path = "/home/panxie/Documents/sign-language/nslt/Data"
    src_file = base_path + "/phoenix2014T.test.sign"
    tgt_file = base_path + "/phoenix2014T.test.gloss"
    tgt_vocab_file = base_path + "/phoenix2014T.vocab.gloss"
    # cnn_model_path = "/home/panxie/Documents/sign-language/nslt/BaseModel/ResNet_18.h5"
    cnn_model_path = "/home/panxie/Documents/sign-language/nslt/BaseModel/bvlc_alexnet.npy"
    tgt_vocab_size, tgt_vocab_file = check_vocab(tgt_vocab_file,
                                                 "./",
                                                 pad="<pad>",
                                                 sos="<s>",
                                                 eos="</s>",
                                                 unk=UNK)
    model = SFNet(input_shape=(227, 227), cnn_model_path=cnn_model_path, tgt_vocab_size=tgt_vocab_size,
                  rnn_units=256, cnn_arch="alexnet")
    tgt_vocab_table = create_tgt_vocab_table(base_path + "/phoenix2014T.vocab.gloss")
    dataset = get_infer_dataset(src_file, tgt_file, tgt_vocab_table)
    cnt = 0
    for data in dataset.take(100):
        loss = model(data, training=False)
        print(loss)
