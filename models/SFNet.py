import tensorflow as tf
from cnn_models import resnet, alexnet
from tensorflow.keras.layers import LSTM, Bidirectional, Dense
import math


class SFNet(tf.keras.Model):
    def __init__(self, input_shape, cnn_model_path, rnn_units, tgt_vocab_size, dropout=None):
        super(SFNet, self).__init__()
        # frame level
        # self.cnn_model = resnet.ResNet(layer_num=18, include_top=True)
        # self.cnn_model.build((None,) + input_shape + (3,))
        # self.cnn_model.load_weights(cnn_model_path)

        self.cnn_model = alexnet.AlexNet(dropout_rate=dropout,
                                         weights_path=cnn_model_path)
        self.cnn_model.build((None,) + input_shape + (3,))
        self.cnn_model.load_weights()

        # gloss level
        self.gloss_rnn = LSTM(units=rnn_units, return_sequences=False, return_state=True)
        self.gloss_fc = Dense(units=tgt_vocab_size)

        # sentence level
        self.sent_rnn = Bidirectional(LSTM(units=rnn_units, return_sequences=True, return_state=False))
        self.sent_fc = Dense(units=tgt_vocab_size)

        # loss
        self.reg_loss_func = tf.keras.losses.KLDivergence()

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

        # gloss level
        # framing step
        L = 12
        S = 3
        F = math.ceil((t - L) / S) + 1
        gloss_frame = []
        for i in range(0, F):
            meta_frame = cnn_out[:, i * S:i * S + L, :]  # [bs, <=L, K]
            meta_frame = tf.pad(meta_frame, [[0, 0], [0, L - meta_frame.shape[1]], [0, 0]])  # [bs, L, K]
            gloss_frame.append(tf.expand_dims(meta_frame, axis=1))
        gloss_frame = tf.concat(gloss_frame, axis=1)  # [bs, F, L, K]
        gloss_frame = tf.reshape(gloss_frame, (bs * F, L, -1))  # [bs*F, L, K]
        _, h_states, c_states = self.gloss_rnn(gloss_frame)  # [bs*F, H]
        gloss_out = tf.reshape(h_states, (bs, F, -1))  # [bs, F, H]
        gloss_fc = self.gloss_fc(gloss_out)  # for regularizer

        # sentence level
        sent_out = self.sent_rnn(gloss_out)  # [bs, F, 2H]
        logits = self.sent_fc(sent_out)  # [bs, F, tgt_vocab_size]
        logits_length = tf.constant([F] * bs, dtype=tf.int32)
        if training:
            tgt_input = tgt_input_ids[:, 1:]
            tgt_len = tgt_len - 1               # remove eos token
            if tf.reduce_max(tgt_len) > F:      # in CTC loss, source len must longer than target len
                tgt_input_ids = tgt_input[:, :F]
                cond = tf.cast(F >= tgt_len, tf.int32)
                tgt_len = tgt_len * cond + (1 - cond) * F
            ctc_loss = tf.nn.ctc_loss(labels=tgt_input_ids,
                                      logits=logits,
                                      label_length=tgt_len,
                                      logit_length=logits_length,
                                      blank_index=-1,
                                      logits_time_major=False)
            reg_loss = self.reg_loss_func(y_true=tf.reshape(logits, (bs * F, -1)),
                                          y_pred=tf.reshape(gloss_fc, (bs * F, -1)))
            ctc_loss = tf.reduce_mean(ctc_loss)
            reg_loss = tf.reduce_mean(reg_loss)
            return ctc_loss, reg_loss
        else:
            decoded, decoded_prob = tf.nn.ctc_beam_search_decoder(inputs=logits,
                                                                  sequence_length=src_len,
                                                                  beam_width=10)
            return tf.sparse.to_dense(decoded[0])


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
    from utils.dataset import get_train_dataset

    base_path = "/home/panxie/Documents/sign-language/nslt/Data"
    src_file = base_path + "/phoenix2014T.test.sign"
    tgt_file = base_path + "/phoenix2014T.test.gloss"
    tgt_vocab_file = base_path + "/phoenix2014T.vocab.gloss"
    cnn_model_path = "/home/panxie/Documents/sign-language/nslt/BaseModel/ResNet_18.h5"
    tgt_vocab_size, tgt_vocab_file = check_vocab(tgt_vocab_file,
                                                 "./",
                                                 pad="<pad>",
                                                 sos="<s>",
                                                 eos="</s>",
                                                 unk=UNK)
    model = SFNet(input_shape=(227, 227), cnn_model_path=cnn_model_path, tgt_vocab_size=tgt_vocab_size, rnn_units=256)
    tgt_vocab_table = create_tgt_vocab_table(base_path + "/phoenix2014T.vocab.gloss")
    dataset = get_train_dataset(src_file, tgt_file, tgt_vocab_table)
    cnt = 0
    for data in dataset.take(100):
        loss = model(data, training=True)
        print(loss)
