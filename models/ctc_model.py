import tensorflow as tf
from cnn_models.alexnet import AlexNet
from cnn_models.alexnet import AlexNet
from models.modules.rnn_seq2seq import Encoder, Decoder
from utils.vocab_utils import EOS_ID, SOS_ID
import tensorflow_addons as tfa


class CTCModel(tf.keras.Model):
    def __init__(self, input_shape, dropout, rnn_units, tgt_vocab_size):
        super(CTCModel, self).__init__()
        self.cnn_model = AlexNet(
            dropout_rate=dropout,
            weights_path="/home/panxie/Documents/sign-language/nslt/BaseModel/bvlc_alexnet.npy")
        self.cnn_model.build((None,) + input_shape + (3,))
        self.cnn_model.load_weights()
        self.rnn = tf.keras.layers.LSTM(units=rnn_units,
                                        return_sequences=True,
                                        return_state=False)
        self.Dense = tf.keras.layers.Dense(units=tgt_vocab_size)

    def call(self, inputs, training=None, mask=None):
        if training:
            src_inputs, tgt_input_ids, tgt_output_ids, src_len, tgt_len = inputs
        else:
            src_inputs, tgt_input_ids, src_len, tgt_len = inputs
        bs, num_frames, h, w, c = src_inputs.shape
        src_inputs = tf.reshape(src_inputs, (bs * num_frames, h, w, c))
        cnn_output = self.cnn_model(src_inputs, training=training)  # [batch, compressed_frames, 512]
        cnn_output = tf.reshape(cnn_output, (bs, num_frames, -1))
        rnn_output = self.rnn(cnn_output)
        rnn_output = tf.transpose(rnn_output, (1, 0, 2)) # [num_frames, batch, rn_units]
        logits = self.Dense(rnn_output)  # [num_frames, batch, tgt_vocab_size]
        if training:
            loss = tf.nn.ctc_loss(labels=tgt_input_ids,
                                  label_length=tgt_len,
                                  logits=logits,
                                  logit_length=src_len,
                                  logits_time_major=True,
                                  blank_index=-1)
            return tf.math.reduce_mean(loss)
        else:
            decoded, decoded_prob = tf.nn.ctc_beam_search_decoder(inputs=logits,
                                                                  sequence_length=src_len,
                                                                  beam_width=10)
            return tf.sparse.to_dense(decoded[0])


if __name__ == "__main__":

    import os
    from utils.vocab_utils import create_tgt_vocab_table, check_vocab, UNK
    from utils.dataset import get_train_dataset

    print(tf.__version__)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
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

    base_path = "/home/panxie/Documents/sign-language/nslt/Data"
    src_file = base_path + "/phoenix2014T.test.sign"
    tgt_file = base_path + "/phoenix2014T.test.gloss"
    tgt_vocab_file = base_path + "/phoenix2014T.vocab.gloss"

    tgt_vocab_size, tgt_vocab_file = check_vocab(tgt_vocab_file,
                                                 "./",
                                                 pad="<pad>",
                                                 sos="<s>",
                                                 eos="</s>",
                                                 unk=UNK)
    model = CTCModel(input_shape=(227, 227), dropout=0.2, tgt_vocab_size=tgt_vocab_size)

    # print(os.getcwd())

    tgt_vocab_table = create_tgt_vocab_table(base_path + "/phoenix2014T.vocab.gloss")
    dataset = get_train_dataset(src_file, tgt_file, tgt_vocab_table)
    cnt = 0
    for data in dataset.take(100):
        decoded, decoded_prob = model(data, training=False)
        print(tf.sparse.to_dense(decoded[0]).shape, decoded_prob.shape)
        print("\n")