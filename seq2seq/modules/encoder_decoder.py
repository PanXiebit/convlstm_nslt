import tensorflow as tf
import tensorflow_addons as tfa


class Encoder(tf.keras.Model):
    def __init__(self, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        # self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        # self.init_hidden = tf.zeros((self.batch_sz, self.enc_units))

    def call(self, x):
        """

        :param x:  cnn output.
        :param hidden:
        :return:
        """
        # x = self.embedding(x)
        # output, state = self.gru(x, initial_state=self.init_hidden)
        output, state = self.gru(x)
        return output, state


class Decoder(tf.keras.Model):
    def __init__(self, emb_size, tgt_vocab_size, rnn_units):
        super(Decoder, self).__init__()
        self.rnn_units = rnn_units
        self.dec_embedding = tf.keras.layers.Embedding(input_dim=tgt_vocab_size, output_dim=emb_size)
        self.output_layer = tf.keras.layers.Dense(units=tgt_vocab_size)

        # rnn
        self.decoder_rnncell = tf.keras.layers.GRUCell(rnn_units)

        # sampler
        self.simpler = tfa.seq2seq.sampler.TrainingSampler()
        # Create attention mechanism with memory = None
        self.attention_mechanism = self.build_attention_mechanism(rnn_units, None, None)
        # AttentionWrapper RNNCell
        self.rnn_cell = self.build_rnn_cell()
        # dynamic decoder
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell,  # AttentionWrapper RNNCell
                                                sampler=self.simpler,  # sampler
                                                output_layer=self.output_layer)

    def build_attention_mechanism(self, unit, memory, memory_sequence_len):
        return tfa.seq2seq.BahdanauAttention(units=unit,
                                             memory=memory,
                                             memory_sequence_length=memory_sequence_len)

    def build_rnn_cell(self):
        rnn_cell = tfa.seq2seq.AttentionWrapper(cell=self.decoder_rnncell,   # rnn_cell
                                                attention_mechanism=self.attention_mechanism, # attention mechanism
                                                attention_layer_size=self.rnn_units)
        return rnn_cell

    def build_decoder_initial_state(self, batch_size, enc_state, Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_size,
                                                                dtype=Dtype)
        decoder_initial_state = decoder_initial_state.clone(cell_state=enc_state)
        return decoder_initial_state


if __name__ == "__main__":
    enc = Encoder(enc_units=15)
    x = tf.random.normal((5, 10, 32))
    enc_out, enc_state = enc(x)
    # print(enc_out.shape, enc_state.shape)

    dec = Decoder(emb_size=300, tgt_vocab_size=2000, rnn_units=15)
    dec.attention_mechanism.setup_memory(memory=enc_out)
    dec_init_state = dec.build_decoder_initial_state(5, enc_state, tf.float32)
    # exit()
    decoder_emb_inp = tf.random.normal((5, 10, 300))
    outputs, _, _ = dec.decoder(decoder_emb_inp, initial_state=dec_init_state,
                                sequence_length=5 * [8 - 1])
    print(outputs.rnn_output.shape)



