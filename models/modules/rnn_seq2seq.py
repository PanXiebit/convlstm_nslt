import tensorflow as tf
import tensorflow_addons as tfa


def single_cell_fn(enc_units, unit_type, residual, init_op, dropout, forget_bias=True):
    # Create multi-layer RNN cell.
    if unit_type == "lstm":
        single_cell = tf.keras.layers.LSTMCell(units=enc_units,
                                               kernel_initializer=init_op,
                                               unit_forget_bias=forget_bias)
        # unit_forget_bias, If True, add 1 to the bias of the forget gate at
        # initialization. Setting it to true will also force
        # `bias_initializer="zeros"`.

    elif unit_type == "gru":
        single_cell = tf.keras.layers.GRUCell(units=enc_units,
                                              kernel_initializer=init_op)
    else:
        raise ValueError("Unknown encoder_type %s" % unit_type)

    if dropout > 0.0:
        single_cell = tf.nn.RNNCellDropoutWrapper(cell=single_cell, input_keep_prob=(1.0 - dropout))

    if residual:
        single_cell = tf.nn.RNNCellResidualWrapper(single_cell)
    return single_cell


def create_rnn_cell(enc_units, unit_type, num_layers, residual, init_op, dropout, training,
                    forget_bias=True):
    dropout = dropout if training else 0.0

    if residual and num_layers > 1:
        if unit_type == "gnmt":
            num_residual_layers = num_layers - 2
        else:
            num_residual_layers = num_layers - 1
    else:
        num_residual_layers = 0

    cell_list = []
    for i in range(num_layers):
        single_cell = single_cell_fn(enc_units=enc_units,
                                     unit_type=unit_type,
                                     init_op=init_op,
                                     residual=(i >= num_layers - num_residual_layers),
                                     dropout=dropout,
                                     forget_bias=forget_bias)
        cell_list.append(single_cell)
    if num_layers == 1:
        cell_list = cell_list[0]
    return cell_list


class Encoder(tf.keras.Model):
    def __init__(self, enc_units, unit_type, num_layers, residual, init_op, dropout, training,
                 forget_bias=True):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.unit_type = unit_type
        self.num_layers = num_layers

        # build encoder rnn cell
        cells = create_rnn_cell(enc_units, unit_type, num_layers, residual, init_op, dropout, training,
                                forget_bias)

        self.enc_rnn = tf.keras.layers.RNN(cell=cells,
                                           return_sequences=True,
                                           return_state=True)

    def call(self, x):
        """Encoder"""
        output = self.enc_rnn(x)
        if self.unit_type == "gru":
            rnn_output, rnn_states = output[0], output[1]
        elif self.unit_type == "lstm":
            rnn_output, rnn_states = output[0], output[1]
        else:
            raise ValueError("Unknown rnn unit_type")
        return rnn_output, tuple([rnn_states] * self.num_layers)


class Decoder(tf.keras.Model):
    def __init__(self, emb_size, tgt_vocab_size, rnn_units, unit_type, num_layers, residual, init_op, dropout, training,
                 forget_bias=True):
        super(Decoder, self).__init__()
        self.rnn_units = rnn_units
        self.dec_embedding = tf.keras.layers.Embedding(input_dim=tgt_vocab_size, output_dim=emb_size)
        self.output_layer = tf.keras.layers.Dense(units=tgt_vocab_size)

        # rnn
        # self.decoder_rnncell = tf.keras.layers.GRUCell(rnn_units)
        cells = create_rnn_cell(rnn_units, unit_type, num_layers, residual, init_op, dropout, training,
                                forget_bias)
        self.decoder_rnncell = tf.keras.layers.StackedRNNCells(cells)

        # sampler
        self.simpler = tfa.seq2seq.sampler.TrainingSampler()
        # Create attention mechanism with memory = None
        self.attention_mechanism = self.build_attention_mechanism(rnn_units, None, None)
        # AttentionWrapper RNNCell
        self.attention_rnn_cell = self.build_attention_rnn_cell()

        # dynamic decoder
        self.decoder = tfa.seq2seq.BasicDecoder(self.attention_rnn_cell,  # AttentionWrapper RNNCell
                                                sampler=self.simpler,  # sampler
                                                output_layer=self.output_layer)

    def build_attention_mechanism(self, unit, memory, memory_sequence_len):
        return tfa.seq2seq.BahdanauAttention(units=unit,
                                             memory=memory,
                                             memory_sequence_length=memory_sequence_len)

    def build_attention_rnn_cell(self):
        attention_rnn_cell = tfa.seq2seq.AttentionWrapper(
            cell=self.decoder_rnncell,  # rnn_cell
            attention_mechanism=self.attention_mechanism,  # attention mechanism
            attention_layer_size=self.rnn_units)
        return attention_rnn_cell

    def build_decoder_initial_state(self, batch_size, enc_state, Dtype):
        decoder_initial_state = self.attention_rnn_cell.get_initial_state(batch_size=batch_size,
                                                                          dtype=Dtype)
        decoder_initial_state = decoder_initial_state.clone(cell_state=enc_state)
        return decoder_initial_state


if __name__ == "__main__":
    enc = Encoder(enc_units=15, unit_type="lstm", num_layers=2, residual=True, init_op="glorot_normal", dropout=0.2,
                  training=True, forget_bias=True)
    x = tf.random.normal((5, 4, 4))
    enc_out, enc_state = enc(x)
    # exit()
    dec = Decoder(emb_size=300, tgt_vocab_size=2000, rnn_units=15, unit_type="lstm", num_layers=2, residual=True,
                  init_op="glorot_normal", dropout=0.2,
                  training=True, forget_bias=True)
    dec.attention_mechanism.setup_memory(memory=enc_out)
    dec_init_state = dec.build_decoder_initial_state(5, enc_state, tf.float32)
    # exit()
    decoder_emb_inp = tf.random.normal((5, 10, 300))
    outputs, _, _ = dec.decoder(decoder_emb_inp, initial_state=dec_init_state,
                                sequence_length=5 * [8 - 1])
    print(outputs.rnn_output.shape)
