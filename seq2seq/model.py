import tensorflow as tf
from cnn_models.mcLSTM import Mclstm
from seq2seq.modules.encoder_decoder import Encoder, Decoder
from utils.vocab_utils import EOS_ID, SOS_ID
import tensorflow_addons as tfa


class Model(tf.keras.Model):
    def __init__(self, rnn_units, tgt_emb_size, tgt_vocab_size):
        super(Model, self).__init__()

        self.cnn_model = Mclstm(res_n=18, init_kernel=32)
        self.Encoder = Encoder(enc_units=rnn_units)
        self.Decoder = Decoder(emb_size=tgt_emb_size, tgt_vocab_size=tgt_vocab_size,
                               rnn_units=rnn_units)

    def call(self, inputs, training=None, mask=None):
        if training:
            src_inputs, tgt_input_ids, tgt_output_ids, src_len, tgt_len = inputs
            return self.decoder(src_inputs, tgt_input_ids, src_len, tgt_len)
        else:
            src_inputs, src_len = inputs
            return self.argmax_predict(src_inputs, src_len)

    def decoder(self, src_inputs, tgt_input_ids, src_len, tgt_len):
        bs = src_len.shape[0]
        cnn_output = self.cnn_model(src_inputs)  # [batch, compressed_frames, 512]
        # print("cnn_output shape: ", cnn_output.shape)

        enc_output, enc_state = self.Encoder(cnn_output)  # [batch, cp_frames, rnn_units], [batch, rnn_units]
        # print("enc_output shape:", enc_output.shape)

        # decoder embedding
        decoder_emb_inp = self.Decoder.dec_embedding(tgt_input_ids)  # [batch, tgt_len, embed_size]
        # print("decoder_emb_inp shape:", decoder_emb_inp.shape)

        # setting up decoder memory and initial state from enc_output and zero state for AttentionWrapperState
        self.Decoder.attention_mechanism.setup_memory(memory=enc_output,
                                                      memory_sequence_length=src_len)

        decoder_initial_state = self.Decoder.build_decoder_initial_state(batch_size=bs,
                                                                         enc_state=enc_state,
                                                                         Dtype=tf.float32)

        outputs, _, _ = self.Decoder.decoder(inputs=decoder_emb_inp,
                                             initial_state=decoder_initial_state,
                                             sequence_length=tgt_len)
        logits = outputs.rnn_output
        return logits

    def argmax_predict(self, src_inputs, src_len, max_tgt_len=50):
        bs = src_len.shape[0]
        cnn_output = self.cnn_model(src_inputs)  # [batch, compressed_frames, 512]
        # print("cnn_output shape: ", cnn_output.shape)

        enc_output, enc_state = self.Encoder(cnn_output)  # [batch, cp_frames, rnn_units], [batch, rnn_units]

        decoder_input = tf.expand_dims([EOS_ID] * bs, axis=1)  # [batch, 1]
        decoder_emb_inp = self.Decoder.dec_embedding(decoder_input)
        greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler(embedding_fn=self.Decoder.dec_embedding)
        greedy_decoder = tfa.seq2seq.BasicDecoder(cell=self.Decoder.rnn_cell,
                                                  sampler=greedy_sampler,
                                                  output_layer=self.Decoder.output_layer)
        self.Decoder.attention_mechanism.setup_memory(enc_output,
                                                      memory_sequence_length=src_len)

        decoder_initial_state = self.Decoder.build_decoder_initial_state(batch_size=bs,
                                                                         enc_state=enc_state,
                                                                         Dtype=tf.float32)
        maximum_iterations = max_tgt_len
        start_token = tf.fill([bs], SOS_ID)

        final_outputs, final_state, final_sequence_lengths = tfa.seq2seq.dynamic_decode(
            decoder=greedy_decoder,
            maximum_iterations=maximum_iterations
        )
