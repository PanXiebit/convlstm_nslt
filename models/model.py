import tensorflow as tf
from cnn_models.mcLSTM import Mclstm
from models.modules.rnn_seq2seq import Encoder, Decoder
from utils.vocab_utils import EOS_ID, SOS_ID
import tensorflow_addons as tfa


class Model(tf.keras.Model):
    def __init__(self, rnn_units, tgt_emb_size, tgt_vocab_size):
        super(Model, self).__init__()

        self.cnn_model = Mclstm(res_n=18, init_kernel=32)
        self.Encoder = Encoder(enc_units=rnn_units)
        self.Decoder = Decoder(emb_size=tgt_emb_size, tgt_vocab_size=tgt_vocab_size,
                               rnn_units=rnn_units)

    def call(self, inputs, beam_size=1, training=None, mask=None):
        if training:
            src_inputs, tgt_input_ids, tgt_output_ids, src_len, tgt_len = inputs
            return self.decoder(src_inputs, tgt_input_ids, src_len, tgt_len)
        else:
            self.beam_size = beam_size
            src_inputs, src_len = inputs
            return self.argmax_predict(src_inputs, src_len, self.beam_size)

    def decoder(self, src_inputs, tgt_input_ids, src_len, tgt_len):
        bs = src_len.shape[0]
        cnn_output, hidden_frame = self.cnn_model(src_inputs)  # [batch, compressed_frames, 512]
        # print("hidden_frame", hidden_frame)
        enc_output, enc_state = self.Encoder(cnn_output)  # [batch, cp_frames, rnn_units], [batch, rnn_units]

        # decoder embedding
        decoder_emb_inp = self.Decoder.dec_embedding(tgt_input_ids)  # [batch, tgt_len, embed_size]

        # setting up decoder memory and initial state from enc_output and zero state for AttentionWrapperState
        self.Decoder.attention_mechanism.setup_memory(memory=enc_output,
                                                      memory_sequence_length=src_len)

        decoder_initial_state = self.Decoder.build_decoder_initial_state(
            batch_size=bs,
            enc_state=enc_state,
            Dtype=tf.float32)

        outputs, _, _ = self.Decoder.decoder(
            inputs=decoder_emb_inp,  # BasicDecoder call() function, call dynamic_decoder function
            initial_state=decoder_initial_state,
            sequence_length=tgt_len)
        logits = outputs.rnn_output
        return logits

    def argmax_predict(self, src_inputs, src_len, beam_size):
        bs = src_inputs.shape[0]
        cnn_output, hidden_frame = self.cnn_model(src_inputs)  # [batch, compressed_frames, 512]
        # print("hidden_frame", hidden_frame)

        enc_output, enc_state = self.Encoder(cnn_output)  # [batch, cp_frames, rnn_units], [batch, rnn_units]

        if beam_size == 1:
            decoder_input = tf.expand_dims([SOS_ID] * bs, axis=1)  # [batch, 1]
            decoder_emb_inp = self.Decoder.dec_embedding(decoder_input)
            # greedy sample
            greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler(embedding_fn=self.Decoder.dec_embedding)
            # BasicDecoder in greedy sample, use the same rnn_cell with training
            maximum_iterations = hidden_frame + 10
            greedy_decoder_instance = tfa.seq2seq.BasicDecoder(
                cell=self.Decoder.rnn_cell,
                sampler=greedy_sampler,
                output_layer=self.Decoder.output_layer,
                maximum_iterations=maximum_iterations  # Must be defined here, otherwise it will generate infinitely
            )
            # BasicDecoder in greedy sample, use the same attention mechanism with training
            self.Decoder.attention_mechanism.setup_memory(enc_output,
                                                          memory_sequence_length=src_len)
            decoder_initial_state = self.Decoder.build_decoder_initial_state(batch_size=bs,
                                                                             enc_state=enc_state,
                                                                             Dtype=tf.float32)
            start_tokens = tf.fill([bs], SOS_ID)
            # the first writing
            outputs, _, _ = greedy_decoder_instance(  # BasicDecoder call() function, call dynamic function
                inputs=decoder_emb_inp,
                initial_state=decoder_initial_state,
                start_tokens=start_tokens,
                end_token=EOS_ID)
            return outputs.rnn_output

            # the second writing
            # (first_finished, first_inputs,first_state) = greedy_decoder_instance.initialize(
            #     self.Decoder.dec_embedding.embeddings,
            #     start_tokens=start_tokens,
            #     end_token=EOS_ID,
            #     initial_state=decoder_initial_state)
            #
            # inputs = first_inputs
            # state = first_state
            # predictions = np.empty((bs, 0), dtype=np.int32)
            # for j in range(maximum_iterations):
            #     outputs, next_state, next_inputs, finished = greedy_decoder_instance.step(j, inputs, state)
            #     inputs = next_inputs
            #     state = next_state
            #     outputs = np.expand_dims(outputs.sample_id, axis=-1)
            #     predictions = np.append(predictions, outputs, axis=-1)
            # return predictions

        # beam search
        else:
            start_tokens = tf.fill([bs], SOS_ID)
            end_token = EOS_ID
            decoder_input = tf.expand_dims([SOS_ID] * bs, 1)
            decoder_emb_inp = self.Decoder.dec_embedding(decoder_input)
            tile_memory = tfa.seq2seq.tile_batch(enc_output, beam_size)
            tile_length = tfa.seq2seq.tile_batch(src_len, beam_size)
            self.Decoder.attention_mechanism.setup_memory(memory=tile_memory,
                                                          memory_sequence_length=tile_length)
            tile_state = tfa.seq2seq.tile_batch(enc_state, beam_size)
            decoder_initial_state = self.Decoder.rnn_cell.get_initial_state(batch_size=bs * beam_size,
                                                                            dtype=tf.float32)
            decoder_initial_state = decoder_initial_state.clone(tile_state)
            beam_decoder_instance = tfa.seq2seq.BeamSearchDecoder(
                cell=self.Decoder.rnn_cell,
                beam_width=beam_size,
                embedding_fn=self.Decoder.dec_embedding.embeddings,
                output_layer=self.Decoder.output_layer,
                maximum_iterations=hidden_frame + 10
            )
            outputs, _, _ = beam_decoder_instance(  # BasicDecoder call() function, call dynamic function
                inputs=decoder_emb_inp,
                initial_state=decoder_initial_state,
                start_tokens=start_tokens,
                end_token=EOS_ID)


if __name__ == "__main__":
    src = tf.random.normal((5, 10, 32, 32, 3))
    src_len = [5, 6, 7, 8, 9]
    model = Model(rnn_units=64, tgt_vocab_size=2000, tgt_emb_size=300)
    out = model(inputs=(src, src_len), beam_size=1, training=False)
    print(out.shape)

    # for param in model.trainable_variables:
    #     print(param.name, param.shape)
    # tfa.models.dynamic_decode()
