from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from cnn_models.mcLSTM import Mclstm

import abc

import tensorflow as tf
from models.modules import model_helper
import tensorflow_addons as tfa
from utils import misc_utils as utils


class BaseModel(tf.keras.Model):
    """Create the model.

        Args:
          hparams: Hyperparameter configurations.
          mode: TRAIN | EVAL | INFER
          iterator: Dataset Iterator that feeds data.
          target_vocab_table: Lookup table mapping target words to ids.
          reverse_target_vocab_table: Lookup table mapping ids to target words. Only
            required in INFER mode. Defaults to None.
          scope: scope of the model.
          single_cell_fn: allow for adding customized cell. When not specified,
            we default to model_helper._single_cell
    """
    def __init__(self, hparams, mode, iterator, target_vocab_table, reverse_target_vocab_table, single_cell_fn):
        super(BaseModel, self).__init__()
        assert isinstance(iterator, tf.data.Dataset)

        self.iterator = iterator
        self.mode = mode
        self.tgt_vocab_table = target_vocab_table

        self.tgt_vocab_size = hparams.tgt_vocab_size
        self.num_layers = hparams.num_layers
        self.num_gpus = hparams.num_gpus
        self.time_major = hparams.time_major

        if self.mode == "train":
            self.cnn_model = Mclstm(res_n=18)
        else:
            self.cnn_model = Mclstm(res_n=18)

        self.embedding_layer = tf.keras.layers.Embedding(input_dim=hparams.tgt_vocab_size,
                                                         output_dim=hparams.num_units,
                                                         embeddings_initializer="uniform")

        self.enc_layer = self._build_encoder(hparams)



    @abc.abstractmethod
    def _build_encoder(self, hparams):
        """Subclass must implement this.

        Build and run an RNN encoder.

        Args:
          hparams: Hyperparameters configurations.

        Returns:
          A tuple of encoder_outputs and encoder_state.
        """
        pass

    def _build_encoder_cell(self, hparams, num_layers, num_residual_layers):
        cell_list = []
        for i in range(num_layers):
            single_cell = model_helper._single_cell(unit_type=hparams.encoder_type,
                                                    num_units=hparams.num_units,
                                                    forget_bias=hparams.forget_bias,
                                                    dropout=hparams.dropout,
                                                    mode=self.mode,
                                                    residual_connection=(i >= num_layers - num_residual_layers),
                                                    device_str=model_helper.get_device_str(hparams.base_gpu)
                                                    )
            cell_list.append(single_cell)
        if len(cell_list) == 1:
            return cell_list[0]
        else:
            return tf.keras.layers.StackedRNNCells(cell_list)

    @abc.abstractmethod
    def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state, source_sequence_length):
        """Subclass must implement this.

        Args:
          hparams: Hyperparameters configurations.
          encoder_outputs: The outputs of encoder for every time step.
          encoder_state: The final state of the encoder.
          source_sequence_length: sequence length of encoder_outputs.

        Returns:
          A tuple of a multi-layer RNN cell used by decoder
            and the intial state of the decoder RNN.
        """
        pass

    def _build_decoder(self):
        pass



class Model(BaseModel):
    def _build_encoder(self, hparams):
        num_layers = hparams.num_layers
        num_residual_layers = hparams.num_residual_layers

        if hparams.encoder_type == "uni":
            enc_cell = self._build_encoder_cell(hparams, num_layers, num_residual_layers)

            enc_layer = tf.keras.layers.RNN(cell=enc_cell,
                                            return_sequences=True,
                                            return_state=True,
                                            time_major=False)
            return enc_layer

        elif hparams.encoder_type == "bi":

            num_bi_layers = int(num_layers / 2)
            num_bi_residual_layers = int(num_residual_layers / 2)
            utils.print_out(
                "  num_bi_layers = %d, num_bi_residual_layers=%d" % (num_bi_layers, num_bi_residual_layers))

            bi_enc_cell = self._build_encoder_cell(hparams, num_bi_layers, num_bi_residual_layers)
            # bw_cell = self._build_encoder_cell(hparams, num_bi_layers, num_bi_residual_layers)

            enc_layer = tf.keras.layers.RNN(cell=bi_enc_cell,
                                            return_sequences=True,
                                            return_state=True,
                                            time_major=False,
                                            go_backwards=True)
            return enc_layer