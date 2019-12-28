

import tensorflow as tf
import utils.misc_utils as utils
import tensorflow_addons as tfa

def get_initializer(init_op, seed=None, init_weight=None):
    """Create an initializer. init_weight is only for uniform."""
    if init_op == "uniform":
        assert init_weight
        return tf.random_uniform_initializer(-init_weight, init_weight, seed=seed)
    elif init_op == "glorot_normal":
        return tf.keras.initializers.glorot_normal(seed=seed)
    elif init_op == "glorot_uniform":
        return tf.keras.initializers.glorot_uniform(seed=seed)
    else:
        raise ValueError("Unknown init_op %s" % init_op)

def _single_cell(unit_type, num_units, forget_bias, dropout, mode, residual_connection=False, device_str=None):

    """Create an instance of a single RNN cell."""
    # dropout (= 1 - keep_prob) is set to 0 during eval and infer
    dropout = dropout if mode == "train" else 0.0

    # Cell Type
    if unit_type == "lstm":
        utils.print_out("  LSTM, forget_bias=%g" % forget_bias, new_line=False)
        single_cell = tf.keras.layers.LSTMCell(units=num_units, unit_forget_bias=forget_bias,
                                               dropout=dropout, recurrent_dropout=dropout)
    elif unit_type == "gru":
        utils.print_out("  GRU", new_line=False)
        single_cell = tf.keras.layers.GRUCell(units=num_units, dropout=dropout, recurrent_dropout=dropout)
    elif unit_type == "layer_norm_lstm":
        utils.print_out("  Layer Normalized LSTM, forget_bias=%g" % forget_bias, new_line=False)
        single_cell = tfa.rnn.LayerNormLSTMCell(units=num_units, unit_forget_bias=forget_bias,
                                                dropout=dropout, recurrent_dropout=dropout)
    else:
        raise ValueError("Unknown unit type %s!" % unit_type)

    # Dropout (= 1 - keep_prob)
    if dropout > 0.0:
        single_cell = tf.nn.RNNCellDropoutWrapper(cell=single_cell, input_keep_prob=(1.0 - dropout))
        utils.print_out("  %s, dropout=%g " % (type(single_cell).__name__, dropout), new_line=False)

    # Residual
    if residual_connection:
        single_cell = tf.nn.RNNCellResidualWrapper(single_cell)
        utils.print_out("  %s" % type(single_cell).__name__, new_line=False)

    # # Device Wrapper
    if device_str:
        single_cell = tf.nn.RNNCellDeviceWrapper(single_cell, device_str)
        utils.print_out("  %s, device=%s" % (type(single_cell).__name__, device_str), new_line=False)

    return single_cell


def get_device_str(device_id):
    """Return a device string for multi-GPU setup."""

    device_str_output = "/gpu:%d" % device_id
    return device_str_output