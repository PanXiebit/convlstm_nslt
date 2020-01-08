import tensorflow as tf
import math
from utils.vocab_utils import EOS_ID
import collections
from six.moves import xrange
import numpy as np


def _pad_tensors_to_same_length(x, y):
    """Pad x and y so that the results have the same length (second dimension)."""
    with tf.name_scope("pad_to_same_length"):
        x_length = tf.shape(x)[1]
        y_length = tf.shape(y)[1]

        max_length = tf.maximum(x_length, y_length)
        if tf.rank(x) == 3:
            x = tf.pad(x, [[0, 0], [0, max_length - x_length], [0, 0]])
        else:
            x = tf.pad(x, [[0, 0], [0, max_length - x_length]])
        y = tf.pad(y, [[0, 0], [0, max_length - y_length]])
        return x, y


def padded_cross_entropy_loss(logits, labels, smoothing, vocab_size):
    """Calculate cross entropy loss while ignoring padding.
    Args:
      logits: Tensor of size [batch_size, length_logits, vocab_size]
      labels: Tensor of size [batch_size, length_labels]
      smoothing: Label smoothing constant, used to determine the on and off values
      vocab_size: int size of the vocabulary
    Returns:
      Returns the cross entropy loss and weight tensors: float32 tensors with
        shape [batch_size, max(length_logits, length_labels)]
    """
    with tf.name_scope("loss"):
        logits, labels = _pad_tensors_to_same_length(logits, labels)

        # Calculate smoothing cross entropy
        with tf.name_scope("smoothing_cross_entropy"):
            confidence = 1.0 - smoothing
            low_confidence = (1.0 - confidence) / tf.cast(vocab_size - 1, tf.float32)
            soft_targets = tf.one_hot(
                tf.cast(labels, tf.int32),
                depth=vocab_size,
                on_value=confidence,
                off_value=low_confidence)
            xentropy = tf.keras.losses.categorical_crossentropy(
                y_pred=logits, y_true=soft_targets, from_logits=True)

            # Calculate the best (lowest) possible value of cross entropy, and
            # subtract from the cross entropy loss.
            normalizing_constant = -(
                    confidence * tf.math.log(confidence) + tf.cast(vocab_size - 1, tf.float32) *
                    low_confidence * tf.math.log(low_confidence + 1e-20))
            xentropy -= normalizing_constant

        weights = tf.cast(tf.not_equal(labels, 0), tf.float32)
        return xentropy * weights, weights


def bleu_score(logits, labels):
    """Approximate BLEU score computation between labels and predictions.
    An approximate BLEU scoring method since we do not glue word pieces or
    decode the ids and tokenize the output. By default, we use ngram order of 4
    and use brevity penalty. Also, this does not have beam search.
    Args:
      logits: Tensor of size [batch_size, length_logits, vocab_size]
      labels: Tensor of size [batch-size, length_labels]
    Returns:
      bleu: int, approx bleu score
    """
    if tf.rank(logits) == 3:
        predictions = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
    else:
        predictions = logits
    # TODO: Look into removing use of py_func
    # bleu = tf.py_function(_compute_bleu, (labels, predictions), tf.float32)
    bleu = _compute_bleu(labels, predictions)
    return bleu

def _compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 use_bp=True):
    """Computes BLEU score of translated segments against one or more references.
    Args:
      reference_corpus: list of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      use_bp: boolean, whether to apply brevity penalty.
    Returns:
      BLEU score.
    """
    reference_length = 0
    translation_length = 0
    bp = 1.0
    geo_mean = 0

    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order

    for (references, translations) in zip(reference_corpus, translation_corpus):
        references = _trim(references)
        translations = _trim(translations)

        reference_length += len(references)
        translation_length += len(translations)
        # print(reference_length, translation_length)
        ref_ngram_counts = _get_ngrams_with_counter(references.numpy(), max_order)
        translation_ngram_counts = _get_ngrams_with_counter(translations.numpy(), max_order)

        overlap = dict((ngram,
                        min(count, translation_ngram_counts[ngram]))
                       for ngram, count in ref_ngram_counts.items())

        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for ngram in translation_ngram_counts:
            possible_matches_by_order[len(ngram) - 1] += translation_ngram_counts[
                ngram]

    precisions = [0] * max_order
    smooth = 1.0

    for i in xrange(0, max_order):
        if possible_matches_by_order[i] > 0:
            precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]
            if matches_by_order[i] > 0:
                precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[
                    i]
            else:
                smooth *= 2
                precisions[i] = 1.0 / (smooth * possible_matches_by_order[i])
        else:
            precisions[i] = 0.0

    if max(precisions) > 0:
        p_log_sum = sum(math.log(p) for p in precisions if p)
        geo_mean = math.exp(p_log_sum / max_order)

    if use_bp:
        ratio = translation_length / reference_length
        bp = math.exp(1 - 1. / ratio) if ratio < 1.0 else 1.0
    bleu = geo_mean * bp
    return np.float32(bleu)

def _trim(ids):
    """Trim EOS and PAD tokens from ids, and decode to return a string."""
    try:
        index = list(ids).index(EOS_ID)
        #tf.logging.info(ids[:index])
        return ids[:index]
    except ValueError:  # No EOS found in sequence
        return ids

def _get_ngrams_with_counter(segment, max_order):
    """Extracts all n-grams up to a given maximum order from an input segment.
    Args:
      segment: text segment from which n-grams will be extracted.
      max_order: maximum length in tokens of the n-grams returned by this
          methods.
    Returns:
      The Counter containing all n-grams upto max_order in segment
      with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in xrange(1, max_order + 1):
        for i in xrange(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def _compute_accuracy(predictions, labels):
    """

    :param predictions: [batch, pred_seq_len]
    :param labels: [batch, seq_len]
    :return:
    """
    _pad_pred, _pad_labels = _pad_tensors_to_same_length(predictions, labels)
    _pad_pred = tf.cast(_pad_pred, _pad_labels.dtype)
    accuracy = tf.cast(tf.equal(_pad_pred, _pad_labels), tf.float32)
    weights = tf.cast(tf.not_equal(_pad_labels, 0), tf.float32)
    accuracy = tf.reduce_sum(accuracy * weights) / tf.reduce_sum(weights)
    return accuracy

import string
import unicodedata
import editdistance

def ocr_metrics(predicts, ground_truth, norm_accentuation=False, norm_punctuation=False):
    """Calculate Character Error Rate (CER), Word Error Rate (WER) and Sequence Error Rate (SER)"""

    if len(predicts) == 0 or len(ground_truth) == 0:
        return (1, 1, 1)

    cer, wer, ser = [], [], []

    for (pd, gt) in zip(predicts, ground_truth):
        if norm_accentuation:
            pd = unicodedata.normalize("NFKD", pd).encode("ASCII", "ignore").decode("ASCII")
            gt = unicodedata.normalize("NFKD", gt).encode("ASCII", "ignore").decode("ASCII")

        if norm_punctuation:
            pd = pd.translate(str.maketrans("", "", string.punctuation))
            gt = gt.translate(str.maketrans("", "", string.punctuation))

        pd_cer, gt_cer = list(pd.lower()), list(gt.lower())
        dist = editdistance.eval(pd_cer, gt_cer)
        cer.append(dist / (max(len(pd_cer), len(gt_cer))))

        pd_wer, gt_wer = pd.lower().split(), gt.lower().split()
        dist = editdistance.eval(pd_wer, gt_wer)
        wer.append(dist / (max(len(pd_wer), len(gt_wer))))

        pd_ser, gt_ser = [pd], [gt]
        dist = editdistance.eval(pd_ser, gt_ser)
        ser.append(dist / (max(len(pd_ser), len(gt_ser))))

    cer_f = sum(cer) / len(cer)
    wer_f = sum(wer) / len(wer)
    ser_f = sum(ser) / len(ser)

    return (cer_f, wer_f, ser_f)


if __name__ == "__main__":
    logits = tf.constant([[2, 2, 0],[4, 5, 0]], dtype=tf.int32)
    labels = tf.constant([[1,2],[3,4]], dtype=tf.int32)
    out = ocr_metrics(logits, labels)
    print(out)