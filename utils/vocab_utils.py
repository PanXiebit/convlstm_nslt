# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility to handle vocabularies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import os
import tensorflow as tf

from tensorflow.python.ops import lookup_ops

from utils import misc_utils as utils

PAD = "<pad>"
UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3


def check_vocab(vocab_file, out_dir, pad = None, sos=None, eos=None, unk=None):
    """Check if vocab_file doesn't exist, create from corpus_file."""
    if tf.io.gfile.exists(vocab_file):
        utils.print_out("# Vocab file %s exists" % vocab_file)
        vocab = []
        with codecs.getreader("utf-8")(tf.io.gfile.GFile(vocab_file, "rb")) as f:
            vocab_size = 0
            for word in f:
                vocab_size += 1
                vocab.append(word.strip())
        # Verify if the vocab starts with unk, sos, eos
        # If not, prepend those tokens & generate a new vocab file
        if not pad: pad = PAD
        if not sos: sos = SOS
        if not eos: eos = EOS
        if not unk: unk = UNK
        assert len(vocab) >= 4
        if vocab[0] != pad or vocab[1] != sos or vocab[2] != eos or vocab[3] != unk:
            utils.print_out("The first 4 vocab words [%s, %s, %s, %s]"
                            " are not [%s, %s, %s, %s]" %
                            (vocab[0], vocab[1], vocab[2], vocab[3], pad, sos, eos, pad))
            vocab = [pad, sos, eos, unk] + vocab
            vocab_size += 4
            new_vocab_file = os.path.join(out_dir, os.path.basename(vocab_file))
            with codecs.getwriter("utf-8")(tf.io.gfile.GFile(new_vocab_file, "wb")) as f:
                for word in vocab:
                    f.write("%s\n" % word)
            vocab_file = new_vocab_file
    else:
        raise ValueError("vocab_file does not exist.")

    vocab_size = len(vocab)
    return vocab_size, vocab_file


def create_tgt_vocab_table(tgt_vocab_file):
    """Creates vocab tables for src_vocab_file and tgt_vocab_file."""
    tgt_vocab_table = lookup_ops.index_table_from_file(tgt_vocab_file, default_value=UNK_ID)
    return tgt_vocab_table

def create_tgt_dict(tgt_vocab_file):
    word2idx = {"<pad>" : 0, "<s>":1, "</s>" :2, "<unk>":3}
    idx2word = {}
    count = len(word2idx)
    with open(tgt_vocab_file, "r") as f:
        for line in f:
            word = line.strip()
            if word not in word2idx:
                word2idx[word] = count
                count += 1
    for word, idx in word2idx.items():
        idx2word[idx] = word
    return word2idx, idx2word

if __name__ == "__main__":
    tgt_vocab_file = "/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.vocab.de"
    # tgt_vocab_table = create_tgt_vocab_table(tgt_vocab_file)
    # print(tgt_vocab_table.lookup(tf.constant("<pad>")))
    word2idx, idx2word = create_tgt_dict(tgt_vocab_file)
    print(idx2word[0])