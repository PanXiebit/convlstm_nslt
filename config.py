
# class Config():
#     # input file
#     tgt_vocab_file = "/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.vocab.de"  # vocab file
#     train_src_file = "/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.train.sign"
#     train_tgt_file = "/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.train.de"
#     eval_src_file = "/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.dev.sign"
#     eval_tgt_file = "/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.dev.de"
#     test_src_file = "/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.test.sign"
#     test_tgt_file = "/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.test.de"
#     # output file
#     output_dir = "./output_dir/checkpoints_resnet_3"
#     best_output = "./output_dir/checkpoints_resnet_3/best_bleu"
#     translation_file = "./output_dir/dev_translation.txt"
#
#     # rnn parameters
#     rnn_units = 1000
#     num_layers = 4
#     encoder_type = "uni"  # "uni"/"bi"/gnmt
#     unit_type = "lstm"
#     init_op = "glorot_normal"
#     forget_bias = True
#     dropout = 0.1
#     residual = True
#     # transformer parameters
#
#     # embedding
#     tgt_emb_size = 300
#
#     # learning rate
#     learning_rate = 0.0001
#     decay_steps = 1000
#
#     # train/eval/infer hyperparameters
#     label_smoothing = 0.1
#     beam_size = 3
#     batch_size = 301
#     max_epochs = 20
#     steps_per_epoch = -1   # run the whole dataset.
#     eval_only = False
#     debug_num = -1

import argparse
def add_arguments(parser):
    """Build ArgumentParser."""
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # output file
    parser.add_argument("--output_dir", type=str, help="output path",
                        default="./output_dir/checkpoints_alexnet")
    parser.add_argument("--best_output", type=str, help="best output path",
                        default="./output_dir/checkpoints_alexnet/best_bleu")
    parser.add_argument("--translation_file", type=str, help="test translation path",
                        default="./output_dir/dev_translation.txt")

    # rnn parameters
    parser.add_argument("--rnn_units", type=int, help="rnn hidden size",
                        default=1000)
    parser.add_argument("--num_layers", type=int, help="rnn number layers",
                        default=4)
    parser.add_argument("--encoder_type", type=str, help="rnn hidden size",
                        default="uni")
    parser.add_argument("--unit_type", type=str, help="rnn hidden size",
                        default="lstm")
    parser.add_argument("--init_op", type=str, help="rnn hidden size",
                        default="glorot_normal")
    parser.add_argument("--forget_bias", type=bool, help="rnn hidden size",
                        default=True)
    parser.add_argument("--dropout", type=float, help="rnn hidden size",
                        default=0.2)
    parser.add_argument("--residual", type=bool, help="rnn hidden size",
                        default=True)

    # hyperparameters
    parser.add_argument("--tgt_emb_size", type=int, help="rnn hidden size",
                        default=300)
    parser.add_argument("--learning_rate", type=float, help="rnn hidden size",
                        default=0.00001)
    parser.add_argument("--decay_steps", type=int, help="rnn hidden size",
                        default=1000)
    parser.add_argument("--label_smoothing", type=float, help="rnn hidden size",
                        default=0.1)
    parser.add_argument("--beam_size", type=int, help="rnn hidden size",
                        default=3)
    parser.add_argument("--batch_size", type=int, help="rnn hidden size",
                        default=301)
    parser.add_argument("--max_epochs", type=int, help="rnn hidden size",
                        default=20)
    parser.add_argument("--steps_per_epoch", type=int, help="rnn hidden size",
                        default=-1)

    # debug
    parser.add_argument("--eval_only", type=bool, help="rnn hidden size",
                        default=False)
    parser.add_argument("--debug_num", type=int, help="rnn hidden size",
                        default=-1)
    parser.add_argument("--input_shape", type=tuple, help="rnn hidden size",
                        default=(227, 227))

    # input file
    parser.add_argument("--tgt_vocab_file", type=str, help="target vocabulary file",
                        default="/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.vocab.de")
    parser.add_argument("--train_src_file", type=str, help="train source file",
                        default="/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.train.sign")
    parser.add_argument("--train_tgt_file", type=str, help="train target file",
                        default="/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.train.de")
    parser.add_argument("--eval_src_file", type=str, help="develop source file",
                        default="/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.dev.sign")
    parser.add_argument("--eval_tgt_file", type=str, help="develop target file",
                        default="/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.dev.de")
    parser.add_argument("--test_src_file", type=str, help="test source file",
                        default="/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.test.sign")
    parser.add_argument("--test_tgt_file", type=str, help="test target file",
                        default="/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.test.de")


nmt_parser = argparse.ArgumentParser()
add_arguments(nmt_parser)
FLAGS, unparsed = nmt_parser.parse_known_args()
print(FLAGS.input_shape)
