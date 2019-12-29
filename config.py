
class Config():
    # input file
    tgt_vocab_file = "/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.vocab.de"
    train_src_file = "/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.train.sign"
    train_tgt_file = "/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.train.de"
    eval_src_file = "/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.dev.sign"
    eval_tgt_file = "/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.dev.de"
    test_src_file = "/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.test.sign"
    test_tgt_file = "/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.test.de"

    # output file
    output_dir = "./output_dir/checkpoints"
    translation_file = "./output_dir/dev_translation.txt"
    reference_file = "./output_dir/dev_reference.txt"

    # rnn hyperparameters
    rnn_units = 256
    tgt_emb_size = 300
    #
    learning_rate = 0.001
    beam_size = 1
    #
    max_epochs = 20
    steps_per_epoch = -1   # run the whole dataset.

