
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
    output_dir = "./output_dir/checkpoints_resnet"
    translation_file = "./output_dir/dev_translation.txt"

    # rnn hyperparameters
    rnn_units = 512
    tgt_emb_size = 300
    label_smoothing = 0.1
    #
    learning_rate = 0.0001
    decay_steps = 1000
    beam_size = 3
    #
    batch_size = 301
    max_epochs = 20
    steps_per_epoch = -1   # run the whole dataset.
    eval_only = False
    debug_num = -1

