
class Config():
    # file
    tgt_vocab_file = "/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.vocab.de"
    train_src_file = "/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.train.sign"
    train_tgt_file = "/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.train.de"
    eval_src_file = "/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.eval.sign"
    eval_tgt_file = "/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.eval.de"
    test_src_file = "/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.test.sign"
    test_tgt_file = "/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.test.de"
    output_dir = "./checkpoints"

    # rnn hyperparameters
    rnn_units = 256
    tgt_emb_size = 300
    #
    learning_rate = 0.001
    #
    max_epochs = 10
    steps_per_epoch = -1   # run the whole dataset.
