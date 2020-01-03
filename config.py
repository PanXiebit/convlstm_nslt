
class Config():
    # input file
    tgt_vocab_file = "/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.vocab.de"  # vocab file
    train_src_file = "/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.train.sign"
    train_tgt_file = "/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.train.de"
    eval_src_file = "/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.dev.sign"
    eval_tgt_file = "/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.dev.de"
    test_src_file = "/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.test.sign"
    test_tgt_file = "/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.test.de"
    # output file
    output_dir = "./output_dir/checkpoints_resnet_3"
    translation_file = "./output_dir/dev_translation.txt"

    # rnn parameters
    rnn_units = 1000
    num_layers = 4
    encoder_type = "uni"  # "uni"/"bi"/gnmt
    unit_type = "lstm"
    init_op = "glorot_normal"
    forget_bias = True
    dropout = 0.1
    residual = True

    # transformer parameters


    # embedding
    tgt_emb_size = 300


    # learning rate
    learning_rate = 0.0001
    decay_steps = 1000

    # train/eval/infer hyperparameters
    label_smoothing = 0.1
    beam_size = 3
    batch_size = 301
    max_epochs = 20
    steps_per_epoch = -1   # run the whole dataset.
    eval_only = False
    debug_num = -1

