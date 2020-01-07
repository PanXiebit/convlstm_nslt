
python -m main_ctc \
	--tgt_vocab_file="/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.vocab.gloss" \
	--train_src_file="/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.train.sign" \
	--train_tgt_file="/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.train.gloss" \
	--eval_src_file="/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.dev.sign" \
	--eval_tgt_file="/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.dev.gloss" \
	--test_src_file="/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.test.sign" \
	--test_tgt_file="/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.test.gloss" \
	--output_dir="./output_dir/checkpoints_alexnet_ctc" \
	--best_output="./output_dir/checkpoints_alexnet_ctc/best_bleu" \
	--rnn_units=200 \
	--steps_per_epoch=-1 \
	--debug_num=-1 \
	> output_dir/train_ctc.log 2>&1 &
