
python -m main_sf_ctc \
	--tgt_vocab_file="/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.vocab.gloss" \
	--train_src_file="/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.train.sign" \
	--train_tgt_file="/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.train.gloss" \
	--eval_src_file="/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.dev.sign" \
	--eval_tgt_file="/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.dev.gloss" \
	--test_src_file="/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.test.sign" \
	--test_tgt_file="/home/panxie/Documents/sign-language/nslt/Data/phoenix2014T.test.gloss" \
	--output_dir="./output_dir/checkpoints_sfnet_ctc" \
	--best_output="./output_dir/checkpoints_sfnet_ctc/best_wer" \
	--rnn_units=256 \
	--steps_per_epoch=-1 \
	--learning_rate=1e-4 \
	--debug_num=-1 \
	> output_dir/train_sf_ctc.log 2>&1 &
