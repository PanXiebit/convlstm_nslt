import os
import tensorflow as tf
from config import Config
from utils import dataset
from seq2seq import model
from utils import vocab_utils, evaluation_utils, misc_utils
import time, math

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def loss_function(y_pred, y):
    """

    :param y_pred:  [batch_size, ty]
    :param y:  [batch_size, Ty, output_vocab_size]
    :return:
    """
    sparsecategoricalcrossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                                  reduction='none')
    loss = sparsecategoricalcrossentropy(y_true=y, y_pred=y_pred)

    mask = tf.logical_not(tf.math.equal(y, 0))  # output 0 for y=0 else output 1
    mask = tf.cast(mask, dtype=loss.dtype)
    loss = mask * loss
    loss = tf.reduce_mean(loss)
    return loss


config = Config()
tgt_vocab_size, tgt_vocab_file = vocab_utils.check_vocab(config.tgt_vocab_file,
                                                         "./",
                                                         sos="<s>",
                                                         eos="</s>",
                                                         unk=vocab_utils.UNK)
tgt_vocab_table = vocab_utils.create_tgt_vocab_table(config.tgt_vocab_file)
word2idx, idx2word = vocab_utils.create_tgt_dict(tgt_vocab_file)

model = model.Model(rnn_units=config.rnn_units, tgt_vocab_size=tgt_vocab_size, tgt_emb_size=config.tgt_emb_size)
optimizer = tf.keras.optimizers.Adam(lr=config.learning_rate)

checkpointdir = config.output_dir
chkpoint_prefix = os.path.join(checkpointdir, "checkpoint")
if not os.path.exists(checkpointdir):
    os.mkdir(checkpointdir)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
try:
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpointdir))
    print("Checkpoint found at {}".format(tf.train.latest_checkpoint(checkpointdir)))
except:
    print("No checkpoint found at {}".format(checkpointdir))


# One step of training on a batch using Teacher Forcing technique
def train_step(batch_data):
    src_inputs, tgt_input_ids, tgt_output_ids, src_len, tgt_len = batch_data
    loss = 0
    with tf.GradientTape() as tape:
        logits = model(batch_data, training=True)
        loss = loss_function(logits, tgt_output_ids)
    variables = model.trainable_variables
    gradients = tape.gradient(target=loss, sources=variables)
    grads_and_vars = zip(gradients, variables)
    optimizer.apply_gradients(grads_and_vars)
    return loss


def eval():
    eval_dataset = dataset.get_infer_dataset(src_file=config.eval_src_file)
    with open(config.translation_file, "w") as f:
        for batch_num, batch_data in enumerate(eval_dataset.take(-1)):
            src_video, src_len = batch_data
            predictions = model((src_video, src_len), beam_size=config.beam_size, training=False)
            predictions = predictions.numpy()
            for i in range(len(predictions)):
                pred_sent = [idx2word[idx] for idx in list(predictions[i])]
                f.write(" ".join(pred_sent) + "\n")
        bleu_score = evaluation_utils.evaluate(config.eval_tgt_file, config.translation_file, metric="bleu")
        accuracy = evaluation_utils.evaluate(config.eval_tgt_file, config.translation_file, metric="accuracy")
    return bleu_score, accuracy


def main():
    train_dataset = dataset.get_train_dataset(src_file=config.train_src_file, tgt_file=config.train_tgt_file,
                                              tgt_vocab_table=tgt_vocab_table)
    init_bleu = 0
    for epoch in range(config.max_epochs):
        total_loss = 0.0
        total_cnt = 0
        step_time = 0
        for global_step, batch_data in enumerate(train_dataset.take(-1)):
            start_time = time.time()
            src_inputs, tgt_input_ids, tgt_output_ids, src_len, tgt_len = batch_data
            batch_size = src_inputs.shape[0]
            batch_loss = train_step(batch_data)
            total_loss += batch_loss * batch_size
            total_cnt += batch_size
            step_time += time.time() - start_time
            if (global_step + 1) % 100 == 0:
                train_loss = total_loss / total_cnt
                train_ppl = misc_utils.safe_exp(total_loss / total_cnt)
                speed = total_cnt / step_time
                logger.info("epoch {} global_step {} example-time {:.2f} total loss: {:.4f} ppl {:.4f}".
                            format(epoch, global_step + 1, speed, train_loss, train_ppl))
                if math.isnan(train_ppl):
                    break
        bleu_score, accuracy = eval()
        logger.info("epoch {} accuarcy {:.4f} bleu : {:.4f}".format(epoch, accuracy, bleu_score))
        if bleu_score > init_bleu:
            checkpoint.save(file_prefix=chkpoint_prefix + "_bleu_{:.4f}".format(bleu_score))
            init_bleu = bleu_score
            logger.info("Currently the best bleu {:.4f}".format(bleu_score))


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    main()
