import os
import tensorflow as tf
from config import FLAGS
from utils import dataset, metrics
from models.ctc_model import CTCModel
from utils import vocab_utils, evaluation_utils, misc_utils, lr_schedule
import time, math

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_learning_rate(learning_rate, hidden_size, learning_rate_warmup_steps, global_step):
    with tf.name_scope("learning_rate"):
        warmup_steps = tf.cast(learning_rate_warmup_steps, tf.float32)
        step = tf.cast(global_step, tf.float32)
        learning_rate *= (hidden_size ** -0.5)
        learning_rate *= tf.minimum(1.0, step / warmup_steps)
        learning_rate *= tf.math.rsqrt(tf.maximum(step, warmup_steps))
        return learning_rate


config = FLAGS
FLAGS.output_dir = "./output_dir/checkpoints_alexnet_ctc"
FLAGS.best_output = "./output_dir/checkpoints_alexnet_ctc/best_bleu"

for arg in vars(FLAGS):
    logger.info("{}, {}".format(arg, getattr(FLAGS, arg)))


tgt_vocab_size, tgt_vocab_file = vocab_utils.check_vocab(config.tgt_vocab_file,
                                                         "./",
                                                         sos="<s>",
                                                         eos="</s>",
                                                         unk=vocab_utils.UNK)

tgt_vocab_table = vocab_utils.create_tgt_vocab_table(config.tgt_vocab_file)
word2idx, idx2word = vocab_utils.create_tgt_dict(tgt_vocab_file)

# model = Model(rnn_units=config.rnn_units, tgt_vocab_size=tgt_vocab_size, tgt_emb_size=config.tgt_emb_size)
model = CTCModel(input_shape=config.input_shape, tgt_vocab_size=tgt_vocab_size, dropout=config.dropout,
                 rnn_units=FLAGS.rnn_units)


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    config.learning_rate,
    decay_steps=config.decay_steps,
    decay_rate=0.96,
    staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

checkpointdir = config.output_dir
chkpoint_prefix = os.path.join(checkpointdir, "checkpoint")
best_output = os.path.join(config.best_output, "checkpoint")
if not os.path.exists(checkpointdir):
    os.mkdir(checkpointdir)
    os.mkdir(config.best_output)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
try:
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpointdir))
    logger.info("======== Checkpoint found at {} ======".
                format(tf.train.latest_checkpoint(checkpointdir)))
    global_step = int(tf.train.latest_checkpoint(checkpointdir).split("-")[1])
except:
    logger.info("======== No checkpoint found at {} ======".format(checkpointdir))
    global_step = 0


# One step of training on a batch using Teacher Forcing technique
# use the @tf.function decorator to take advance of static graph computation (remove it when you want to debug)
# @tf.function
def train_step(batch_data):
    with tf.GradientTape() as tape:
        loss = model(batch_data, training=True)
    # !!! weather to train variables of cnn model
    variables = model.trainable_variables
    gradients = tape.gradient(target=loss, sources=variables)
    grads_and_vars = zip(gradients, variables)
    optimizer.apply_gradients(grads_and_vars)
    return loss


def infer():
    """External evaluation"""
    infer_dataset = dataset.get_infer_dataset(src_file=config.test_src_file, tgt_file=config.test_tgt_file,
                                              tgt_vocab_table=tgt_vocab_table)
    total_cnt, total_acc= 0.0, 0.0
    for batch_num, batch_data in enumerate(infer_dataset.take(config.debug_num)):
        src, tgt, src_len, tgt_len = batch_data
        predicts = model(batch_data, training=False)
        bs = predicts.shape[0]
        batch_acc = metrics._compute_accuracy(predictions=predicts, labels=tgt)
        total_cnt += bs
        total_acc += bs * batch_acc
        if batch_num % 50 == 0:
            predictions = predicts[0].numpy()
            label = tgt[0].numpy()
            pred_sent = [idx2word[idx] for idx in list(predictions)]
            label_sent = [idx2word[idx] for idx in list(label)]
            logger.info("\n reference sentences: {} \n predicted senteces: {}".format(" ".join(label_sent),
                                                                                      " ".join(pred_sent)))
    test_acc = total_acc / total_cnt
    return test_acc


def main(global_step=global_step):
    train_dataset = dataset.get_train_dataset(src_file=config.train_src_file, tgt_file=config.train_tgt_file,
                                              tgt_vocab_table=tgt_vocab_table, batch_size=config.batch_size)
    init_acc = 0
    if config.eval_only:
        logger.info("======== Evaluation only ===============")
        test_acc = infer()
        logger.info("Test acc {:.4f}".format(test_acc))
    else:
        for epoch in range(config.max_epochs):
            total_loss, total_cnt, step_time = 0.0, 0.0, 0.0
            for batch_data in train_dataset.take(config.steps_per_epoch):
                start_time = time.time()
                src_inputs, tgt_input_ids, tgt_output_ids, src_len, tgt_len = batch_data
                batch_size = src_inputs.shape[0]
                batch_loss = train_step(batch_data)
                total_loss += batch_loss * batch_size
                total_cnt += batch_size
                step_time += time.time() - start_time
                if (global_step + 1) % 100 == 0:
                    train_loss = total_loss / total_cnt
                    speed = total_cnt / step_time
                    logger.info("epoch {} global_step {} example-time {:.2f} total loss: {:.4f}".
                                format(epoch, global_step + 1, speed, train_loss))
                    total_loss, total_cnt, step_time = 0.0, 0.0, 0.0
                global_step += 1
            test_acc = infer()
            checkpoint.save(file_prefix=chkpoint_prefix + "_acc_{:.4f}".format(test_acc) + "-" + str(global_step))
            logger.info("Saving model to {}".format(
                chkpoint_prefix + "_acc_{:.4f}".format(test_acc) + "-" + str(global_step)))
            if test_acc > init_acc:
                checkpoint.save(
                    file_prefix=best_output + "_acc_{:.4f}".format(test_acc) + "-" + str(global_step))
                init_acc = test_acc
                logger.info("Currently the best acc {:.4f}".format(test_acc))


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
            logger.info(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    main()