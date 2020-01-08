import os
import tensorflow as tf
from config import FLAGS
from utils import dataset, metrics
from models.mclstm_model import Model
from models.resnet_model import ModelResNet
from utils import vocab_utils, evaluation_utils, misc_utils, lr_schedule
from utils.lr_schedule import CustomSchedule
import time, math

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


#def get_learning_rate(learning_rate, hidden_size, learning_rate_warmup_steps, global_step):
#    with tf.name_scope("learning_rate"):
#        warmup_steps = tf.cast(learning_rate_warmup_steps, tf.float32)
#        step = tf.cast(global_step, tf.float32)
#        learning_rate *= (hidden_size ** -0.5)
#        learning_rate *= tf.minimum(1.0, step / warmup_steps)
#        learning_rate *= tf.math.rsqrt(tf.maximum(step, warmup_steps))
#        return learning_rate


config = FLAGS
logging.info(config)
tgt_vocab_size, tgt_vocab_file = vocab_utils.check_vocab(config.tgt_vocab_file,
                                                         "./",
                                                         sos="<s>",
                                                         eos="</s>",
                                                         unk=vocab_utils.UNK)
tgt_vocab_table = vocab_utils.create_tgt_vocab_table(config.tgt_vocab_file)
word2idx, idx2word = vocab_utils.create_tgt_dict(tgt_vocab_file)

# model = Model(rnn_units=config.rnn_units, tgt_vocab_size=tgt_vocab_size, tgt_emb_size=config.tgt_emb_size)
model = ModelResNet(input_shape=config.input_shape, tgt_vocab_size=tgt_vocab_size, tgt_emb_size=config.tgt_emb_size,
                    rnn_units=config.rnn_units, unit_type=config.unit_type,
                    num_layers=config.num_layers, residual=config.residual,
                    init_op=config.init_op, dropout=config.dropout, forget_bias=config.forget_bias)


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    config.learning_rate,
    decay_steps=config.decay_steps,
    decay_rate=0.96,
    staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
# learning_rate = CustomSchedule(config.rnn_units)
# optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
#                                      epsilon=1e-9)

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
    global_step = tf.Variable(int(tf.train.latest_checkpoint(checkpointdir).split("-")[1]), 
            trainable=False, dtype=tf.float32)
    global_epoch = int(tf.train.latest_checkpoint(checkpointdir).split("-")[2]) 
except:
    logger.info("======== No checkpoint found at {} ======".format(checkpointdir))
    global_step = tf.Variable(0, dtype=tf.float32, trainable=False)
    global_epoch = 0


# One step of training on a batch using Teacher Forcing technique
# use the @tf.function decorator to take advance of static graph computation (remove it when you want to debug)
# @tf.function
def train_step(batch_data):
    src_inputs, tgt_input_ids, tgt_output_ids, src_path, src_len, tgt_len = batch_data
    with tf.GradientTape() as tape:
        logits = model(batch_data, training=True)
        xentropy, weights = metrics.padded_cross_entropy_loss(logits, tgt_output_ids,
                                                              config.label_smoothing, vocab_size=tgt_vocab_size)
        loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)
    variables = model.Encoder.trainable_variables + model.Decoder.trainable_variables
    gradients = tape.gradient(target=loss, sources=variables)
    grads_and_vars = zip(gradients, variables)
    optimizer.apply_gradients(grads_and_vars)
    return loss


def eval():
    """internal evaluation """
    dev_dataset = dataset.get_train_dataset(src_file=config.eval_src_file, tgt_file=config.eval_tgt_file,
                                            tgt_vocab_table=tgt_vocab_table, batch_size=config.batch_size)
    total_cnt, total_loss, total_bleu = 0.0, 0.0, 0.0
    for batch_num, batch_data in enumerate(dev_dataset.take(config.debug_num)):
        src_inputs, tgt_input_ids, tgt_output_ids, src_path, src_len, tgt_len = batch_data
        logits = model(batch_data, training=True)
        bs = logits.shape[0]
        xentropy, weights = metrics.padded_cross_entropy_loss(logits, tgt_output_ids,
                                                              config.label_smoothing, vocab_size=tgt_vocab_size)
        batch_loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)
        batch_bleu = metrics.bleu_score(logits=logits, labels=tgt_output_ids)
        total_cnt += bs
        total_loss += bs * batch_loss
        total_bleu += bs * batch_bleu
    eval_loss = total_loss / total_cnt
    eval_bleu = total_bleu / total_cnt
    return eval_bleu, eval_loss


def infer():
    """External evaluation"""
    infer_dataset = dataset.get_infer_dataset(src_file=config.test_src_file, tgt_file=config.test_tgt_file,
                                              tgt_vocab_table=tgt_vocab_table)
    total_cnt, total_loss, total_bleu = 0.0, 0.0, 0.0
    for batch_num, batch_data in enumerate(infer_dataset.take(config.debug_num)):
        src, tgt, src_len, tgt_len = batch_data
        pred_logits = model((src, src_len), beam_size=config.beam_size, training=False)
        bs = pred_logits.shape[0]
        batch_bleu = metrics.bleu_score(logits=pred_logits, labels=tgt)
        total_cnt += bs
        total_bleu += bs * batch_bleu
        if batch_num % 50 == 0:
            predictions = pred_logits[0].numpy()
            label = tgt[0].numpy()
            pred_sent = [idx2word[idx] for idx in list(predictions)]
            label_sent = [idx2word[idx] for idx in list(label)]
            logger.info("\n reference sentences: {} \n predicted senteces: {}".format(" ".join(label_sent),
                                                                                      " ".join(pred_sent)))
    test_bleu = total_bleu / total_cnt
    return test_bleu


def main(global_step=global_step):
    train_dataset = dataset.get_train_dataset(src_file=config.train_src_file, tgt_file=config.train_tgt_file,
                                              tgt_vocab_table=tgt_vocab_table, batch_size=config.batch_size)
    init_bleu = 0
    if config.eval_only:
        logger.info("======== Evaluation only ===============")
        eval_bleu, eval_loss = eval()
        test_bleu = infer()
        logger.info("Eval loss {:.4f}, bleu {:.4f}".format(eval_loss, eval_bleu))
        logger.info("Test bleu {:.4f}".format(test_bleu))
    else:
        for epoch in range(global_epoch + 1, config.max_epochs):
            total_loss, total_cnt, step_time = 0.0, 0.0, 0.0
            for batch_data in train_dataset.take(config.steps_per_epoch):
                start_time = time.time()
                src_inputs, tgt_input_ids, tgt_output_ids, src_path, src_len, tgt_len = batch_data
                batch_size = src_inputs.shape[0]
                batch_loss = train_step(batch_data)
                total_loss += batch_loss * batch_size
                total_cnt += batch_size
                step_time += time.time() - start_time
                if (global_step + 1) % 100 == 0:
                    train_loss = total_loss / total_cnt
                    train_ppl = misc_utils.safe_exp(total_loss / total_cnt)
                    speed = total_cnt / step_time
                    # current_lr = learning_rate(global_step)
                    logger.info("epoch {} global_step {} example-time {:.2f} total loss: {:.4f} ppl {:.4f}".
                                format(epoch, global_step + 1, speed, train_loss, train_ppl))
                    if math.isnan(train_ppl):
                        break
                    total_loss, total_cnt, step_time = 0.0, 0.0, 0.0
                global_step = tf.add(global_step, 1)
            eval_bleu, eval_loss = eval()
            test_bleu = infer()
            logger.info("Epoch {}, Internal eval bleu {:.4f} loss {:.4f}, External test bleu {:.4f}".
                        format(epoch, eval_bleu, eval_loss, test_bleu))
            checkpoint.save(file_prefix=chkpoint_prefix + "_bleu_{:.4f}".format(test_bleu) + "-" + str(global_step))
            logger.info("Saving model to {}".format(
                chkpoint_prefix + "_bleu_{:.4f}".format(test_bleu) + "-" + str(global_step)))
            if test_bleu > init_bleu:
                checkpoint.save(
                    file_prefix=best_output + "_bleu_{:.4f}".format(test_bleu) + "-" + str(global_step))
                init_bleu = test_bleu
                logger.info("Currently the best bleu {:.4f}".format(test_bleu))


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
