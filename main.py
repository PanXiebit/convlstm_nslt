import os
import tensorflow as tf
from config import Config
from utils import dataset, metrics
from models.model import Model
from models.model_resnet import ModelResNet
from utils import vocab_utils, evaluation_utils, misc_utils, lr_schedule
import time, math

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# def loss_function(y_pred, y):
#     """
#
#     :param y:  [batch_size, tgt_len]
#     :param y_pred:  [batch_size, tgt_len, output_vocab_size]
#     :return:
#     """
#     sparsecategoricalcrossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
#                                                                                   reduction='none')
#
#     loss = sparsecategoricalcrossentropy(y_true=y, y_pred=y_pred)
#
#     mask = tf.logical_not(tf.math.equal(y, 0))  # output 0 for y=0 else output 1
#     mask = tf.cast(mask, dtype=loss.dtype)
#     loss = mask * loss
#     loss = tf.reduce_mean(loss)
#     return loss

def get_learning_rate(learning_rate, hidden_size, learning_rate_warmup_steps, global_step):
    with tf.name_scope("learning_rate"):
        warmup_steps = tf.cast(learning_rate_warmup_steps, tf.float32)
        step = tf.cast(global_step, tf.float32)
        learning_rate *= (hidden_size ** -0.5)
        learning_rate *= tf.minimum(1.0, step / warmup_steps)
        learning_rate *= tf.math.rsqrt(tf.maximum(step, warmup_steps))
        return learning_rate

config = Config()
tgt_vocab_size, tgt_vocab_file = vocab_utils.check_vocab(config.tgt_vocab_file,
                                                         "./",
                                                         sos="<s>",
                                                         eos="</s>",
                                                         unk=vocab_utils.UNK)
tgt_vocab_table = vocab_utils.create_tgt_vocab_table(config.tgt_vocab_file)
word2idx, idx2word = vocab_utils.create_tgt_dict(tgt_vocab_file)

# model = Model(rnn_units=config.rnn_units, tgt_vocab_size=tgt_vocab_size, tgt_emb_size=config.tgt_emb_size)
model = ModelResNet(rnn_units=config.rnn_units, tgt_vocab_size=tgt_vocab_size, tgt_emb_size=config.tgt_emb_size)

checkpointdir = config.output_dir
if tf.train.latest_checkpoint(checkpointdir) is not None:
    # global_step = tf.Variable(initial_value=int(tf.train.latest_checkpoint(checkpointdir).split("-")[-1]),
    #                           trainable=False, dtype=tf.int32)
    global_step = int(tf.train.latest_checkpoint(checkpointdir).split("-")[-1])
else:
    # global_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)
    global_step = 0

# optimizer = lr_schedule.create_optimizer(init_lr=config.learning_rate,
#                                          num_train_steps=global_step,
#                                          num_warmup_steps=1000)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    config.learning_rate,
    decay_steps=config.decay_steps,
    decay_rate=0.96,
    staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

chkpoint_prefix = os.path.join(checkpointdir, "checkpoint")
if not os.path.exists(checkpointdir):
    os.mkdir(checkpointdir)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
try:
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpointdir))
    logger.info("======== Checkpoint found at {} ======".
                format(tf.train.latest_checkpoint(checkpointdir)))
except:
    logger.info("======== No checkpoint found at {} ======".format(checkpointdir))
    global_step = 0


# One step of training on a batch using Teacher Forcing technique
def train_step(batch_data):
    src_inputs, tgt_input_ids, tgt_output_ids, src_len, tgt_len = batch_data
    loss = 0
    with tf.GradientTape() as tape:
        logits = model(batch_data, training=True)
        # loss = loss_function(logits, tgt_output_ids)
        xentropy, weights = metrics.padded_cross_entropy_loss(logits, tgt_output_ids,
                                                              config.label_smoothing, vocab_size=tgt_vocab_size)
        loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)
    variables = model.Encoder.trainable_variables + model.Decoder.trainable_variables
    # for param in variables:
    #     logger.info("{}, {}".format(param.name, param.shape))
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
        src_inputs, tgt_input_ids, tgt_output_ids, src_len, tgt_len = batch_data
        logits = model(batch_data, training=True)
        bs = logits.shape[0]
        xentropy, weights = metrics.padded_cross_entropy_loss(logits, tgt_output_ids,
                                                              config.label_smoothing, vocab_size=tgt_vocab_size)
        batch_loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)
        batch_bleu = metrics.bleu_score(logits=logits, labels=tgt_output_ids)
        # print("Internal batch bleu :", batch_bleu)
        total_cnt += bs
        total_loss += bs * batch_loss
        total_bleu += bs * batch_bleu
    eval_loss = total_loss / total_cnt
    eval_bleu = total_bleu / total_cnt
    return eval_bleu, eval_loss


# def infer():
#     """External evaluation"""
#     infer_dataset = dataset.get_infer_dataset(src_file=config.test_src_file)
#     with open(config.translation_file, "w") as f:
#         for batch_num, batch_data in enumerate(infer_dataset.take(-1)):
#             src_video, src_len = batch_data
#             predictions = model((src_video, src_len), beam_size=config.beam_size, training=False)
#             predictions = predictions.numpy()
#             for i in range(len(predictions)):
#                 pred_sent = [idx2word[idx] for idx in list(predictions[i])]
#                 f.write(" ".join(pred_sent) + "\n")
#     bleu_score = evaluation_utils.evaluate(config.test_tgt_file, config.translation_file, metric="bleu")
#     accuracy = evaluation_utils.evaluate(config.test_tgt_file, config.translation_file, metric="accuracy")
#     return bleu_score, accuracy

def infer():
    """External evaluation"""
    infer_dataset = dataset.get_infer_dataset(src_file=config.test_src_file, tgt_file=config.test_tgt_file,
                                              tgt_vocab_table=tgt_vocab_table)
    total_cnt, total_loss, total_bleu = 0.0, 0.0, 0.0
    for batch_num, batch_data in enumerate(infer_dataset.take(config.debug_num)):
        src, tgt, src_len, tgt_len = batch_data
        pred_logits = model((src, src_len), beam_size=config.beam_size, training=False)
        bs = pred_logits.shape[0]
        # xentropy, weights = metrics.padded_cross_entropy_loss(pred_logits, tgt,
        #                                                       config.label_smoothing, vocab_size=tgt_vocab_size)
        # batch_loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)
        batch_bleu = metrics.bleu_score(logits=pred_logits, labels=tgt)
        # print("External batch bleu :", batch_bleu)
        total_cnt += bs
        # total_loss += bs * batch_loss
        total_bleu += bs * batch_bleu
        if batch_num % 100 == 0:
            predictions = pred_logits[0].numpy()
            label = tgt[0].numpy()
            pred_sent = [idx2word[idx] for idx in list(predictions)]
            label_sent = [idx2word[idx] for idx in list(label)]
            logger.info("reference sentences: {},\n predicted senteces: {}".format(" ".join(label_sent), " ".join(pred_sent)))
    # test_loss = total_loss / total_cnt
    test_bleu = total_bleu / total_cnt
    # return test_bleu, test_loss
    return test_bleu


def main(global_step=global_step):
    train_dataset = dataset.get_train_dataset(src_file=config.train_src_file, tgt_file=config.train_tgt_file,
                                              tgt_vocab_table=tgt_vocab_table, batch_size=config.batch_size)
    init_bleu = 0
    if config.eval_only:
        logger.info("======== Evaluation only ===============")
        # eval_bleu, eval_loss = eval()
        test_bleu = infer()
        # test_bleu, test_accuracy = infer()
        # logger.info("Eval loss {:.4f}, bleu {:.4f}".format(eval_loss, eval_bleu))
        logger.info("Test bleu {:.4f}".format(test_bleu))
        # logger.info("Test accuracy {:.4f}, bleu : {:.4f}".format(test_accuracy, test_bleu))
    else:
        for epoch in range(config.max_epochs):
            total_loss, total_cnt, step_time = 0.0, 0.0, 0.0
            for batch_data in train_dataset.take(config.debug_num):
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
                    total_loss, total_cnt, step_time = 0.0, 0.0, 0.0
                global_step += 1
            eval_bleu, eval_loss = eval()
            test_bleu = infer()
            logger.info("Epoch {}, Internal eval bleu {:.4f} loss {:.4f}, External test bleu {:.4f}".
                        format(epoch, eval_bleu, eval_loss, test_bleu))
            if test_bleu > init_bleu:
                checkpoint.save(
                    file_prefix=chkpoint_prefix + "_bleu_{:.4f}".format(test_bleu) + "-" + str(global_step))
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
