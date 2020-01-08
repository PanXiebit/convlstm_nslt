import os
import tensorflow as tf
from config import FLAGS
from cnn_models import alexnet, resnet
from models.SFNet import SFNet
from utils import dataset, vocab_utils
import time
import editdistance
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


config = FLAGS
for arg in vars(config):
    logger.info("{}, {}".format(arg, getattr(config, arg)))

tgt_vocab_size, tgt_vocab_file = vocab_utils.check_vocab(config.tgt_vocab_file,
                                                         "./",
                                                         sos="<s>",
                                                         eos="</s>",
                                                         unk=vocab_utils.UNK)

tgt_vocab_table = vocab_utils.create_tgt_vocab_table(tgt_vocab_file)
word2idx, idx2word = vocab_utils.create_tgt_dict(tgt_vocab_file)

# model = Model(rnn_units=config.rnn_units, tgt_vocab_size=tgt_vocab_size, tgt_emb_size=config.tgt_emb_size)
model = SFNet(input_shape=config.input_shape, tgt_vocab_size=tgt_vocab_size,
              rnn_units=config.rnn_units, cnn_model_path=config.alexnet_weight_path, dropout=config.dropout)


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    config.learning_rate,
    decay_steps=config.decay_steps,
    decay_rate=config.decay_rate,
    staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


chkpoint_prefix = os.path.join(config.output_dir, "checkpoint")
best_output_predfix = os.path.join(config.best_output, "checkpoint")
if not os.path.exists(config.output_dir):
    os.mkdir(config.output_dir)
    os.mkdir(config.best_output)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
try:
    status = checkpoint.restore(tf.train.latest_checkpoint(config.output_dir))
    logger.info("======== Checkpoint found at {} ======".
                format(tf.train.latest_checkpoint(config.output_dir)))
    global_step = tf.Variable(int(tf.train.latest_checkpoint(config.output_dir).split("-")[1]), trainable=False)
except:
    logger.info("======== No checkpoint found at {} ======".format(config.output_dir))
    global_step = tf.Variable(0, trainable=False)


# One step of training on a batch using Teacher Forcing technique
# use the @tf.function decorator to take advance of static graph computation (remove it when you want to debug)


# @tf.function
def train_step(batch_data, epoch, var_print):
    with tf.GradientTape() as tape:
        ctc_loss, reg_loss = model(batch_data, training=True)
        if epoch == 0:
            loss = ctc_loss
        else:
            loss = ctc_loss + reg_loss
    # trainable variables
    if isinstance(model.cnn_model, resnet.ResNet):
        # logger.info("==== CNN model using ResNet ====")
        total_variables = model.trainable_variables
        # variable not to training
        remove_var = {}
        for var in model.cnn_model.fc.trainable_variables:
            remove_var[var.name] = var
        if epoch == 0:
            for var in model.gloss_fc.trainable_variables:
                remove_var[var.name] = var
        # last variable needed to train
        variables = []
        for var in total_variables:
            if var.name not in remove_var:
                variables.append(var)
                if var_print: logger.info("{}, {}".format(var.name, var.shape))
    elif isinstance(model.cnn_model, alexnet.AlexNet):
        # logger.info("==== CNN model using AlexNet ====")
        total_variables = model.trainable_variables
        # variable not to training
        remove_var = {}
        if epoch == 0:
            for var in model.gloss_fc.trainable_variables:
                remove_var[var.name] = var
        # last variable needed to train
        variables = []
        for var in total_variables:
            if var.name not in remove_var:
                variables.append(var)
                if var_print: logger.info("{}, {}".format(var.name, var.shape))
    else:
        raise ValueError("cnn model architecture isn't existed!")
    gradients = tape.gradient(target=loss, sources=variables)
    grads_and_vars = zip(gradients, variables)
    optimizer.apply_gradients(grads_and_vars)
    return ctc_loss, reg_loss


def infer():
    """External evaluation"""
    infer_dataset = dataset.get_infer_dataset(src_file=config.test_src_file, tgt_file=config.test_tgt_file,
                                              filter=True, tgt_vocab_table=tgt_vocab_table)
    total_wer = []
    for batch_num, batch_data in enumerate(infer_dataset.take(config.debug_num)):
        src, tgt, src_len, tgt_len = batch_data
        predicts = model(batch_data, training=False)
        for pred_ids, tgt_ids in zip(predicts.numpy(), tgt.numpy()):
            pred_sent = [idx2word(idx) for idx in list(pred_ids)]
            tgt_sent = [idx2word(idx) for idx in list(tgt_ids)]
            dist = editdistance.eval(pred_sent, tgt_sent)
            wer = dist / (max(len(pred_sent), len(tgt_sent)))
            total_wer.append(wer)
        if batch_num % 50 == 0:
            logger.info("\n reference sentences: {} \n predicted senteces: {}".format(" ".join(tgt_sent),
                                                                                      " ".join(pred_sent)))
    test_wer = sum(total_wer) / len(total_wer)
    return test_wer


def main(global_step=global_step):
    train_dataset = dataset.get_train_dataset(src_file=config.train_src_file, tgt_file=config.train_tgt_file,
                                              tgt_vocab_table=tgt_vocab_table, batch_size=config.batch_size)
    init_wer = 0
    if config.eval_only:
        logger.info("======== Evaluation only ===============")
        test_wer = infer()
        logger.info("Test wer {:.4f}".format(test_wer))
    else:
        for epoch in range(config.max_epochs):
            total_ctc_loss, total_reg_loss, total_cnt, step_time = 0.0, 0.0, 0.0, 0.0
            for batch_data in train_dataset.take(config.steps_per_epoch):
                start_time = time.time()
                src_inputs, tgt_input_ids, tgt_output_ids, src_path, src_len, tgt_len = batch_data
                batch_size = src_inputs.shape[0]
                ctc_loss, reg_loss = train_step(batch_data, epoch, global_step == 0)
                total_ctc_loss += ctc_loss * batch_size
                total_reg_loss += reg_loss * batch_size
                total_cnt += batch_size
                step_time += time.time() - start_time
                if (global_step + 1) % 100 == 0:
                    train_ctc_loss = total_ctc_loss / total_cnt
                    train_reg_loss = total_reg_loss / total_cnt
                    speed = total_cnt / step_time
                    logger.info("epoch {} global_step {} example-time {:.2f} ctc loss: {:.4f} reg loss: {:.4f}".
                                format(epoch, global_step + 1, speed, train_ctc_loss, train_reg_loss))
                    total_ctc_loss, total_reg_loss, total_cnt, step_time = 0.0, 0.0, 0.0, 0.0
                global_step = tf.add(global_step, 1)

            test_wer = infer()
            save_file_prefix=chkpoint_prefix + "_wer_{:.4f}".format(test_wer) + "-" + str(global_step.numpy())
            checkpoint.save(save_file_prefix)
            logger.info("Saving model to {}".format(save_file_prefix))
            if test_wer > init_wer:
                checkpoint.save(
                    file_prefix=best_output_predfix + "_wer_{:.4f}".format(test_wer) + "-" + str(global_step.numpy()))
                init_wer = test_wer
                logger.info("Currently the best wer {:.4f}".format(test_wer))


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
