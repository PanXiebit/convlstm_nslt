import os
import tensorflow as tf
from config import Config
from utils import dataset
from seq2seq import model
from utils import vocab_utils
# from utils.vocab_utils import create_tgt_vocab_table
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def loss_function(y_pred, y):
    # shape of y [batch_size, ty]
    # shape of y_pred [batch_size, Ty, output_vocab_size]
    sparsecategoricalcrossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                                  reduction='none')
    loss = sparsecategoricalcrossentropy(y_true=y, y_pred=y_pred)
    # skip loss calculation for padding sequences i.e. y = 0
    # [ <start>,How, are, you, today, 0, 0, 0, 0 ....<end>]
    # [ 1, 234, 3, 423, 3344, 0, 0 ,0 ,0, 2 ]
    # y is a tensor of [batch_size,Ty] . Create a mask when [y=0]
    # mask the loss when padding sequence appears in the output sequence
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

model = model.Model(rnn_units=config.rnn_units, tgt_vocab_size=tgt_vocab_size, tgt_emb_size=config.tgt_emb_size)
optimizer = tf.keras.optimizers.Adam(lr=config.learning_rate)

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

checkpointdir = config.output_dir
chkpoint_prefix = os.path.join(checkpointdir, "checkpoint")
if not os.path.exists(checkpointdir):
    os.mkdir(checkpointdir)

checkpoint = tf.train.Checkpoint(optimizer = optimizer, model=model)
try:
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpointdir))
    print("Checkpoint found at {}".format(tf.train.latest_checkpoint(checkpointdir)))
except:
    print("No checkpoint found at {}".format(checkpointdir))


def eval():
    eval_dataset = dataset.get_dataset(src_file=config.eval_src_file, tgt_file=config.eval_tgt_file,
                                       tgt_vocab_table=tgt_vocab_table)
    for batch_num, batch_data in enumerate(eval_dataset.take(-1)):
        pass



def main():
    train_dataset = dataset.get_dataset(src_file=config.train_src_file, tgt_file=config.train_tgt_file,
                                        tgt_vocab_table=tgt_vocab_table)
    for i in range(config.max_epochs):
        total_loss = 0.0
        for batch_num, batch_data in enumerate(train_dataset.take(config.steps_per_epoch)):
            batch_loss = train_step(batch_data)
            if (batch_num + 1) % 5 == 0:
                logger.info("total loss: {} epoch {} batch {} ".format(batch_loss.numpy(), i, batch_num + 1))
                # checkpoint.save(file_prefix=chkpoint_prefix)




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
    train()