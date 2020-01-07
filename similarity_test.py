from cnn_models import alexnet, resnet
from utils import dataset
import tensorflow as tf
from utils.vocab_utils import create_tgt_vocab_table

# CNN model, extract feature
input_shape = (227, 227)
# alexnet_model = alexnet.AlexNet(dropout_rate=00,
#                             weights_path="/home/panxie/Documents/sign-language/nslt/BaseModel/bvlc_alexnet.npy")
# alexnet_model.build((None,) + input_shape + (3,))
# alexnet_model.load_weights()

resnet_model = resnet.ResNet(layer_num=18, include_top=True)
resnet_model.build((None,) + input_shape + (3,))
resnet_model.load_weights("/home/panxie/Documents/sign-language/nslt/BaseModel/ResNet_18.h5")


#
def scaled_dot_product_attention(q, k, v, mask):
    """ caculate the attention weights.
    q, k, v must have matching leading dimensions.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.
    :param q: query shape == [..., q_len, d_model]
    :param k: key shape == [..., kv_len, d_model]
    :param v: value shape == [..., kv_len, d_model]
    :param mask: Float tensor with shape broadcastable to [..., q_len, kv_len]
    :return:
        output, attention_weights. [..., q_len, kv_len]
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # [..., q_len, kv_len]
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.sqrt(dk)
    # print("scaled_attention_logits", tf.nn.top_k(scaled_attention_logits[0, 0, :], k=12))
    # add the mask to the scaled tensor
    if mask is not None:
        scaled_attention_logits += (mask * 1e-9)
    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., q_len, kv_len)
    output = tf.matmul(attention_weights, v)  # [.., q_len, d_model] ? [.., k_len, d_model]
    return output, attention_weights, scaled_attention_logits


# print(os.getcwd())
base_path = "/home/panxie/Documents/sign-language/nslt/Data"
src_file = base_path + "/phoenix2014T.dev.sign"
tgt_file = base_path + "/phoenix2014T.dev.de"
tgt_vocab_table = create_tgt_vocab_table(base_path + "/phoenix2014T.vocab.de")
dataset = dataset.get_train_dataset(src_file, tgt_file, tgt_vocab_table)
cnt = 0
for data in dataset.take(1):
    cnt += 1
    src_inputs, tgt_in, tgt_out, src_path, src_len, tgt_len = data
    bs, t, h, w, c = src_inputs.shape
    print(src_inputs.shape, src_path)
    # src_inputs = tf.reshape(src_inputs, (bs*t, h, w, c))
    # cnn_output, _ = resnet_model(src_inputs, training=False)
    cnn_output = tf.reshape(src_inputs, (bs, t, -1))
    attention_out, atten_weights, atten_logits = scaled_dot_product_attention(cnn_output, cnn_output, cnn_output, mask=None)
    for i in range(100):
        # print(atten_logits[0, i, :])
        print(tf.nn.top_k(atten_logits[0, i, :], k=10).indices)

