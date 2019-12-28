import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from utils.dataset import get_iterator
import sonnet as snt

resnet50 =  tf.keras.applications.resnet.ResNet50(weights="imagenet", include_top=False)

# snt.Conv3DLSTM()