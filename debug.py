import tensorflow as tf
import pickle, json, numpy

checkpoint_dir = "/home/panxie/Documents/sign-language/nslt_tf2/nstl/output_dir/checkpoints"

ckpt = tf.train.latest_checkpoint(checkpoint_dir)


path = "/home/panxie/Documents/sign-language/nslt_tf2/nstl/output_dir/checkpoints/debug.json"
save_dict = {}
reader = tf.compat.v1.train.NewCheckpointReader(ckpt)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    # print("tensor_name: ", key)   # 打印变量名
    # print(reader.get_tensor(key)) # 打印变量值
    value = reader.get_tensor(key)
    if isinstance(value, numpy.ndarray):
        save_dict[key] = value.tolist()
    else:
        print(key)

with open(path, "w") as f:
    json.dump(save_dict, f, indent=4)