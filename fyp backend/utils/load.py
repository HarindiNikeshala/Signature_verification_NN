import tensorflow as tf
import os
import tensorflow.compat.v1 as tf

def load_model_from_checkpoint(ckpt_name="model.ckpt"):
   
    current_dir = os.path.dirname(__file__)  # utils/
    checkpoint_path = os.path.join(current_dir, '..', 'models')
    ckpt_dir = os.path.normpath(checkpoint_path)

    tf.compat.v1.disable_eager_execution()  # Required to use Sessions
    sess = tf.compat.v1.Session()
    saver = tf.compat.v1.train.import_meta_graph(f"{ckpt_dir}/{ckpt_name}.meta")
    saver.restore(sess, f"{ckpt_dir}/{ckpt_name}")
    graph = tf.compat.v1.get_default_graph()
    print("Model restored.")
    return sess, graph



# sess = load_model_from_checkpoint("model.ckpt")

# # Access the default graph
# graph = tf.compat.v1.get_default_graph()

# for op in graph.get_operations():
#     print(op.name)