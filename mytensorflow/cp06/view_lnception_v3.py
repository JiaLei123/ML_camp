import tensorflow as tf
import os
MODEL_PATH = r"E:\ML_learning\tensorFlow\inception_dec_2015"
MODEL_FILE = "tensorflow_inception_graph.pb"


with tf.Session() as sess:
    with tf.gfile.FastGFile(os.path.join(MODEL_PATH, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

        writer = tf.summary.FileWriter("output1", sess.graph)
        writer.close()