import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

BOTTLENECK_TENSOR_SIZE = 2048
BOTTLENECK_TENSOR_NAME = "pool_3/_reshape:0"

JPEG_DATA_TENSOR_NAME = "DecodeJpeg/contents:0"

MODEL_PATH = r"E:\ML_learning\tensorFlow\inception_dec_2015"
MODEL_FILE = "tensorflow_inception_graph.pb"

CHACH_DIR = "./temp"

INPUT_DATA = r"E:\ML_learning\tensorFlow\flower_photos"

VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 1000


def create_image_lists(testing_presentage, validation_precentage):
    result = {}
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ["jpg", 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, "*." + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue

        label_name = dir_name.lower()
        train_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            chance = np.random.randint(100)
            if chance < validation_precentage:
                validation_images.append(base_name)
            elif chance < (testing_presentage + validation_precentage):
                testing_images.append(base_name)
            else:
                train_images.append(base_name)
        result[label_name] = {
            'dir': dir_name,
            'training': train_images,
            'testing': testing_images,
            'validation': validation_images
        }
    return result


def get_image_path(image_lists, image_dir, label_name, index, category):
    """
    get the image path base on category, data_set, index of image
    :param image_lists:
    :param image_dir:
    :param label_name:
    :param index:
    :param category:
    :return:
    """
    label_lists = image_lists[label_name]
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


def get_bottleneck_path(image_lists, label_name, index, category):
    return get_image_path(image_lists, CHACH_DIR, label_name, index, category) + ".txt"


def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    '''
    使用加载好的训练好的Inception-v3模型(不包含全连接层)处理一张图片，得到这个图片的特征向量
    :param sess:
    :param image_data:
    :param image_data_tensor:
    :param bottleneck_tensor:
    :return:
    '''
    # 将当前图片作为输入计算瓶颈层张量的值。
    # bottleneck_tensor 计算节点，这是最重要的，表明要计算到的地方。
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    # 将bottleneck_values得到的四维数组压缩成一个一维数据（特征向量）
    bottleneck_value = np.squeeze(bottleneck_values)
    return bottleneck_value


def get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    '''
    每个图片的特征向量会单独存储为一个文件，所以如果这文件不存在就要使用模型计算出它的特征向量并保存
    :param sess:
    :param image_lists:
    :param label_name:
    :param index:
    :param category:
    :param jpeg_data_tensor:
    :param bottleneck_tensor:
    :return:
    '''
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CHACH_DIR, sub_dir)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category)

    if not os.path.exists(bottleneck_path):
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)
        image_data = gfile.FastGFile(image_path, 'rb').read()
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        bottleneck_string = ",".join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [x for x in bottleneck_string.split(",")]

    return bottleneck_values


def get_random_cached_bottlenecks(sess, image_lists, n_classes, how_many, category, jpeg_data_tensor,
                                  bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, image_index, category, jpeg_data_tensor,
                                              bottleneck_tensor)
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    for label_index, label_name in enumerate(label_name_list):
        category = 'testing'
        for index, unused_base_name in enumerate(image_lists[label_name][category]):
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor,
                                                  bottleneck_tensor)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


def main():
    image_lists = create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    n_classes = len(image_lists.keys())
    # 读取已经训练好的Inception-v3模型，这个模型中保存了每个节点取值的计算方法以及变量（参数）的取值。
    with gfile.FastGFile(os.path.join(MODEL_PATH, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # 加载模型，并返回数据输入所对应的张量以及计算瓶颈层结果所对应的张量
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, return_elements=[BOTTLENECK_TENSOR_NAME,
                                                                                          JPEG_DATA_TENSOR_NAME])
    # 定义新的神经网络全连接层的输入，也就是Inception-v3模型前向传播到达瓶颈层的输出。
    bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name="BottleneckInputPlaceholder")
    # 新的label
    ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name="GroundTruthInput")

    # 新的全连接层用来解决图片分类的问题，
    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input, weights) + biases
        final_tensor = tf.nn.softmax(logits)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)

    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(STEPS):
            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(sess, image_lists, n_classes, BATCH,
                                                                                  'training', jpeg_data_tensor,
                                                                                  bottleneck_tensor)
            sess.run(train_step,
                     feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})
            if i % 100 == 0 or i + 1 == STEPS:
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(sess, image_lists,
                                                                                                n_classes,
                                                                                                BATCH, 'validation',
                                                                                                jpeg_data_tensor,
                                                                                                bottleneck_tensor)
                validation_accuracy = sess.run(evaluation_step,
                                               feed_dict={bottleneck_input: validation_bottlenecks,
                                                          ground_truth_input: validation_ground_truth})

                print(
                    "Step %d Validation accuracy on random %d sampled examples = %f" % (i, BATCH, validation_accuracy))

        test_bottlenecks, test_ground_truth = get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor,
                                                                   bottleneck_tensor)
        test_accuracy = sess.run(evaluation_step,
                                 feed_dict={bottleneck_input: test_bottlenecks,
                                            ground_truth_input: test_ground_truth})

        print("Final accuracy on %f" % test_accuracy)

if __name__ == "__main__":
    main()