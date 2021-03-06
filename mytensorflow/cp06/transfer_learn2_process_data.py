import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

INPUT_DATA = r"E:\ML_learning\tensorFlow\flower_photos"
OUTPUT_FILE = "./temp2/flower_processed_data.npy"

VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10


def create_image_lists(sess, testing_presentage, validation_precentage):
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True

    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    validation_images = []
    validation_labels = []

    for sub_dir in sub_dirs:
        print("parse dat for %s" % sub_dir)
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

            current_label = dir_name.lower()
            for file_name in file_list:
                image_raw_data = gfile.FastGFile(file_name, 'rb').read()
                image = tf.image.decode_jpeg(image_raw_data)
                if image.dtype != tf.float32:
                    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
                image = tf.image.resize_images(image, [299, 299])
                image_value = sess.run(image)

                chance = np.random.randint(100)
                if chance < validation_precentage:
                    validation_images.append(image_value)
                    validation_labels.append(current_label)
                elif chance < (testing_presentage + validation_precentage):
                    testing_images.append(image_value)
                    testing_labels.append(current_label)
                else:
                    training_images.append(image_value)
                    training_labels.append(current_label)
    state = np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(state)
    np.random.shuffle(training_labels)
    return np.asarray(
        [training_images, training_labels, validation_images, validation_labels, testing_images, testing_labels])


def main():
    with tf.Session() as sess:
        processed_data = create_image_lists(sess, TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
        np.save(OUTPUT_FILE, processed_data)


if __name__ == "__main__":
    main()
