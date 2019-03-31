import matplotlib.pyplot as plt
import tensorflow as tf

# read image data
image_raw_data = tf.gfile.FastGFile("cat.jpg").read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)

    print(img_data.eval())

    plt.imshow(img_data.eval())
    plt.show()