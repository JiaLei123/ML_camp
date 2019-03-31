import tensorflow as tf

# 声明一个队列，队列中最多100个元素
queue = tf.FIFOQueue(100, "float")
# 定义队列的入队操作
# tf.random_normal([1]) 产生一个shape为[1]的随机数
enqueue_op = queue.enqueue([tf.random_normal([1])])

# 创建多个线性运行入队操作
# [enqueue_op] * 5 表示需要启动五个线程，每个线程中运行的是enqueue_op操作
# 由于计算图的概念，这里只是定义计算，具体的操作要在sess.run中完成
qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)

# 将刚才定义的QueueRunner加入tf计算图上指定的集合 tf.GraphKeys.QUEUE_RUNNERS
tf.train.add_queue_runner(qr)

# 定义出队操作
out_tensor = queue.dequeue()

with tf.Session() as sess:
    # 使用Coordinator协同启动的线程
    coord = tf.train.Coordinator()

    # 调用start_queue_runners来启动所有的线程，它会启动tf.GraphKeys.QUEUE_RUNNERS集合中所有的QueueRunner
    # 这里要和add_queue_runner配合使用
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 获取队列中的值
    for _ in range(3):
        print(sess.run(out_tensor)[0])

    # 停止所有的线程
    coord.request_stop()
    coord.join(threads)




