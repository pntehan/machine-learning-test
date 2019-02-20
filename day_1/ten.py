import tensorflow as tf
import numpy as np

# x_data = np.float32(np.random.rand(2, 100))
# y_data = np.dot([0.300, 0.200], x_data) + 0.400
#
# # print(x_data)
# # print(y_data)
#
# x = tf.placeholder(tf.float32, [2, 100])
# y = tf.placeholder(tf.float32, [100])
# b = tf.Variable(tf.zeros([1]))
# w = tf.Variable(tf.random_uniform([2], -1.0, 1.0))
# y_ = tf.matmul(tf.reshape(w, [1, 2]), x) + b
#
# loss = tf.reduce_mean(tf.square(y - y_))
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(loss)
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     for step in range(201):
#         sess.run(train, feed_dict={x: x_data, y: y_data})
#         if step % 10 == 0:
#             print(step, sess.run(w), sess.run(b))

# sess = tf.InteractiveSession()
#
# x = tf.Variable([1.0, 2.0])
# a = tf.constant([3.0, 3.0])
#
# x.initializer.run()
#
# sub = tf.subtract(x, a)
# print(sub.eval())
# sess.close()
#
# weights = tf.Variable(tf.random_normal([2, 2], stddev=0.35), name='weights')
# w2 = tf.Variable(weights.initialized_value(), name='w2')
#
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     print(sess.run(weights))
#     print(sess.run(w2))







