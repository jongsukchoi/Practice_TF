import tensorflow as tf

X = tf.placeholder(tf.float32,[None, 5])
print(X)

x_data = [[1,2,3,4,5],[6,7,8,9,10]]

W = tf.Variable(tf.random_normal([5,2]))
b = tf.Variable(tf.random_normal([2,1]))

hyper = tf.matmul(X,W)+b

sess = tf.Session()

sess.run(tf.global_variables_initializer())

print("--- x_data ---")
print(x_data)
print("--- W ---")
print(sess.run(W))
print("--- b ---")
print(sess.run(b))
print("--- expr ---")

print(sess.run(hyper, feed_dict={X: x_data}))

sess.close()