import tensorflow as tf

hello = tf.constant('Hello, Tensorflow!')

print(hello)

a = tf.constant(10)
b = tf.constant(20)
c = tf.add(a,b)
d = a + b
print(c)

sess = tf.Session()
print(sess.run(hello), sess.run([a,b,c,d]))

sess.close()