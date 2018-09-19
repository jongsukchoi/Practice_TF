import tensorflow as tf
import numpy as np

x_data = np.array([[0,0],[1,0],[1,1],[0,0],[0,0],[0,1]])

y_data = np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0],[1,0,0],[0,0,1]])

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([2,3],-1.,1.))
b = tf.Variable(tf.zeros([3]))

Layer_1 = tf.add(tf.matmul(X,W),b)

Act_Layer_1 = tf.nn.relu(Layer_1)

model = tf.nn.softmax(Act_Layer_1)

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(model),axis=1))
#cost = tf.reduce_mean(tf.square(model - Y))
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=L),axis=0)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(100):
    sess.run(optimizer,feed_dict={X:x_data,Y:y_data})
    if step % 10 == 9:
        print(step, sess.run(cost,feed_dict={X:x_data,Y:y_data}))


prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
print('실제값:', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))

