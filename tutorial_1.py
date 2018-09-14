import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets("./data/MNIST/",one_hot=True)

print('Size of :')
print('-Training-set "\t\t{}'.format(len(data.train.labels)))
print('-Test-set "\t\t{}'.format(len(data.test.labels)))
print('-Validation-set "\t\t{}'.format(len(data.validation.labels)))

print(data.test.labels[0:5,:])

data.test.cls = np.array([label.argmax() for label in data.test.labels])

print(data.test.cls[0:5])

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size,img_size)

num_classes = 10

def plot_images(images, cls_true, cls_pred = None):
    assert len(images) == len(cls_true) == 9

    fig, axes = plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.3,wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape),cmap='binary')

        if cls_pred is None:
            xlabel = "True:{0}".format(cls_true[i])
        else:
            xlabel = "True:{0}, Pred:{1}".format(cls_true[i],cls_pred[i])

        ax.set_xlabel(xlabel)

        ax.set_xticks([])
        ax.set_yticks([])

images = data.test.images[0:9]

cls_true = data.test.cls[0:9]

plot_images(images=images,cls_true=cls_true)

x = tf.placeholder(tf.float32,[None,img_size_flat])

y_true = tf.placeholder(tf.float32, [None, num_classes])

y_true_cls = tf.placeholder(tf.int64,[None])

weight = tf.Variable(tf.zeros([img_size_flat,num_classes]))

bias = tf.Variable(tf.zeros([num_classes]))

logits = tf.matmul(x,weight)+bias

y_pred = tf.nn.softmax(logits)

y_pred_cls = tf.argmax(y_pred,dimension=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)

cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

correct_prediction = tf.equal(y_pred_cls,y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

sess = tf.Session()

sess.run(tf.global_variables_initializer())

batch_size = 1000

def optimize(num_iterations):
    for i in range(num_iterations):
        x_batch, y_true_batch = data.train.next_batch(batch_size)

        feed_dict_train = {x:x_batch,y_true:y_true_batch}

        sess.run(optimizer,feed_dict=feed_dict_train)

feed_dict_test = {x:data.test.images,
                  y_true:data.test.labels,
                  y_true_cls:data.test.cls}

def print_accuracy():
    acc = sess.run(accuracy,feed_dict=feed_dict_test)

    print("Accuracy on test-set:{0:.1%}".format(acc))

def print_confusion_matrix():
    cls_true = data.test.cls

    cls_pred = sess.run(y_pred_cls,feed_dict=feed_dict_test)

    cm = confusion_matrix(y_true=cls_true,y_pred=cls_pred)

    print(cm)

    # 이미지로서 혼동 행렬을 표시한다.
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # 도표에 대해 다양한 조정을 한다.
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def plot_example_errors():
    # 각 이미지의 예측이 올바른 지와 예측 클래스값을 얻는다.
    correct, cls_pred = sess.run([correct_prediction, y_pred_cls],
                                    feed_dict=feed_dict_test)

    # 잘못 분류된 인덱스를 얻는다
    incorrect = (correct == False)

    # 올바르게 분류되지 않은 이미지를 얻는다
    images = data.test.images[incorrect]

    # 이들 이미지의 예측 클래스를 얻는다
    cls_pred = cls_pred[incorrect]

    # 이들 이미지의 실제 클래스를 얻는다.
    cls_true = data.test.cls[incorrect]

    # 9개의 이미지를 나타낸다.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


def plot_weights():
    # 가중치를 얻는다
    w = sess.run(weight)

    # 가중치의 가장 큰 값과 낮은 값을 구한다.
    # 이것은 이미지를 서로 비교할 수 있도록 색 강도를 조정하는데 사용된다.
    w_min = np.min(w)
    w_max = np.max(w)

    # 3x4 격자의 figure를 만들고 마지막 2개는 사용되지 않는다.
    fig, axes = plt.subplots(3, 4)
    # 격자 간격 조정
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # 첫 10개 subplot만 사용됨
        if i < 10:
            # i번째 숫자에 대한 가중치를 얻고 이를 이미지로 바꾼다.
            # w의 shape는 (img_size_flat, 10)
            image = w[:, i].reshape(img_shape)

            # 각 부분도표의 이름을 붙인다.
            ax.set_xlabel("Weights: {0}".format(i))

            # 이미지로 나타낸다
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')


        # 눈금을 지운다
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


print_accuracy()
plot_example_errors()

optimize(num_iterations=1)
print_accuracy()
plot_example_errors()
plot_weights()


optimize(num_iterations=9)
print_accuracy()
plot_example_errors()
plot_weights()

optimize(num_iterations=990)
print_accuracy()
plot_example_errors()
plot_weights()

print_confusion_matrix()