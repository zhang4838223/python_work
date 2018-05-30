import input_data
import tensorflow as tf
import demo01 as d
#数据预处理
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#开启会话
sess = tf.InteractiveSession()

cross_entropy = -tf.reduce_sum(d.y_ * tf.log(d.y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(d.y_conv, 1), tf.argmax(d.y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())

for i in range(20000) :
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict = {d.x:batch[0], d.y_:batch[1], d.keep_prob:1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={d.x:batch[0], d.y_:batch[1],d.keep_prob:0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={d.x:mnist.test.images, d.y_:mnist.test.labels, d.keep_prob:1.0}))

sess.close()



