import tensorflow as tf

m = tf.Variable([3], dtype=tf.float32)
b = tf.Variable([2], dtype=tf.float32)
x = tf.placeholder(tf.float32)
y = m*x + b

yE = tf.placeholder(dtype=tf.float32)
squared_change = tf.square(y - yE)
cost = tf.reduce_sum(squared_change)

optimizer = tf.train.GradientDescentOptimizer(0.01)
trainer = optimizer.minimize(cost)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

for i in range(10):
    sess.run(trainer, {x: [1.0, 3.4, 2.3, 9.0, 7.2, 8.9], yE: [1.0, 5.7, 3.7, 10.2, 8.2, 9.0]})

print(sess.run([m, b]))