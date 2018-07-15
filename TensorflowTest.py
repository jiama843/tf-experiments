import tensorflow as tf

#Constants
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)

#Placeholders (Used for inputs)
node3 = tf.placeholder(tf.float32)
node4 = tf.placeholder(tf.float32)

node34 = node3 + node4
#print(node1, node2)


#linear model
m = tf.Variable([2.0], dtype=tf.float32)
b = tf.Variable([3.0], dtype=tf.float32)
x = tf.placeholder(dtype=tf.float32)
y = m*x + b

#Cost Functions
yE = tf.placeholder(dtype=tf.float32)
squared_diffs = tf.square(y - yE)
cost = tf.reduce_sum(squared_diffs)

#Change variables
changeM = tf.assign(m, [1.0])
changeB = tf.assign(b, [1.0])

#Sessions
sess = tf.Session()

#Used to initialize variables
init = tf.global_variables_initializer()
sess.run(init)

sess.run([changeM, changeB]);

print(sess.run([node1, node2]))
print(sess.run(node34, {node3: 9.9, node4: 1.1}))

print(sess.run(y, {x: [1.0]}))

#Like this: {arg1, arg2}
print(sess.run(cost, {x:[1.0, 2.0, 3.0], yE: [3.0, 4.0, 5.0]}))