import numpy as np
import tensorflow as tf

class weightsHLayer1:

    def _init_(self, w11, w12, w13, w14, w15, w16, w17, w18, w19, w21, w22, w23, w24, w25, w26, w27, w28, w29):
        self.w11 = w11
        self.w12 = w12
        self.w13 = w13
        self.w14 = w14
        self.w15 = w15
        self.w16 = w16
        self.w17 = w17
        self.w18 = w18
        self.w19 = w19

    def get_w():
         return w11


#define Model
def tic_tac_model_fn(features, label, mode):

    x = tf.placeholder(tf.float32, shape=[1, 18])
    y = tf.placeholder(tf.float32, shape=[1, 9])

    #weights for first nodes
    w1h1 = tf.Variable([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=)


    #hidden layers (1)

    h1 = tf.sigmoid((w11 * x11) + (w12 * x12) + (w13 * x13) + (w14 * x14) + (w15 * x15) +
                    (w16 * x16) + (w17 * x17) + (w18 * x18) + (w19 * x19) +
                    (w21 * x21) + (w22 * x22) + (w23 * x23) + (w24 * x24) + (w25 * x25) +
                    (w26 * x26) + (w27 * x27) + (w28 * x28) + (w29 * x29))

    h2 = tf.sigmoid((w11 * x11) + (w12 * x12) + (w13 * x13) + (w14 * x14) + (w15 * x15) +
                    (w16 * x16) + (w17 * x17) + (w18 * x18) + (w19 * x19) +
                    (w21 * x21) + (w22 * x22) + (w23 * x23) + (w24 * x24) + (w25 * x25) +
                    (w26 * x26) + (w27 * x27) + (w28 * x28) + (w29 * x29))

    h3 = tf.sigmoid((w11 * x11) + (w12 * x12) + (w13 * x13) + (w14 * x14) + (w15 * x15) +
                    (w16 * x16) + (w17 * x17) + (w18 * x18) + (w19 * x19) +
                    (w21 * x21) + (w22 * x22) + (w23 * x23) + (w24 * x24) + (w25 * x25) +
                    (w26 * x26) + (w27 * x27) + (w28 * x28) + (w29 * x29))

    h4 = tf.sigmoid((w11 * x11) + (w12 * x12) + (w13 * x13) + (w14 * x14) + (w15 * x15) +
                    (w16 * x16) + (w17 * x17) + (w18 * x18) + (w19 * x19) +
                    (w21 * x21) + (w22 * x22) + (w23 * x23) + (w24 * x24) + (w25 * x25) +
                    (w26 * x26) + (w27 * x27) + (w28 * x28) + (w29 * x29))

    h5 = tf.sigmoid((w11 * x11) + (w12 * x12) + (w13 * x13) + (w14 * x14) + (w15 * x15) +
                    (w16 * x16) + (w17 * x17) + (w18 * x18) + (w19 * x19) +
                    (w21 * x21) + (w22 * x22) + (w23 * x23) + (w24 * x24) + (w25 * x25) +
                    (w26 * x26) + (w27 * x27) + (w28 * x28) + (w29 * x29))

    h6 = tf.sigmoid((w11 * x11) + (w12 * x12) + (w13 * x13) + (w14 * x14) + (w15 * x15) +
                    (w16 * x16) + (w17 * x17) + (w18 * x18) + (w19 * x19) +
                    (w21 * x21) + (w22 * x22) + (w23 * x23) + (w24 * x24) + (w25 * x25) +
                    (w26 * x26) + (w27 * x27) + (w28 * x28) + (w29 * x29))

    h7 = tf.sigmoid((w11 * x11) + (w12 * x12) + (w13 * x13) + (w14 * x14) + (w15 * x15) +
                    (w16 * x16) + (w17 * x17) + (w18 * x18) + (w19 * x19) +
                    (w21 * x21) + (w22 * x22) + (w23 * x23) + (w24 * x24) + (w25 * x25) +
                    (w26 * x26) + (w27 * x27) + (w28 * x28) + (w29 * x29))

    h8 = tf.sigmoid((w11 * x11) + (w12 * x12) + (w13 * x13) + (w14 * x14) + (w15 * x15) +
                    (w16 * x16) + (w17 * x17) + (w18 * x18) + (w19 * x19) +
                    (w21 * x21) + (w22 * x22) + (w23 * x23) + (w24 * x24) + (w25 * x25) +
                    (w26 * x26) + (w27 * x27) + (w28 * x28) + (w29 * x29))

    h9 = tf.sigmoid((w11 * x11) + (w12 * x12) + (w13 * x13) + (w14 * x14) + (w15 * x15) +
                    (w16 * x16) + (w17 * x17) + (w18 * x18) + (w19 * x19) +
                    (w21 * x21) + (w22 * x22) + (w23 * x23) + (w24 * x24) + (w25 * x25) +
                    (w26 * x26) + (w27 * x27) + (w28 * x28) + (w29 * x29))


    return tf.estimator.EstimatorSpec(
        mode = mode,
        #predictions = ,
        #loss = loss ,
        #train_op = train
    )



#inference

    #define input data


    #define input nodes

    #define neural network nodes

#loss

    #initialize cost function

#Training

    #Start session

    #Training loop

    #Comparing Output/Backpropogation

    #Saving checkpoints

