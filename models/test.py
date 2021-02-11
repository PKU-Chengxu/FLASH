import tensorflow as tf
import numpy as np

labels_reshaped = np.random.randint(0, high=4, size=(5),dtype='i')
logits = tf.constant(np.random.rand(5,4), tf.float32)

unk_tensor = tf.fill(tf.shape(labels_reshaped) , 1)
pred_unk = tf.cast(tf.equal(labels_reshaped, unk_tensor), tf.int32)
labels_rm_unk = tf.add(labels_reshaped, tf.multiply(10 * pred_unk, [tf.shape(logits)[1]]))
pad_tensor = tf.fill(tf.shape(labels_reshaped) , 0)
pred_pad = tf.cast(tf.equal(labels_reshaped, pad_tensor), tf.int32)
labels_rm_pad = tf.add(labels_rm_unk, tf.multiply(10 * pred_pad, [tf.shape(logits)[1]]))

pred_1 = tf.nn.in_top_k(logits, labels_rm_pad, k = 1)
correct_pred_1 = tf.cast(pred_1, tf.int32)
output_1 = tf.count_nonzero(correct_pred_1)



# correct predictions
pred = tf.cast(tf.argmax(logits, 1), tf.int32)
correct_pred = tf.cast(tf.equal(pred, labels_reshaped), tf.int32)

# predicting unknown is always considered wrong
unk_tensor = tf.fill(tf.shape(labels_reshaped), 1)
pred_unk = tf.cast(tf.equal(pred, unk_tensor), tf.int32)
correct_unk = tf.multiply(pred_unk, correct_pred)

# predicting padding is always considered wrong
pad_tensor = tf.fill(tf.shape(labels_reshaped), 0)
pred_pad = tf.cast(tf.equal(pred, pad_tensor), tf.int32)
correct_pad = tf.multiply(pred_pad, correct_pred)

output_2 = tf.count_nonzero(correct_pred) - tf.count_nonzero(correct_unk) - tf.count_nonzero(correct_pad)

with tf.Session() as sess:
    print(labels_reshaped)
    print(sess.run(labels_rm_pad))
    # print(sess.run(logits))
    print('--------------------------')
    print(sess.run(pred_1))
    print(sess.run(correct_pred_1))
    print(sess.run(output_1))
    # print('--------------------------')
    # print(sess.run(correct_pred))
    # print(sess.run(correct_unk))
    # print(sess.run(correct_pad))
    # print(sess.run(output_2))