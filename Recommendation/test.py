import numpy as np
import tensorflow as tf

def argmax():
    input = tf.constant(np.random.rand(3,4))
    with tf.Session() as sess:
        print sess.run(input)
        print sess.run(tf.argmax(input,1))

def top_test():
    input = tf.constant(np.random.rand(3,4))
    k = 2
    output = tf.nn.top_k(input, k)
    input = tf.to_float(input)
    index = tf.constant([[3],[3],[3]])
    index = tf.reshape(index, shape=index.get_shape()[:-1])
    output2 = tf.nn.in_top_k(input,index,2)
    a = tf.reduce_sum(tf.to_float(output2))

    with tf.Session() as sess:
        print sess.run(input)
        print sess.run(output)
        print sess.run(a)

    return output

def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=False,
              scale=None,
              scope='embedding',
              reuse=None):

    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)
    return outputs

def aa3():
    item_lists = tf.range(10)
    items = embedding(inputs=item_lists,
                           vocab_size=10,
                           num_units=64,
                           zero_pad=False,
                           scale=False,
                           scope="items")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(tf.shape(items)))
        print('-----------------------------')

def aa2():
    x = tf.Variable(tf.random_normal(shape=(4, 3, 2)))
    y = tf.range(tf.shape(x)[1])
    y_ = y.get_shape()
    z = tf.expand_dims(y, 0)
    z_ = z.get_shape()
    c = tf.tile(z, [tf.shape(x)[0], 2])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(x))
        print('-----------------------------')
        print(sess.run(y))
        print(y_)
        print('-----------------------------')
        print(sess.run(z))
        print(z_[1])
        print('-----------------------------')
        print(sess.run(c))
        print('-----------------------------')

def aa1():
    raw = tf.Variable(tf.random_normal(shape=(4, 3, 2)))
    raw2 = tf.Variable(tf.random_normal(shape=(2,5)))
    raw3 = tf.Variable(tf.random_normal(shape=(2,5,1)))
    raw4 = tf.Variable(tf.random_normal(shape=(2, 5)))
    transed_1 = tf.transpose(raw, perm=[1, 0, 2])
    transed_2 = tf.transpose(raw, perm=[2, 0, 1])
    transed_3 = tf.transpose(raw)
    raw5 = tf.reshape(raw3,shape=raw3.get_shape()[:-1])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(raw3.get_shape()[:-1])
        print('-----------------------------')
        print(raw3.eval())
        # print(raw.eval())
        print('-----------------------------')
        print(raw5.eval())
        # print('-----------------------------')
        # print(sess.run(transed_1))
        # print('-----------------------------')
        # print(sess.run(transed_2))
        # print('-----------------------------')
        # print(sess.run(transed_3))

def relative(length, max_relative_position):
    range_vec = tf.range(length)
    range_mat = tf.reshape(tf.tile(range_vec, [length]), [length, length])
    distance_mat = range_mat - tf.transpose(range_mat)
    print distance_mat


def lalala():
    print (-2**32+1)
    a = np.arange(12).reshape(3,4)
    print a
    print np.array_split(a, 2,axis = 1)

    list =[]
    iid = 'asin'
    time = 'unixReviewTime'
    score = 'overall'
    item_score_time1 = ['iid1', score, 3]
    item_score_time2 = ['iid2', score, 1]
    item_score_time3 = ['iid3', score, 4]
    item_score_time4 = ['iid4', score, 2]
    list.append(item_score_time1)
    list.append(item_score_time2)
    list.append(item_score_time3)
    list.append(item_score_time4)
    list.sort(key= lambda time :time[2])

    for i in range(0,4):
        print i
    list_a = [1,2,3,4,5,6,7,8,9]
    print list_a[1:0]
    out_seqs = []
    labs = []
    for i in range(1, len(list_a)):
        tar = list_a[-i]
        labs += [tar]
        out_seqs += [list_a[:-i]]
    print list

if __name__ == '__main__':
    argmax()