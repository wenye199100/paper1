import numpy as np
import tensorflow as tf


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
    relative(5,2)
    print 'haha'