import tensorflow as tf
import numpy as np

from hyperparams import Hyperparams as hp
from data_load import *
from self_attention import *
from preprocess import *
from train import Graph
import codecs
import math


def caculate_f1_score(list_of_expected, list_of_preds):
    precision, recall, f1_score = [], [], []
    sum_1, sum_5, sum_10 = 0.0, 0.0, 0.0
    for expected, preds in zip(list_of_expected, list_of_preds):
        sum_1 += float(expected in preds[:1])
        sum_5 += float(expected in preds[:5])
        sum_10 += float(expected in preds)
    precision.append(sum_1 / float(len(expected) * 1))
    precision.append(sum_5 / float(len(expected) * 5))
    precision.append(sum_10 / float(len(expected) * 10))

    recall.append(sum_1 / float(len(expected) * 1))
    recall.append(sum_5 / float(len(expected) * 1))
    recall.append(sum_10 / float(len(expected) * 1))

    f1_score = map(lambda (a,b): 2 * a * b, zip(precision, recall)) / map(lambda (a,b): a + b, zip(precision, recall))
    return f1_score

def caculate_ndcg(list_of_expected, list_of_preds):
    ndcg = []
    sum_1, sum_5, sum_10 = 0.0, 0.0, 0.0
    for expected, preds in zip(list_of_expected, list_of_preds):
        if expected in preds[:1]:
            sum_1 += 1

        if expected in preds[:5]:
            sum_5 += 1.0 / math.log((float(preds.index(expected) + 1)), 2)

        if expected in preds[:10]:
            sum_10 += 1.0 / math.log((float(preds.index(expected) + 1)), 2)
    ndcg.append(sum_1 / float(len(expected) * 1))
    ndcg.append(sum_5 / float(len(expected) * 1))
    ndcg.append(sum_10 / float(len(expected) * 1))
    return ndcg


def eval():
    g = Graph(is_training=False)
    print "Graph loaded"

    # load test data
    X_test, U_test, Y_test = load_test_data()
    user2idx, idx2user, item2idx, idx2item = load_all_dict(hp.fname)

    with g.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ## Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")

            ## Get model name
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1]  # model name

            ## Inference
            if not os.path.exists('results'): os.mkdir('results')
            with codecs.open("results/" + mname, "w", "utf-8") as fout:
                list_of_preds, expected = [], []
                for i in range(len(X_test) // hp.batch_size):

                    x_test = X_test[i * hp.batch_size: (i+1) * hp.batch_size]
                    u_test = U_test[i * hp.batch_size: (i+1) * hp.batch_size]
                    y_test = Y_test[i * hp.batch_size: (i+1) * hp.batch_size]

                    preds = np.zeros((hp.batch_size, 1), np.int32)

                    _preds = sess.run(g.preds_1, {g.x: x_test, g.u: u_test, g.y: preds})
                    preds_1 = _preds
                    
                    _preds = sess.run(g.preds_5, {g.x: x_test, g.u: u_test, g.y: preds})
                    preds_5 = _preds

                    _preds = sess.run(g.preds_10, {g.x: x_test, g.u: u_test, g.y: preds})
                    preds_10 = _preds

                    for x, u, y, pred in zip(x_test, u_test, y_test, preds_10):
                        fout.write("--seuqence: " + x + "\n")
                        fout.write("--user: "  + u + "\n")
                        fout.write("--expected: " + y + "\n")
                        fout.write("--predict: " + pred + "\n\n")
                        fout.flush()

                        list_of_expected.append(y)
                        list_of_preds.append(pred)

                # Calculate precision, recall and map




