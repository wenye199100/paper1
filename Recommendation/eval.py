import tensorflow as tf
import numpy as np

from hyperparams import Hyperparams as hp
from data_load import *
from self_attention import *
from preprocess import *
from train import Graph
import codecs

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
                list_of_refs, hypotheses = [], []
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

                    for x, u, y, pred in zip(x_test, u_test, y_test, preds_1):
                        fout.write("--seuqence: " + x + "\n")
                        fout.write("--user: "  + u + "\n")
                        fout.write("--expected: " + y + "\n")
                        fout.write("--predict: " + pred + "\n\n")
                        fout.flush()


