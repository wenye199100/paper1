import tensorflow as tf

from hyperparams import Hyperparams as hp
from data_load import *
from self_attention import *
from preprocess import *
from eval import *

from tqdm import tqdm

class Graph():

    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        self.is_trainning = is_training
        with self.graph.as_default():
            if is_training:
                self.x, self.u, self.y, self.num_batch = get_batch_data()
            else:
                self.x = tf.placeholder(tf.int32, shape=(None, hp.max_len))
                self.u = tf.placeholder(tf.int32, shape=(None, 1))
                self.y = tf.placeholder(tf.int32, shape=(None, 1))

            user2idx, idx2user, item2idx, idx2item = load_all_dict(hp.fname)


            # Encoder
            with tf.variable_scope("encoder"):
                self.enc = embedding(self.x,
                                     vocab_size=(len(item2idx) + 1),
                                     num_units=hp.hidden_units,
                                     scale=True,
                                     scope="enc_embed")



                self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                                      vocab_size=hp.max_len,
                                      num_units=hp.hidden_units,
                                      zero_pad=False,
                                      scale=False,
                                      scope="enc_pe")

                self.user = embedding(tf.tile(self.u, multiples=[1, hp.max_len]),
                                     vocab_size=(len(user2idx)+1),
                                     num_units=hp.hidden_units,
                                     scale=True,
                                     scope="enc_user")
                self.enc += self.user

                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        self.enc = multihead_attention(queries=self.enc,
                                                       keys=self.enc,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=True,
                                                       causality=False,
                                                       relative_mode=False,
                                                       max_relative_position=2
                                                       )
                        self.enc = feedforward(self.enc, num_units=[4*hp.hidden_units, hp.hidden_units])

                # self.C (N, C, T_q) -> (N, C, 1)
                self.C = tf.layers.dense(
                    tf.transpose(self.enc, [0, 2, 1]),1)
                # self.C (N, C, 1) -> (N, C)
                self.C = tf.reshape(self.C, shape=self.C.get_shape()[:-1])

                # self.B (C, C)
                self.B = tf.get_variable("B", [hp.hidden_units, hp.hidden_units])

                # self.items (L, C)
                item_lists = tf.range(len(item2idx))
                self.items = embedding(inputs=item_lists,
                                       vocab_size=len(item2idx),
                                       num_units=hp.hidden_units,
                                       zero_pad=False,
                                       scale=False,
                                       scope="items")
                self.BI_scores = tf.matmul(self.C, self.B) # (N, C)
                self.scores = tf.matmul(self.BI_scores, self.items, transpose_b=True) # (N, L)
                self.scores = tf.nn.softmax(self.scores)

                self.preds_1 = tf.reshape(tf.to_int32(tf.argmax(self.scores, axis=-1)), [tf.shape(self.x)[0],1])
                self.acc_1 = tf.reduce_sum(tf.to_float(tf.equal(self.preds_1, self.y))) / tf.to_float(tf.shape(self.y)[0])

                self.y_rank1 = tf.reshape(self.y, shape=self.y.get_shape()[:-1])
                self.top_5 = tf.nn.top_k(self.scores, 5)
                self.preds_5 = self.top_5[1]
                self.sum_5 = tf.nn.in_top_k(self.scores, self.y_rank1, 5)
                self.acc_5 = tf.reduce_sum(tf.to_float(self.sum_5)) / tf.to_float(tf.shape(self.y)[0])

                self.top_10 = tf.nn.top_k(self.scores, 10)
                self.preds_10 = self.top_10[1]
                self.sum_10 = tf.nn.in_top_k(self.scores, self.y_rank1, 10)
                self.acc_10 = tf.reduce_sum(tf.to_float(self.sum_10)) / tf.to_float(tf.shape(self.y)[0])

                tf.summary.scalar('acc_1', self.acc_1)
                tf.summary.scalar('acc_5', self.acc_5)
                tf.summary.scalar('acc_10', self.acc_10)

                if self.is_training:
                    self.y_ = tf.one_hot(self.y, depth=len(item2idx))
                    self.y_smoothed = label_smoothing(self.y_)
                    self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.y_smoothed)
                    self.mean_loss = tf.reduce_sum(self.loss)

                    self.global_step = tf.Variable(0, name='global_step', trainable=False)
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                    self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)

                    tf.summary.scalar('mean_loss', self.mean_loss)
                    self.merged = tf.summary.merge_all()

    def make_trainable_true(self):
        self.is_trainning = True

    def make_trainable_false(self):
        self.is_trainning = False

if __name__ == '__main__':
    user2idx, idx2user, item2idx, idx2item = load_all_dict(hp.fname)
    if not os.path.exists('results'): os.mkdir('results')
    if not os.path.exists(hp.logdir + '/{}'.format(hp.fname)): os.mkdir(hp.logdir + '/{}'.format(hp.fname))

    g = Graph("train")
    print("Graph loaded")

    sv = tf.train.Supervisor(graph=g.graph,
                             logdir=hp.logdir,
                             save_model_secs=0)
    with sv.managed_session() as sess:
        for epoch in range(1, hp.num_epochs + 1):
            if sv.should_stop(): break
            for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                sess.run(g.train_op)

            gs = sess.run(g.global_step)
            sv.saver.save(sess, hp.logdir+ '/{}'.format(hp.fname) + '/model_epoch_%02d_gs_%d' % (epoch, gs))


            g.make_trainable_false()
            mname = hp.fname + "/epoch_%02d" % epoch
            with codecs.open("results/" + mname, "w", "utf-8") as fout:
                list_of_preds, list_of_expected = [], []
                for i in range(len(X_test) // hp.batch_size):

                    x_test = X_test[i * hp.batch_size: (i + 1) * hp.batch_size]
                    u_test = U_test[i * hp.batch_size: (i + 1) * hp.batch_size]
                    y_test = Y_test[i * hp.batch_size: (i + 1) * hp.batch_size]

                    preds = np.zeros((hp.batch_size, 1), np.int32)

                    # _preds = sess.run(g.preds_1, {g.x: x_test, g.u: u_test, g.y: preds})
                    # preds_1 = _preds
                    #
                    # _preds = sess.run(g.preds_5, {g.x: x_test, g.u: u_test, g.y: preds})
                    # preds_5 = _preds

                    _preds = sess.run(g.preds_10, {g.x: x_test, g.u: u_test, g.y: preds})
                    preds_10 = _preds

                    for x, u, y, pred in zip(x_test, u_test, y_test, preds_10):
                        fout.write("--seuqence: " + x + "\n")
                        fout.write("--user: " + u + "\n")
                        fout.write("--expected: " + y + "\n")
                        fout.write("--predict: " + pred + "\n\n")
                        fout.flush()

                        list_of_expected.append(y)
                        list_of_preds.append(pred)

                f1_score = caculate_f1_score(list_of_expected, list_of_preds)
                fout.write("F1 Score = " + f1_score)
                ndcg_score = caculate_ndcg(list_of_expected, list_of_preds)
                fout.write("NDCG = " + ndcg_score)

            g.make_trainable_true()

    print("Done")