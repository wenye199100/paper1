import tensorflow as tf

from hyperparams import Hyperparams as hp
from data_load import *
from self_attention import *
from preprocess import *

from tqdm import tqdm

class Graph():
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.x, self.u, self.y, self.num_batch = get_batch_data()
            else:
                self.x = tf.placeholder(tf.int32, shape=(None, hp.max_len))
                self.u = tf.placeholder(tf.int32, shape=(None, 1))
                self.y = tf.placeholder(tf.int32, shape=(None, 1))

            user2idx, idx2user, item2idx, idx2item = load_all_dict(hp.fname)


            # Encoder
            with tf.varible_scope("encoder"):
                self.enc = embedding(self.x,
                                     vocab_size=len(item2idx),
                                     num_units=hp.hidden_units,
                                     scale=True,
                                     scope="enc_embed")



                self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                                      vocab_size=hp.max_len,
                                      num_units=hp.hidden_units,
                                      zero_pad=False,
                                      scale=False,
                                      scope="enc_pe")

                self.enc += embedding(tf.tile(self.u, multiples=[1, hp.max_len]),
                                     vocab_size=len(user2idx),
                                     num_units=hp.hidden_units,
                                     scale=True,
                                     scope="enc_user")

                for i in range(hp.num_blocks):
                    with tf.varible_scope("num_blocks_{}".format(i)):
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
                item_lists = tf.range(tf.shape(self.x)[1])
                self.items = embedding(inputs=item_lists,
                                       vocab_size=len(item2idx),
                                       num_units=hp.hidden_units,
                                       zero_pad=False,
                                       scale=False,
                                       scope="items")
                self.BI_scores = tf.matmul(self.C, self.B) # (N, C)
                self.scores = tf.matmul(self.BI_scores, self.items, transpose_b=True) # (N, L)
                self.scores = tf.nn.softmax(self.scores)

                self.preds_1 = tf.to_int32(tf.arg_max(self.scores, dimension=-1))
                self.acc_1 = tf.reduce_sum(tf.to_float(tf.equal(self.preds_1, self.y))) / tf.shape(self.y)[0]

                self.y_rank1 = tf.reshape(self.y, shape=self.y.get_shape()[:-1])
                self.top_5 = tf.nn.top_k(self.scores, 5)
                self.preds_5 = self.top_5[1]
                self.sum_5 = tf.nn.in_top_k(self.scores, self.y_rank1, 5)
                self.acc_5 = tf.reduce_sum(tf.to_float(self.sum_5)) / tf.shape(self.y)[0]

                self.top_10 = tf.nn.top_k(self.scores, 10)
                self.preds_10 = self.top_10[1]
                self.sum_10 = tf.nn.in_top_k(self.scores, self.y_rank1, 10)
                self.acc_10 = tf.reduce_sum(tf.to_float(self.sum_10)) / tf.shape(self.y)[0]

                tf.summary.scalar('acc_1', self.acc_1)
                tf.summary.scalar('acc_5', self.acc_5)
                tf.summary.scalar('acc_10', self.acc_10)

                if is_training:
                    self.y_ = tf.one_hot(self.y, depth=len(item2idx))
                    self.y_smoothed = label_smoothing(self.y_)
                    self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.y_smoothed)
                    self.mean_loss = tf.reduce_sum(self.loss)

                    self.global_step = tf.Variable(0, name='global_step', trainable=False)
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                    self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)

                    tf.summary.scalar('mean_loss', self.mean_loss)
                    self.merged = tf.summary.merge_all()

if __name__ == '__main__':
    user2idx, idx2user, item2idx, idx2item = load_all_dict(hp.fname)

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
            sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))

    print("Done")