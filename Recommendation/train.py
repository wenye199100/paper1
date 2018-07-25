import tensorflow as tf

from hyperparams import Hyperparams as hp
from data_load import *
from self_attention import *
from prepocess import *

from tqdm import tqdm

class Graph():
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.X_train, self.U_train, self.Y_train, self.X_test, self.U_test, self.Y_test = make_train_test_data()
            user2idx, idx2user, item2idx, idx2item = load_all_dict(hp.fname)
            if is_training:
                self.x, self.u, self.y, self.num_batch = get_batch_data(self.X_train, self.U_train, self.Y_train)
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

                self.enc += embedding(self.u,
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
                BI_scores = tf.matmul(self.C, self.B) # (N, C)
                self.scores = tf.matmul(BI_scores, self.items, transpose_b=True)
                self.scores = tf.nn.softmax(self.scores)

                if is_training:
                    
                    # Loss