from hyperparams import Hyperparams as hp
from preprocess import load_all_dict, sort_user_sequence_dict
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle

def generate_subseq(uid, x_list, users, seqs, labs):
    for i in range(1, len(x_list)):
        users += [uid]
        lab = x_list[-i]
        labs += [lab]
        seqs += [x_list[:-i]]


def generate_seq(uid, iids_seq, users, seqs, labs):
    max_len_with_lab = hp.max_len + 1
    if len(iids_seq) > max_len_with_lab:
        diff = len(iids_seq) - max_len_with_lab
        generate_subseq(uid, iids_seq[:max_len_with_lab], users, seqs, labs)
        for i in range(1, diff + 1):
            users += [uid]
            lab = iids_seq[-i]
            labs += [lab]
            seqs += [iids_seq[-(i + max_len_with_lab -1):-i]]
    else:
        generate_subseq(uid, iids_seq, users, seqs, labs)



def create_data():
    user2idx, idx2user, item2idx, idx2item = load_all_dict(hp.fname)
    user_sequence = sort_user_sequence_dict(hp.fname)
    print "Process Load Dict Done."

    # Make user sequence indexed
    users, seqs, labs = [], [], []
    for k, v in user_sequence.items():
        if k in user2idx:
            user_idx = user2idx.get(k)
            user_idx_seq = [item2idx.get(item[0]) for item in v if item[0] in item2idx]
            generate_seq(user_idx, user_idx_seq, users, seqs, labs)

    # Pad
    X = np.zeros([len(seqs), hp.max_len], np.int32)
    Y = np.zeros([len(labs), 1], np.int32)
    U = np.zeros([len(users), 1], np.int32)
    for i, (x, y, u) in enumerate(zip(seqs, labs, users)):
        X[i] = np.lib.pad(x, [0, hp.max_len - len(x)], 'constant', constant_values=(0, 0))
        Y[i] = y
        U[i] = u
    return X, Y, U


def make_train_test_data():
    X, Y, U = create_data()
    UX = np.concatenate((U, X), axis=1)
    UX_train, UX_test, Y_train, Y_test = train_test_split(UX, Y, test_size=0.33, random_state=0)
    U_train = UX_train[:, 0:1]
    X_train = UX_train[:, 1:]
    U_test = UX_test[:, 0:1]
    X_test = UX_test[:, 1:]
    return X_train, U_train, Y_train, X_test, U_test, Y_test


def save_train_test_data():
    X_train, U_train, Y_train, X_test, U_test, Y_test = make_train_test_data()
    with open('train-test-data/train', 'wb') as fout:
        pickle.dump(X_train, fout)
        pickle.dump(U_train, fout)
        pickle.dump(Y_train, fout)
    with open('train-test-data/test', 'wb') as fout:
        pickle.dump(X_test, fout)
        pickle.dump(U_test, fout)
        pickle.dump(Y_test, fout)

def load_train_test_data():
    with open('train-test-data/train', 'rb') as fout:
        X_train = pickle.load(fout)
        U_train = pickle.load(fout)
        Y_train = pickle.load(fout)
    with open('train-test-data/test', 'rb') as fout:
        X_test = pickle.load(fout)
        U_test = pickle.load(fout)
        Y_test = pickle.load(fout)
    return X_train, U_train, Y_train, X_test, U_test, Y_test

def load_train_data():
    with open('train-test-data/train', 'rb') as fout:
        X_train = pickle.load(fout)
        U_train = pickle.load(fout)
        Y_train = pickle.load(fout)
    return X_train, U_train, Y_train

def load_test_data():
    with open('train-test-data/test', 'rb') as fout:
        X_test = pickle.load(fout)
        U_test = pickle.load(fout)
        Y_test = pickle.load(fout)
    return X_test, U_test, Y_test

def get_batch_data():
    X_train, U_train, Y_train = load_train_data()
    num_batch = len(X_train) // hp.batch_size

    X = tf.convert_to_tensor(X_train, tf.int32)
    U = tf.convert_to_tensor(U_train, tf.int32)
    Y = tf.convert_to_tensor(Y_train, tf.int32)

    a = X.get_shape()

    input_queues = tf.train.slice_input_producer([X, U, Y])

    x, u, y = tf.train.shuffle_batch(input_queues,
                                     num_threads=8,
                                     batch_size=hp.batch_size,
                                     capacity=hp.batch_size*64,
                                     min_after_dequeue=hp.batch_size*32,
                                     allow_smaller_final_batch=False)
    return x, u, y, num_batch


if __name__ == '__main__':
    save_train_test_data()
    X_train, U_train, Y_train, X_test, U_test, Y_test = load_train_test_data()
    x, u, y, num_batch = get_batch_data()

    # user2idx, idx2user, item2idx, idx2item = load_all_dict(hp.fname)
    # y_ = tf.one_hot(y, depth=len(item2idx))
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     print(sess.run(tf.shape(y_)))
    #     print('-----------------------------')
    print 'Done'