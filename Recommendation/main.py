import tensorflow as tf

from hyperparams import Hyperparams as hp
from data_load import *
from self_attention import *
from preprocess import *
from eval import *

from tqdm import tqdm

def step_preprocess():
    dataset_name = hp.dataset_name
    type_dict = hp.type_dict
    prefix = hp.prefix
    postfix = hp.postfix

    print "--- Loaded Hyperparams for Dataset. ---"

    fname = hp.fname
    ftype = type_dict[fname]
    fpath = hp.data_fpath + fname + prefix[fname] + postfix[fname]

    file_needed = ["item", "user", "user_dict", "item_dict", "user_sequence_dict", "user_sequence_dict_filtered"]

    if not os.path.exists("preprocessed/{}".format(fname)): os.mkdir("preprocessed/{}".format(fname))

    # check make_vocab
    exist_dataset = 1
    for i in range(len(file_needed)):
        exist_dataset = exist_dataset * os.path.exists("preprocessed/{}/{}".format(fname, file_needed[i]))
    if not exist_dataset:
        print "--- Dataset: {} haven't been processed, start processing. ---".format(hp.fname)

        # make file preprocessed/fname/ -> user, item, user_sequence_dict
        make_vocab(fpath, fname, ftype)
        print "--- Made the user & item cnt file, and unfiltered user sequence dict. ---"

        # make file preprocessed/fname/ -> user_dict, item_dict
        make_all_map(fname)
        print "--- Made the user & item mapping file. ---"

        # make file preprocessed/fname/ -> user_sequence_dict_filtered
        sort_user_sequence_dict(fname)
        print "--- Made filtered user sequence dict. ---"

    # for name in dataset_name:
    #     fname = name
    #     ftype = type_dict[fname]
    #     fpath = hp.data_fpath + fname + prefix[fname] + postfix[fname]
    #
    #     file_needed = ["item", "user", "user_dict", "item_dict", "user_sequence_dict", "user_sequence_dict_filtered"]
    #
    #     if not os.path.exists("preprocessed/{}".format(fname)): os.mkdir("preprocessed/{}".format(fname))
    #
    #     # check make_vocab
    #     exist_dataset = 1
    #     for i in range(len(file_needed)):
    #         exist_dataset = exist_dataset * os.path.exists("preprocessed/{}/{}".format(fname, file_needed[i]))
    #     if not exist_dataset:
    #         print "--- Dataset: {} haven't been processed, start processing. ---".format(hp.fname)
    #
    #         # make file preprocessed/fname/ -> user, item, user_sequence_dict
    #         make_vocab(fpath, fname, ftype)
    #         print "--- Made the user & item cnt file, and unfiltered user sequence dict. ---"
    #
    #         # make file preprocessed/fname/ -> user_dict, item_dict
    #         make_all_map(fname)
    #         print "--- Made the user & item mapping file. ---"
    #
    #         # make file preprocessed/fname/ -> user_sequence_dict_filtered
    #         sort_user_sequence_dict(fname)
    #         print "--- Made filtered user sequence dict. ---"

        print "--- Dataset: {} already been processed. ---".format(hp.fname)

def step_data_load():
    if not os.path.exists('train-test-data/{}'.format(hp.fname)):
        save_train_test_data()
    else:
        exist_split_dataset = 1
        exist_split_dataset = exist_split_dataset * os.path.exists('train-test-data/{}/train'.format(hp.fname))
        exist_split_dataset = exist_split_dataset * os.path.exists('train-test-data/{}/test'.format(hp.fname))
        if not exist_split_dataset:
            save_train_test_data()

    print "--- Dataset: {} already been split into train & test set. ---".format(hp.fname)



def step_train():
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
            sv.saver.save(sess, hp.logdir + '/{}'.format(hp.fname) + '/model_epoch_%02d_gs_%d' % (epoch, gs))

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

if __name__ == '__main__':
    step_preprocess()
    step_data_load()
