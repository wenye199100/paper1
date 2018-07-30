# -*- coding: utf-8 -*-
import json
import os
import pickle
import gzip
import codecs
from collections import Counter
from hyperparams import Hyperparams as hp

def make_vocab(fpath, fname):
    """
    读取数据文件之后处理
    1 save <item, cnt> & <user, cnt> into fname_item & fname_user
    2 save <user, item, rating, time> into fname_user_sequence_dict

    :param fpath: 数据文件存储的位置
    :param fname: 处理后文件的前缀
    :return:
    """
    user_vocab = []
    item_vocab = []
    user_item_score_time = {}
    with open(fpath) as json_file:
        for line in json_file:
            unit = json.loads(line)
            uid = unit['reviewerID']
            iid = unit['asin']
            time = unit['unixReviewTime']
            score = unit['overall']
            item_score_time = [iid, score, time]
            if uid in user_item_score_time.keys():
                user_item_score_time[uid].append(item_score_time)
            else:
                tlist = []
                tlist.append(item_score_time)
                user_item_score_time[uid] = tlist
            user_vocab.append(uid)
            item_vocab.append(iid)
        user2cnt = Counter(user_vocab)
        item2cnt = Counter(item_vocab)
    if not os.path.exists('preprocessed'): os.mkdir('preprocessed')
    with open('preprocessed/{}_user'.format(fname), 'w') as fout:
        for user, cnt in user2cnt.most_common(len(user2cnt)):
            fout.write(u"{}\t{}\n".format(user, cnt))
    with open('preprocessed/{}_item'.format(fname), 'w') as fout:
        for item, cnt in item2cnt.most_common(len(item2cnt)):
            fout.write(u"{}\t{}\n".format(item, cnt))
    with open('preprocessed/{}_user_sequence_dict'.format(fname), 'wb') as fout:
        pickle.dump(user_item_score_time, fout)

def make_all_map(fname):
    make_mapping(fname, "user", hp.min_user_cnt)
    make_mapping(fname, "item", hp.min_item_cnt)

def load_all_dict(fname):
    user2idx, idx2user = load_dict(fname, "user")
    item2idx, idx2item = load_dict(fname, "item")
    return user2idx, idx2user, item2idx, idx2item

def load_dict(fname, type):
    with open('preprocessed/{}_{}_dict'.format(fname, type), 'rb') as fout:
        word2idx = pickle.load(fout)
        idx2word = pickle.load(fout)
    return word2idx, idx2word


def make_mapping(fname, type, limit_num):
    """
    将make_vocab里得到的<type, cnt>转换成用户或者商品字典
    save the dict in to fname_type_dict----> fname_user_dict / fname_item_dict
    :param fname:
    :param type: user or item
    :param limit_num: pass user/item if cnt < limet
    :return:
    """
    vocab = []
    with open("preprocessed/{}_{}".format(fname, type)) as lines:
        for line in lines:
            if(int(line.split()[1]) >= limit_num):
                vocab.append(line.split()[0])
    # vocab = [line.split()[0] for line in open(fname) if int(line.split()[1] >= 40)]
    word2idx = {word: (idx + 1) for idx, word in enumerate(vocab)}
    idx2word = {(idx + 1): word for idx, word in enumerate(vocab)}
    with open('preprocessed/{}_{}_dict'.format(fname, type), 'wb') as fout:
        pickle.dump(word2idx, fout)
        pickle.dump(idx2word, fout)
        pickle.dump(type, fout)


def sort_user_sequence_dict(fname):
    with open('preprocessed/{}_user_sequence_dict'.format(fname), 'rb') as fout:
        user_item_score_time = pickle.load(fout)
        user_item_score_time_filtered = {}
        user2idx, idx2user = load_dict(fname, "user")
        for k, v in user_item_score_time.items():
            if k in user2idx:
                v.sort(key = lambda time: time[2])
                user_item_score_time_filtered[k] = v
    return user_item_score_time_filtered




if __name__ == '__main__':
    fname = hp.fname
    dataset_name=["movielens","amazon"]
    fname = dataset_name[0]
    if not os.path.exists("preprocessed/{}".format(fname)): os.mkdir("preprocessed/{}".format(fname))

    # check make_vocab
    if not os.path.exists("preprocessed/{}/user".format(fname)):
        






    select = str(0)
    while select == str(0):
        select = raw_input("0.Change fname.(Default is '{}')\n"
                       "1.Count User/Item and Delete Element Less Than Limit Number (Saved).\n"
                       "2.Mapping Word to Id.(Pickle Saved)\n"
                       "3.Load From Pickle Files.\n"
                       "4.Sort User Sequence Dict.\n"
                       "Please input:>>>".format(fname))
        if select == str(0):
            fname = raw_input("Input new fname >>>")
    if select == str(1):
        make_vocab(hp.data_fpath, fname)
        print("Process User/Item to Cnt Done.")
    elif select == str(2):
        make_all_map(fname)
        print "Process Dict Done."
    elif select == str(3):
        user2idx, idx2user, item2idx, idx2item = load_all_dict(fname)
        print "Process Load Dict Done."
    elif select == str(4):
        sort_user_sequence_dict(fname)
    else:
        print "Done."