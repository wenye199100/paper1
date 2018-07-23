class Hyperparams:

    #data path
    data_fpath = '/home/yewenwen/Project/Paper1/self-attention/Recommendation/data/Musical_Instruments_5.json'
    data_train = 'data/train'
    data_test = 'data/test'
    fname = 'vocab'

    #training
    batch_size = 32
    lr = 0.0001
    logdir = 'logdir'

    #model
    max_len = 5
    min_user_cnt = 10
    min_item_cnt = 10
    hidden_units = 512
    num_blocks = 6
    num_epochs = 20
    num_heads = 8
    drop_rate = 0.1



