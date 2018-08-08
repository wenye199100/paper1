class Hyperparams:

    #data path
    data_fpath = 'data/'
    data_train = 'data/train'
    data_test = 'data/test'
    fname = 'ratings_Musical_Instruments'

    dataset_name = ["movielens", "Musical_Instruments_5", "ratings_Musical_Instruments"]
    type_dict = {"movielens": 0, "Musical_Instruments_5": 1, "ratings_Musical_Instruments":2}
    prefix = {"movielens": "/", "Musical_Instruments_5": "/Musical_Instruments_5", "ratings_Musical_Instruments": "/ratings_Musical_Instruments"}
    postfix = {"movielens": "ratings.dat", "Musical_Instruments_5": ".json", "ratings_Musical_Instruments":".csv"}



    #training
    batch_size = 32
    lr = 0.0001
    logdir = 'logdir'

    #model
    max_len = 100
    min_user_cnt = 10
    min_item_cnt = 10
    hidden_units = 512
    num_blocks = 6
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.1



