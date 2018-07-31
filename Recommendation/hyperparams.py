class Hyperparams:

    #data path
    data_fpath = 'data/'
    data_train = 'data/train'
    data_test = 'data/test'
    fname = 'Musical_Instruments_5'

    dataset_name = ["movielens", "Musical_Instruments_5"]
    type_dict = {"movielens": 0, "Musical_Instruments_5": 1}
    prefix = {"movielens": "/", "Musical_Instruments_5": "/Musical_Instruments_5"}
    postfix = {"movielens": "ratings.dat", "Musical_Instruments_5": ".json"}



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
    dropout_rate = 0.1



