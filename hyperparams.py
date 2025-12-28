class Hyperparams:
    '''Hyper parameters'''
    # data
    # train_fpath = './data/full_train_set_2990000.csv'
    train_fpath = './data/sudoku.csv'
    test_fpath = './data/full_test_set_10000.csv'
    
    # model
    num_blocks = 20
    num_filters = 512
    filter_size = 3
    
    # training scheme
    lr = 0.001
    logdir = './logdir/with_resnet/'
    batch_size = 1024
    num_epochs = 5

    result_fpath = './results/output_new_model_with_new_test.csv'