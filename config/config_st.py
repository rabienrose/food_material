class config_st:
    tfrecord_addr=None
    tfrecord_test_addr=None
    net_type=None
    loss_type=None
    opt_type=None
    accuracy_type=None
    preprocess_type=None
    max_step=None
    debug_step_len=None
    result_addr=None
    stop_accu=None
    batchsize=None
    class_num=None
    is_training=None
    def __init__(self):
        super(config_st, self).__init__()
        self.tfrecord_addr = './output/chamo.tfrecord'
        self.tfrecord_test_addr = './output/chamo_test.tfrecord'
        self.max_step = 100000
        self.debug_step_len = 50
        self.result_addr = './output/chamo.ckpt'
        self.stop_accu=0.01
        self.batchsize=10
        self.class_num=5
        self.preprocess_type = 'default'
        self.net_type = 'mobilenet_v2'
        self.loss_type = 'default'
        self.accuracy_type='default'
        self.opt_type='default'
        self.loading_his=False
        self.is_training=True
