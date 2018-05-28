class config_st:
    tfrecord_addr=None
    net_type=None
    loss_type=None
    opt_type=None
    preprocess_type=None
    def __init__(self):
        super(config_st, self).__init__()
        self.tfrecord_addr = './data_preprocessing/chamo.tfrecord'
        self.preprocess_type = 'default'
        self.net_type = 'vgg16'
        self.loss_type = 'default'
        self.opt_type='default'