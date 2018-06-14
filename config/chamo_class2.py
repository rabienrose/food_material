from config.config_st import config_st
def get_config():
    config=config_st()
    config.tfrecord_addr='E:/data/青椒/tfrecord/'
    config.tfrecord_test_addr = 'E:/data/青椒/tfrecord/'
    config.debug_step_len = 100
    config.batchsize=16
    config.loss_type='class2'
    config.net_type = 'mobilenet_v2'
    config.class_num = 3
    config.result_addr = './output/'
    config.ckpt_name='chamo_150000.000000_0.033395_0.550000'
    config.loading_his = False
    config.accuracy_type = 'class2'
    return config
