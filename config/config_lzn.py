from config.config_st import config_st
def get_config():
    config=config_st()
    config.tfrecord_addr='/home/leo/Downloads/chamo/tfrecord/'
    config.tfrecord_test_addr = '/home/leo/Downloads/chamo/tfrecord/'
    config.debug_step_len = 100
    config.batchsize=32
    config.loss_type='entropy_loss'
    config.net_type = 'mobilenet_v2'
    config.class_num = 102
    config.result_addr = './output/'
    config.loading_his = False
    return config
