from config.config_st import config_st
def get_config():
    config=config_st()
    config.tfrecord_addr='/home/chamo/Documents/data/tfrecord/'
    config.tfrecord_test_addr = '/home/chamo/Documents/data/tfrecord/'
    config.debug_step_len = 100
    config.batchsize=32
    config.loss_type='entropy_loss'
    config.net_type = 'mobilenet_v2'
    config.class_num = 73
    config.result_addr = './output/'
    config.ckpt_name='chamo_150000.000000_0.033395_0.550000'
    config.loading_his = True
    config.accuracy_type = 'multi'
    return config
