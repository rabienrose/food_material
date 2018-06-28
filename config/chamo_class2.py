from config.config_st import config_st
def get_config():
    config=config_st()
    config.tfrecord_addr='/home/leo/Downloads/chamo/v3_material/tfrecord/'
    config.tfrecord_test_addr = '/home/leo/Downloads/chamo/v3_material/tfrecord/'
    config.debug_step_len = 100
    config.batchsize=32
    config.loss_type='class2'
    config.net_type = 'mobilenet_v2'
    config.class_num = 3
    config.result_addr = './output/'
    config.ckpt_name='/chamo_178000.000000_0.037930'
    config.loading_his = True
    config.accuracy_type = 'class2'
    return config
