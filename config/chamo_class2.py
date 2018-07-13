from config.config_st import config_st
def get_config():
    config=config_st()
    config.tfrecord_addr = '/home/leo/Downloads/chamo/mat_service/train_tfrecord/'
    config.tfrecord_test_addr = '/home/leo/Downloads/chamo/mat_service/test_tfrecord/'
    #config.tfrecord_addr ='/home/leo/Downloads/chamo/v3_material/m_all/all/train1/tfrecord'
    #config.tfrecord_test_addr ='/home/leo/Downloads/chamo/v3_material/m_all/all/test/tfrecord'
    config.debug_step_len = 100
    config.batchsize=32
    config.loss_type='class2'
    config.net_type = 'mobilenet_v2'
    config.class_num = 7
    config.result_addr = '/home/leo/Downloads/chamo/mat_service/checkpoint'
    config.ckpt_name = 'base_cp'
    config.loading_his = True
    config.accuracy_type = 'class2'
    return config
