from config.config_st import config_st
def get_config():
    config=config_st()
    config.tfrecord_addr='./tfrecord/'
    config.tfrecord_test_addr = './tfrecord/refined_test.tfrecord'
    config.debug_step_len = 10
    config.batchsize=10
    config.loss_type='entropy_loss'
    config.net_type = 'mobilenet_v2'
    config.class_num = 20
    config.result_addr = './output/chamo.ckpt'
    return config
