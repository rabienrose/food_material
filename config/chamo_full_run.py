from config.config_st import config_st
def get_config():
    config=config_st()
    config.tfrecord_addr='./tfrecord/refined.tfrecord'
    config.tfrecord_test_addr = './tfrecord/refined_test.tfrecord'
    config.debug_step_len = 50
    config.batchsize=32
    config.loss_type='entropy_loss'
    config.net_type = 'mobilenet_v2'
    config.class_num=4
    config.result_addr = './output/chamo.ckpt'
    return config
