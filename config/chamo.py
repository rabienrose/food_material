from config.config_st import config_st
def get_config():
    config=config_st()
    config.tfrecord_addr='./output/chamo.tfrecord'
    config.tfrecord_test_addr = './output/chamo.tfrecord'
    config.debug_step_len = 10
    config.batchsize=10
    config.loss_type='entropy_loss'
    return config