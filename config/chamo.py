from config.config_st import config_st
def get_config():
    config=config_st()
    config.tfrecord_addr='./data_convertor/chamo.tfrecord'
    config.loss_type='entropy_loss'
    return config