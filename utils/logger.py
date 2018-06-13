import logging
import config.config_st
import os

config_obj = config.config_st.config_st()
log_path = config_obj.program_log_path

hello_log = log_path + 'hello.log'

logger = logging.getLogger(__name__)


def init():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not os.path.exists(os.path.dirname(hello_log)):
        os.makedirs(os.path.dirname(hello_log))
    if not os.path.exists(hello_log):
        f = open(hello_log, 'w')
        f.close()
    handler = logging.FileHandler(hello_log)
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter('%(asctime)s -  %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)


def I(TAG, message):
    output = TAG + ' - ' + message
    logger.info(output)


def D(TAG, message):
    output = TAG + ' - ' + message
    logger.debug(output)


def W(TAG, message):
    output = TAG + ' - ' + message
    logger.warning(output)


def E(TAG, message):
    output = TAG + ' - ' + message
    logger.error(output)


def CRITICAL(TAG, message):
    output = TAG + ' - ' + message
    logger.critical(output)


if __name__ == '__main__':
    init()
    for i in range(5):
        I('TAG', 'test')
