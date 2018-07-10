# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import optparse
import logging
from logging.handlers import TimedRotatingFileHandler
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import yaml
import requests
import datetime
from urllib.parse import quote, unquote
import psutil
from threading import Lock
from copy import copy

app = Flask(__name__)
PROD = True
CORS(app)

COUNTER_LOCK = Lock()
REQ_COUNTER = 0

# define error code
ERROR_CODE_1_INVALID_METHOD = 1
ERROR_CODE_2_PARAM_ERROR = 2
ERROR_CODE_3_PAYLOAD_PERROR = 3
ERROR_CODE_4_PAYLOAD_FERROR = 4
ERROR_CODE_5_PREDICT_ERROR = 5


def _payload_to_imgs(img_payload, payload_format, enable_pp=True):
    ''' 读取、加载payload中指定的一组图像，并转换成一个 PIL.Image对象的列表

    :param img_payload: 一个图像资源的列表 (payload)
    :param format: str, payload中图像资源的格式， 'url', 'base64', 'filepath'
    :return: PIL.Image对象的列表
    '''

    logger = app.config['logger']
    np_img_list = []
    err_idx_list = []

    for idx, obj in enumerate(img_payload):
        if payload_format == 'url':
            _url = unquote(obj)
            np_img = url2img(_url, timeout=app.config['img_req_timeout'])
        elif payload_format == 'base64':
            np_img = url2img(obj, timeout=60)
        else:
            np_img = filepath2img(obj)

        if np_img is not None and isinstance(np_img, np.ndarray) and len(np_img.shape) == 3:
            if enable_pp:
                np_img_list.append(np_img)
            else:
                if np_img.shape == (299, 299, 3):
                    np_img_list.append(np_img)
                else:
                    err_idx_list.append(idx)
                    logger.warning(
                        '_payload_to_imgs: Incorrect image size from payload; image idx in payload: {} shape: {} payload_format: {}'.format(
                            idx, np_img.shape, payload_format))

        else:
            err_idx_list.append(idx)
            logger.warning(
                '_payload_to_imgs: got invalid image from payload; image idx in payload: {} payload_format: {}'.format(
                    idx,
                    payload_format))

    return np_img_list, err_idx_list


def _predict_in_thread(net, req_id, callback_url, top_n, img_payload, payload_format, enable_pp, req_ts):
    ''' 一个线程 worker，由异步API接口调用，执行预测图像类别的任务

    :param net: 预测模型（网络）的一个实例
    :param req_id: str, 客户端指定的请求ID，用于返回消息时传回
    :param top_n: int, 预测结果包括前N个菜品的名称和对应的置信度
    :param img_payload: dict, 图像资源的字典
    :param payload_format: str, "base64", "url", "filepath" 中的一种
    '''

    logger = app.config['logger']

    ret = dict(req_id=req_id, code=0, status='OK', req_timestamp=req_ts)
    logger.info(
        '_predict_in_thread: req_id: {}, callback_url: {}, num images in payload: {}, payload format {}'.format(req_id, callback_url,
                                                                                                                len(img_payload), payload_format))

    t1 = datetime.datetime.now()
    np_img_list, err_idx_list = _payload_to_imgs(img_payload, payload_format)
    t2 = datetime.datetime.now()
    logger.info(
        '_predict_in_thread: req_id: {}, successfully loaded {} images in {} sec'.format(req_id,
                                                                                         len(np_img_list),
                                                                                         (t2 - t1).total_seconds()))
    ret['payload_err_index'] = err_idx_list

    if len(np_img_list) == 0:
        # when no valid image in the payload
        ret['code'] = ERROR_CODE_4_PAYLOAD_FERROR
        ret['status'] = 'Invalid content in the payload. No valid images retrieved.'
        logger.warning('_predict_in_thread: No valid images in payload. payload: {}'.format(img_payload))
    else:
        # when no error or partially error
        if len(err_idx_list) > 0:
            ret['code'] = ERROR_CODE_3_PAYLOAD_PERROR
            logger.warning('_predict_in_thread: Invalid images in payload. index {}'.format(err_idx_list))

        t1 = datetime.datetime.now()
        ret['results'] = net.predict_in_batch(np_img_list=np_img_list, top_n=top_n, enable_pp=enable_pp)
        t2 = datetime.datetime.now()
        logger.info(
            '_predict_in_thread: req_id: {}, batch size: {} inference time {} sec'.format(req_id,
                                                                                          len(np_img_list),
                                                                                          (t2 - t1).total_seconds()))
        logger.debug('_predict_in_thread: inference top N results {}'.format(ret['results']))

    logger.info('_predict_in_thread: req_id {}, POST {} to callback URL: {}'.format(req_id, ret, callback_url))
    try:
        t1 = datetime.datetime.now()
        callback_timeout = app.config['callback_timeout']
        res = requests.post(callback_url, json=ret, timeout=callback_timeout)
        t2 = datetime.datetime.now()
        logger.info(
            '_predict_in_thread: req_id: {}, callback_url {}, callback finished in {} sec'.format(req_id,
                                                                                             callback_url,
                                                                                             (t2 - t1).total_seconds()))
    except requests.exceptions.ConnectionError:
        reason = "A Connection error occurred."
        logger.error(
            '_predict_in_thread: req_id {} callback_url {} details: {} timestamp {}'.format(req_id, callback_url,
                                                                                            reason,
                                                                                            datetime.datetime.now()))
    except requests.exceptions.HTTPError:
        reason = "An HTTP error occurred."
        logger.error(
            '_predict_in_thread: req_id {} callback_url {} details: {} timestamp {}'.format(req_id, callback_url,
                                                                                            reason,
                                                                                            datetime.datetime.now()))
    except requests.exceptions.URLRequired:
        reason = "A valid URL is required to make a request."
        logger.error(
            '_predict_in_thread: req_id {} callback_url {} details: {} timestamp {}'.format(req_id, callback_url,
                                                                                            reason,
                                                                                            datetime.datetime.now()))

    except requests.exceptions.Timeout:
        reason = "The request timed out."
        logger.error(
            '_predict_in_thread: req_id {} callback_url {} details: {} timestamp {}'.format(req_id, callback_url,
                                                                                            reason,
                                                                                            datetime.datetime.now()))
    else:
        logger.info(
            '_predict_in_thread response: server {}, port {}, req_id {}, status_code(POST to callback) {}, content {}, timestamp {}'.format(
                app.config['server_ip'], app.config['server_port'], req_id, res.status_code, ret,
                datetime.datetime.now()))

    global REQ_COUNTER
    COUNTER_LOCK.acquire()
    REQ_COUNTER -= 1
    COUNTER_LOCK.release()

    return

@app.route('/api/imgclassify/dish', methods=['POST'])
def handle_api_classify_dish():
    ''' 处理菜品图像批量识别的同步接口 '''

    ret = dict(code=0, status='OK', payload_err_index=[])
    net = app.config['net']
    supported_formats = app.config['supported_formats']
    logger = app.config['logger']
    logger.debug('request headers: {}'.format(request.headers))
    logger.debug('request environ: {}'.format(request.environ))

    if request.method == 'POST':
        payload = request.json
        img_payload = payload.get('images', [])
        payload_format = str(payload.get('format', ''))
        payload_format = payload_format.lower()
        top_n = payload.get('topN', 5)
        enable_pp = payload.get('preprocess', True)

        logger.debug('handle_api_classify_dish - enable_pp {} payload_format {} top_n{}'.format(enable_pp, payload_format, top_n))
        if payload_format == 'url' or payload_format == 'filepath':
            logger.debug(
                'handle_api_classify_dish - img_payload: {}'.format(img_payload))

        if len(img_payload) == 0 or len(payload_format) == 0 or (payload_format not in supported_formats):
            ret['code'] = ERROR_CODE_2_PARAM_ERROR
            ret['status'] = 'Invalid param in JSON. Keys available in JSON: {}'.format(payload.keys())
            logger.warning('handle_api_classify_dish - {}'.format(ret['status']))
        else:
            logger.info('handle_api_classify_dish - batch size (requested): {}'.format(len(img_payload)))

            t1 = datetime.datetime.now()
            np_img_list, err_idx_list = _payload_to_imgs(img_payload, payload_format, enable_pp)
            t2 = datetime.datetime.now()
            logger.info('handle_api_classify_dish - successfully loaded {} images in {} sec'.format(len(np_img_list), (
                    t2 - t1).total_seconds()))
            ret['payload_err_index'] = err_idx_list

            if len(np_img_list) == 0:
                ret['code'] = ERROR_CODE_4_PAYLOAD_FERROR
                ret['status'] = 'Invalid content in the payload. No valid images retrieved.'
                logger.warning('handle_api_classify_dish: No valid images in payload. Payload {}'.format(payload))
            else:
                if len(err_idx_list) > 0:
                    ret['code'] = ERROR_CODE_3_PAYLOAD_PERROR
                    logger.warning('handle_api_classify_dish: Invalid images in payload. index {}'.format(err_idx_list))

                t1 = datetime.datetime.now()
                ret['results'] = net.predict_in_batch(np_img_list=np_img_list, top_n=top_n, enable_pp=enable_pp)
                t2 = datetime.datetime.now()
                logger.info(
                    'handle_api_classify_dish - batch size: {}, inference time: {} sec'.format(len(np_img_list),
                                                                                               (t2 - t1).total_seconds()))
                logger.debug('handle_api_classify_dish: inference top N results {}'.format(ret['results']))

    else:
        # if request method is NOT POST
        ret['code'] = ERROR_CODE_1_INVALID_METHOD
        ret['status'] = 'Invalid request method. Require POST but get {}'.format(request.method)
        logger.warning('handle_api_classify_dish - {}'.format(ret['status']))

    logger.info(
        'handle_api_classify_dish response: server {}, port {}, content {}, timestamp {}'.format(
            app.config['server_ip'], app.config['server_port'], ret, datetime.datetime.now()))

    return jsonify(ret)


@app.route('/api/async/imgclassify/dish', methods=['POST'])
def handle_async_classify_dish():
    ''' 处理菜品图像批量识别的异步接口 '''
    global REQ_COUNTER
    ret = dict(code=0, status='OK')
    net = app.config['net']
    executor = app.config['executor']
    supported_formats = app.config['supported_formats']
    logger = app.config['logger']
    logger.debug('request headers: {}'.format(request.headers))
    logger.debug('request environ: {}'.format(request.environ))

    if request.method == 'POST':
        payload = request.json
        req_id = payload.get('requestId', '')
        img_payload = payload.get('images', [])
        payload_format = str(payload.get('format', ''))
        payload_format = payload_format.lower()
        top_n = payload.get('topN', 5)
        callback_url = payload.get('callback', None)
        enable_pp = payload.get('preprocess', True)
        req_ts = payload.get('req_timestamp', str(datetime.datetime.now()))

        logger.debug('handle_async_classify_dish - requestId {} enable_pp {} payload_format {} top_n {} callback_url {}'.format(req_id, enable_pp, payload_format, top_n, callback_url))
        if payload_format == 'url' or payload_format == 'filepath':
            logger.debug(
                'handle_async_classify_dish - img_payload: {}'.format(img_payload))

        if len(req_id) == 0 or len(img_payload) == 0 or len(payload_format) == 0 or (
                payload_format not in supported_formats):
            ret['code'] = ERROR_CODE_2_PARAM_ERROR
            ret['status'] = 'Invalid param in JSON. Keys available in JSON: {}'.format(payload.keys())
            logger.warning('handle_async_classify_dish - {}'.format(ret['status']))
        elif callback_url is None:
            ret['code'] = ERROR_CODE_2_PARAM_ERROR
            ret['status'] = 'Invalid param in JSON. Missing callback URL in the param.'
            logger.warning('handle_async_classify_dish - {}'.format(ret['status']))
        else:
            ret['requestId'] = req_id
            executor.submit(_predict_in_thread, net, req_id, callback_url, top_n, img_payload, payload_format,
                            enable_pp, req_ts)
            COUNTER_LOCK.acquire()
            REQ_COUNTER += 1
            COUNTER_LOCK.release()
    else:
        ret['code'] = ERROR_CODE_1_INVALID_METHOD
        ret['status'] = 'Invalid request method. Require POST but get {}'.format(request.method)
        logger.warning('handle_async_classify_dish - {}'.format(ret['status']))

    ret['active_req_num'] = copy(REQ_COUNTER)
    logger.info(
        'handle_async_classify_dish response: server {}, port {}, active_requests {}, content {}, timestamp {}'.format(
            app.config['server_ip'], app.config['server_port'], ret['active_req_num'], ret, datetime.datetime.now()))

    return jsonify(ret)


@app.route('/api/imgclassify/dishlist', methods=['GET'])
def handle_api_get_dishlist():
    ''' 获取当前菜品图片名称和ID的列表'''

    ret = dict(code=0, status='OK')
    net = app.config['net']
    logger = app.config['logger']

    if request.method == 'GET':
        ret['dishlist'] = net.get_dishlist()
    else:
        ret['code'] = ERROR_CODE_1_INVALID_METHOD
        ret['status'] = 'Invalid request method. Require GET but get {}'.format(request.method)
        logger.warning('handle_async_classify_dish - {}'.format(ret['status']))

    return jsonify(ret)


@app.route('/api/imgclassify/version', methods=['GET'])
def handle_api_get_version():
    ''' 获取当前模型的版本号 '''

    ret = dict(code=0, status='OK')
    logger = app.config['logger']
    if request.method == 'GET':
        ret['model_version'] = app.config['model_version']
    else:
        ret['code'] = ERROR_CODE_1_INVALID_METHOD
        ret['status'] = 'Invalid request method. Require GET but get {}'.format(request.method)
        logger.warning('handle_api_get_version - {}'.format(ret['status']))

    return jsonify(ret)


@app.route('/api/check/alive', methods=['GET'])
def handle_api_alive():
    ''' 获取当前服务的健康状态 '''
    global REQ_COUNTER
    mem_info = psutil.virtual_memory()
    status = 'Overload' if (mem_info.percent/100.) > 0.8 or REQ_COUNTER > app.config['max_active_req'] else 'OK'
    ret = dict(status=status, num_active_req=REQ_COUNTER, memory_percent=mem_info.percent)

    resp = jsonify(ret)
    resp.status_code = 429 if status == 'Overload' else 200

    return resp


########################################################
# Util function
########################################################
def load_config(config_file='config.yaml'):
    ''' 读取配置文件，并返回一个python dict 对象

    :param config_file: 配置文件路径
    :return: python dict 对象
    '''
    with open(config_file, 'r') as stream:
        try:
            config = yaml.load(stream)
        except yaml.YAMLError as e:
            print(e)
            return None
    return config


def warm_up():
    ''' Warm up model for the 1st time prediction  '''
    net = app.config['net']
    logger = app.config['logger']
    logger.info('Warming up the mode for the 1st prediction')
    _ = net.predict(img_path='models/img01.jpg')
    logger.info('Finished model warm-up')


def self_test(batch_size=8, num_requests=10):
    t3=datetime.datetime.now()
    ''' Run self test after warm-up, in order to get some performance stats in run-time env '''
    net = app.config['net']
    logger = app.config['logger']
    t4=datetime.datetime.now()

    logger.info('Start self test. send {} requests, batch_size: {}'.format(num_requests, batch_size))
    img_payload = batch_size * ["models/img01.jpg"]
    t1 = datetime.datetime.now()
    for i in range(num_requests):
        np_img_list, err_idx_list = _payload_to_imgs(img_payload, "filepath")
        _ = net.predict_in_batch(np_img_list=np_img_list, top_n=5)
    t2 = datetime.datetime.now()

    total_time = (t2 - t1).total_seconds()
    app.config['t_per_batch'] = total_time / float(num_requests)
    app.config['t_per_img'] = total_time / float(batch_size * num_requests)

    # 30 sec / time per batch-request, e.g. 30 sec/ 0.5 sec = 60 max req in line
    # app.config['max_active_req'] = # max(10, int(app.config['max_latency'] / app.config['t_per_batch']))

    logger.info('Test result: total time used {} sec; time cost per batch {} sec; time cost per image {} sec.'.format(
        total_time, app.config['t_per_batch'], app.config['t_per_img']))
    logger.info('Finished self testing.')


########################################################
# Web service starting point
########################################################
def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=8080)
    parser.add_option(
        '-c', '--config',
        help="config file",
        type='str', default="config.yaml")
    parser.add_option(
        '-n', '--name',
        help="name for this flask instance",
        type='str', default="webser0")
    parser.add_option(
        '-l', '--logfile',
        help="log file path",
        type='str', default="/tmp/webserv.log")
    opts, args = parser.parse_args()

    if os.path.exists(opts.config) and os.path.isfile(opts.config):
        # load config file
        print("进入config")
        print(type(app.config))
        config = load_config(config_file=opts.config)
        # setup logging format and logger
        handler = TimedRotatingFileHandler(opts.logfile, when="midnight", interval=1)
        handler.suffix = "%Y%m%d"
        logFormatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(logFormatter)

        logger = logging.getLogger('DishRecog')
        logger.addHandler(handler)
        logger.setLevel(int(config.get('log_level')))

        # save local env
        logger.info('config: {}'.format(config))
        app.config['logger'] = logger
        app.config['net'] = FoodNet(net_config=config)
        app.config['supported_formats'] = ['url', 'base64', 'filepath']
        app.config['executor'] = ThreadPoolExecutor(max_workers=config.get('async_workers', 4))
        app.config['model_version'] = config.get('model_version', "unknown")
        # app.config['max_latency'] = config.get('max_latency', 30)
        app.config['max_active_req'] = config.get('max_latency', 10)
        app.config['server_id'] = config.get('server_id', "unknown")
        app.config['server_ip'] = config.get('server_ip', "unknown")
        app.config['server_port'] = opts.port
        app.config['img_req_timeout'] = config.get('img_req_timeout', 5.0)  # download image from URL timeout
        app.config['callback_timeout'] = config.get('callback_timeout', 5.0)  # callback POST request timeout

        #warm_up()
        #self_test()
        # start HTTP server
        print("进入HTTP server")
        http_server = HTTPServer(WSGIContainer(app))
        http_server.bind(opts.port)
        http_server.start(num_processes=1)
        print("进入IOloop")
        print(app.config['server_id'])
        print(app.config['server_ip'])
        IOLoop.instance().start()
        print("IOloop开始")
    else:
        exit()


if __name__ == '__main__':
    print("wahahah")
    start_from_terminal(app)
