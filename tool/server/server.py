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
import zipfile
import group_img
import data_convertor.img_2_tfrecord_2class
import shutil
import subprocess

app = Flask(__name__)
PROD = True
CORS(app)

result_root='result'
eval_img_root='img_eval'
temp_root='temp'
train_img_root='/home/leo/Downloads/chamo/mat_service/train'
train_tfrecord_root='/home/leo/Downloads/chamo/mat_service/train_tfrecord'
train_temp='/home/leo/Downloads/chamo/mat_service/train_temp'

def get_materil_list():
    folders = os.listdir(train_img_root)
    m_list=[]
    for m in folders:
        m_list.append([m])
    return m_list

def create_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
        os.mkdir(dir)

def check_proc(proc_name, keyword):
    p = subprocess.Popen('ps -ef|grep ' + keyword, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    proc_id='-1'
    for line in p.stdout.readlines():
        line_str = str(line)
        if proc_name in line_str:
            proc_id = line_str.split(' ')[6]
            break
    return proc_id

def kill_proc(proc_name, keyword):
    proc_id=check_proc(proc_name, keyword)
    if proc_id!='-1':
        os.system('echo "1" | sudo -S kill ' + proc_id)
        return True
    return False

def turn_proc(cmd, proc_name, keyword):
    re=kill_proc(proc_name, keyword)
    if re==False:
        os.system(cmd+' &')

@app.route('/api/chamo/img_proc', methods=['POST'])
def handle_api_img_proc():
    ret = dict(code=0, status='OK', payload_err_index=[])
    payload = request.json
    img_payload = payload.get('name', '')
    print(img_payload)
    ret['name'] = 'chamo'
    return jsonify(ret)

@app.route('/api/chamo/img_upload', methods=['POST'])
def handle_api_img_upload():
    create_dir(temp_root)
    create_dir(eval_img_root)
    ret = dict(code=0, status='OK', payload_err_index=[])
    print(request.files)
    file = request.files['fileImg']
    zip_file=temp_root+'/'+file.filename
    file.save(zip_file)
    f_zip = zipfile.ZipFile(zip_file, 'r')
    for f in f_zip.namelist():
        f_zip.extract(f, eval_img_root)
    return jsonify(ret)

@app.route('/api/chamo/train_upload', methods=['POST'])
def handle_api_train_upload():
    create_dir(temp_root)
    create_dir(train_temp)
    ret = dict(code=0, status='OK', payload_err_index=[])
    print(request.files)
    file = request.files['filetrain']
    zip_file=temp_root+'/'+file.filename
    file.save(zip_file)
    f_zip = zipfile.ZipFile(zip_file, 'r')
    for f in f_zip.namelist():
        f_zip.extract(f, train_temp)
    turn_proc('python update_train_imgs.py', 'python', 'update_train_imgs.py')
    return jsonify(ret)

@app.route('/api/chamo/evaluation', methods=['POST'])
def handle_api_evaluation():
    create_dir(result_root)
    ret = dict(code=0, status='OK', payload_err_index=[])
    imgs = os.listdir(eval_img_root)
    img_data_list = []
    for img_name in imgs:
        img_data_list.append(eval_img_root + '/' + img_name)
    group_img.process_img(img_data_list, result_root)
    return jsonify(ret)

@app.route('/api/chamo/get_result', methods=['POST'])
def handle_api_get_result():
    ret = dict(code=0, status='OK', payload_err_index=[])
    imgs = os.listdir(result_root)
    ret['result']=[]
    for img in imgs:
        ret['result'].append(img)
    return jsonify(ret)

@app.route('/api/chamo/conv_tfrecord', methods=['POST'])
def handle_api_conv_tfrecord():
    ret = dict(code=0, status='OK', payload_err_index=[])
    material_dict = {}
    count = 0
    material_list=get_materil_list()
    for item in material_list:
        material_dict[item[0]] = count
        count = count + 1
    data_convertor.img_2_tfrecord_2class.convert_a_folder(train_img_root, train_tfrecord_root, material_dict)
    return jsonify(ret)

@app.route('/api/chamo/train', methods=['POST'])
def handle_api_train():
    ret = dict(code=0, status='OK', payload_err_index=[])
    create_dir('../../output')
    turn_proc('python ../../main.py chamo_class2', 'python', 'main.py')
    return jsonify(ret)

@app.route('/api/chamo/tensorboard', methods=['POST'])
def handle_api_tensorboard():
    ret = dict(code=0, status='OK', payload_err_index=[])
    create_dir('../../logs')
    turn_proc('tensorboard --logdir=./', 'tensorboard', 'logdir')
    return jsonify(ret)

@app.route('/api/chamo/check_proc', methods=['POST'])
def handle_api_check_proc():
    ret = dict(code=0, status='OK', payload_err_index=[])
    payload = request.json
    proc_name = payload.get('proc_name', '')
    keyword = payload.get('keyword', '')
    print(proc_name)
    print(keyword)
    proc_id = check_proc(proc_name, keyword)
    if proc_id=='-1':
        ret['re']=0
    else:
        ret['re'] = 1
    return jsonify(ret)

########################################################
# Web service starting point
########################################################
def start_from_terminal(app):
    parser = optparse.OptionParser()
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=8080)
    parser.add_option(
        '-n', '--name',
        help="name for this flask instance",
        type='str', default="webser0")
    parser.add_option(
        '-l', '--logfile',
        help="log file path",
        type='str', default="./webserv.log")
    opts, args = parser.parse_args()

    handler = TimedRotatingFileHandler(opts.logfile, when="midnight", interval=1)
    handler.suffix = "%Y%m%d"
    logFormatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(logFormatter)

    print("进入HTTP server")
    http_server = HTTPServer(WSGIContainer(app))
    http_server.bind(opts.port)
    http_server.start(num_processes=1)
    print("进入IOloop")
    IOLoop.instance().start()

if __name__ == '__main__':
    start_from_terminal(app)
