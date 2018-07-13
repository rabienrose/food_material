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
import tool.scrap_img.scrap_img
import tool.scrap_img.scrap_sougou
import data_convertor.process_img
import tool.server.update_train_imgs
import data_scraping.materil_name

app = Flask(__name__)
PROD = True
CORS(app)

data_root='/home/leo/Downloads/chamo/mat_service'
result_root=data_root+'/result'
eval_img_root=data_root+'/img_eval'
temp_root=data_root+'/temp'
train_img_root=data_root+'/train'
test_img_root=data_root+'/test'
train_tfrecord_root=data_root+'/train_tfrecord'
test_tfrecord_root=data_root+'/test_tfrecord'
upload_temp=data_root+'/upload_temp'
upload_cache=data_root+'/upload_cache'
scrap_root=data_root+'/scrap_re'
result_group=data_root+'/result_group'
base_cp=data_root+'/checkpoint/base_cp'
cur_cp=data_root+'/checkpoint/cur_cp'
release_cp=data_root+'/checkpoint/release_cp'
logs=data_root+'/checkpoint/logs'
cur_material_file=data_root + '/materials.txt'
release_material_file=data_root + '/materials_release.txt'

def get_materil_list(is_release):
    file_name=cur_material_file
    if is_release:
        file_name=release_material_file
    with open(file_name, 'r') as f:
        m_list = []
        for line in f.readlines():
            m_list.append([line[0:-1]])
    return m_list

def save_materil_list():
    with open(cur_material_file, 'w') as f:
        folders = os.listdir(train_img_root)
        for m in folders:
            f.write(m+"\n")

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
            splited=line_str.split(' ')
            print(splited)
            for item in splited:
                if item.isdigit():
                    proc_id=item
                    break
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

def zip_dir(dirname,zipfilename):
    filelist = []
    if os.path.isfile(dirname):
        filelist.append(dirname)
    else :
        for root, dirs, files in os.walk(dirname):
            for dir in dirs:
                filelist.append(os.path.join(root,dir))
            for name in files:
                filelist.append(os.path.join(root, name))

    zf = zipfile.ZipFile(zipfilename, "w", zipfile.zlib.DEFLATED)
    for tar in filelist:
        arcname = tar[len(dirname):]
        #print arcname
        zf.write(tar,arcname)
    zf.close()

@app.route('/api/chamo/img_proc', methods=['POST'])
def handle_api_img_proc():
    ret = dict(code=0, status='OK', payload_err_index=[])
    payload = request.json
    img_payload = payload.get('name', '')
    print(img_payload)
    ret['name'] = 'chamo'
    return jsonify(ret)

@app.route('/api/chamo/release_model', methods=['POST'])
def handle_api_release_model():
    ret = dict(code=0, status='OK', payload_err_index=[])
    create_dir(release_cp)
    tool.server.update_train_imgs.copy_files(cur_cp,release_cp)
    shutil.copyfile(cur_material_file, release_material_file)
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

@app.route('/api/chamo/merge_upload', methods=['POST'])
def handle_api_merge_upload():
    ret = dict(code=0, status='OK', payload_err_index=[])
    create_dir(result_root)
    files=os.listdir(upload_cache)
    for file_t in files:
        zip_file=upload_cache+'/'+file_t
        f_zip = zipfile.ZipFile(zip_file, 'r')
        create_dir(upload_temp)
        for f in f_zip.namelist():
            f_zip.extract(f, upload_temp)
        if file_t=='train.zip':
            tool.server.update_train_imgs.main(upload_temp, train_img_root)
        else:
            tool.server.update_train_imgs.main(upload_temp, test_img_root)
    save_materil_list()
    return jsonify(ret)

@app.route('/api/chamo/evaluation', methods=['POST'])
def handle_api_evaluation():
    create_dir(result_root)
    create_dir(result_group)
    ret = dict(code=0, status='OK', payload_err_index=[])
    imgs = os.listdir(eval_img_root)
    img_data_list = []
    for img_name in imgs:
        img_data_list.append(eval_img_root + '/' + img_name)
    checkpt=release_cp+'/chamo.ckpt'
    group_img.process_img(img_data_list, result_root,result_group, checkpt, get_materil_list(True))
    zip_dir(result_group, data_root + '/group.zip')
    ret['re']='group.zip'
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
    create_dir(train_tfrecord_root)
    create_dir(test_tfrecord_root)
    ret = dict(code=0, status='OK', payload_err_index=[])
    material_dict = {}
    count = 0
    material_list=get_materil_list(False)
    for item in material_list:
        material_dict[item[0]] = count
        count = count + 1
    data_convertor.img_2_tfrecord_2class.convert_a_folder(train_img_root, train_tfrecord_root, material_dict)
    data_convertor.img_2_tfrecord_2class.convert_a_folder(test_img_root, test_tfrecord_root, material_dict)
    return jsonify(ret)

@app.route('/api/chamo/train', methods=['POST'])
def handle_api_train():
    ret = dict(code=0, status='OK', payload_err_index=[])
    if check_proc('python', 'main.py')=='-1':
        create_dir(cur_cp)
        create_dir(logs)
    turn_proc('python ../../main.py chamo_class2 '+str(len(get_materil_list(False))), 'python', 'main.py')
    return jsonify(ret)

@app.route('/api/chamo/update_material', methods=['POST'])
def handle_api_update_material():
    ret = dict(code=0, status='OK', payload_err_index=[])
    save_materil_list()
    ret['re']=get_materil_list(False)
    return jsonify(ret)

@app.route('/api/chamo/tensorboard', methods=['POST'])
def handle_api_tensorboard():
    ret = dict(code=0, status='OK', payload_err_index=[])
    turn_proc('tensorboard --logdir='+logs, 'tensorboard', 'logdir')
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

@app.route('/api/chamo/scrap', methods=['POST'])
def handle_api_scrap():
    ret = dict(code=0, status='OK', payload_err_index=[])
    payload = request.json
    key_word_strs = payload.get('key_word', '')
    isPosi = payload.get('isPosi', '')
    maxcount = payload.get('maxcount', '')
    print(key_word_strs)
    print(isPosi)
    print(maxcount)
    key_words = key_word_strs.split(',')
    if key_word_strs=='':
        create_dir(scrap_root)
        os.mkdir(scrap_root + '/positive')
        os.mkdir(scrap_root + '/negative')
        return jsonify(ret)
    for key_word in key_words:
        create_dir(temp_root)
        scrap_root_sub=scrap_root
        if isPosi=='1':
            tool.scrap_img.scrap_img.scrap(key_word, temp_root, int(maxcount))
            scrap_root_sub=scrap_root+'/positive'
        else:
            tool.scrap_img.scrap_sougou.getSoGoImG(key_word, int(maxcount), temp_root)
            scrap_root_sub = scrap_root + '/negative'
        data_convertor.process_img.checkFormat(temp_root, 6)
        data_convertor.process_img.checkChannel(temp_root, 6)
        files = os.listdir(scrap_root_sub)
        if len(files)>0:
            tool.server.update_train_imgs.check_sim(temp_root, scrap_root_sub)
        tool.server.update_train_imgs.copy_files(temp_root, scrap_root_sub)
    zip_dir(scrap_root, data_root+'/scrap.zip')
    ret['re']='scrap.zip'
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
