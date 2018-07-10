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

app = Flask(__name__)
PROD = True
CORS(app)

@app.route('/api/chamo/img_proc', methods=['POST'])
def handle_api_classify_dish():
    ret = dict(code=0, status='OK', payload_err_index=[])
    if request.method == 'POST':
        payload = request.json
        img_payload = payload.get('images', [])
        print(img_payload)
    else:
        ret['code'] = 0
        ret['status'] = 'Invalid request method. Require POST but get {}'.format(request.method)

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
