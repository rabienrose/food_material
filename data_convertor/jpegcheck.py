# -*- coding: utf-8 -*-
import os
import shutil
import sys
import numpy as np
import click
import time

def jpeg_check(path):
    print(path)
    files= os.listdir(path)
    for file in files:
        file_path = os.path.join(path,file)
        if not os.path.isdir(file_path):
            jpgcheck = 'jpeginfo' +' -c ' + file_path + ' -d>/dev/null 2>&1' +'\n'
            os.system(jpgcheck)
            if os.path.isfile(file_path):
                if os.path.splitext(file_path)[1] != ".jpg":
                    newname = os.path.splitext(file_path)[0]+".jpg"
                    os.rename(file_path,newname)
            else:
               print("%s is deleted\n" % file_path)
