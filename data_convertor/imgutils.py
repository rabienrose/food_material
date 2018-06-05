#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Image data clean up tool before training CNN models

Usage:
    run as:
        python imgutils.py --input_dir "/path/to/img" --pattern "*/*.jpg" --task "1-2-3" --nproc 2

    example:
        python imgutils.py --input_dir "/path/to/img" --pattern "*/*.jpg" --task "1-2-3" --nproc 2
"""
import os
import sys
import click
import piexif
from PIL import Image
import numpy as np
from multiprocessing import Pool

def task_proc(task_list, fpath_list):
    '''

    :param task_list:
    :param fpath_list:
    :return:
    '''
    for img_path in fpath_list:
        # check if the image file is valid
        if '1' in task_list:
            im = Image.open(img_path)
            try:
                im.load()
            except:
                print('removing the corrupted image file: {}'.format(img_path))
                os.system('rm {}'.format(img_path))

        # remove EXIF info from the img file
        if '2' in task_list:
            try:
                piexif.remove(img_path)
            except:
                print('piexif raise exceptions when processing {}'.format(img_path))

        # convert image to RGB and 3-channel mode
        if '3' in task_list:
            saving_flag = False

            pil_im = Image.open(img_path)
            if pil_im.mode != "RGB":
                pil_im = pil_im.convert("RGB")
                saving_flag = True
                print("RGB issue: %s" % img_path)

            im = np.array(pil_im)
            if len(im.shape) < 2:
                im = im[0]
                pil_im = Image.fromarray(im)
                pil_im.save(img_path, "JPEG")
                print("too few channel: %s" % img_path)
            elif len(im.shape) == 3 and im.shape[2] >= 4:
                im = im[:, :, :3]
                pil_im = Image.fromarray(im)
                pil_im.save(img_path, "JPEG")
                print("too much channel: %s" % img_path)
            else:
                if saving_flag:
                    pil_im.save(img_path, "JPEG")

def main(input_dir, pattern, task, nproc):
    from glob import glob
    from os import path

    all_files = glob(path.join(input_dir, pattern))
    all_files.sort()
    print('Found {} files in {} folder'.format(len(all_files), input_dir))

    task_list = str(task).split('-')
    tt_files = len(all_files)
    files_per_proc = int(tt_files / nproc)
    p = Pool()
    for idx in range(nproc):
        p.apply_async(task_proc, args=(task_list, all_files[idx * files_per_proc: (idx + 1) * files_per_proc],))
    p.close()
    p.join()


if __name__ == '__main__':
    main()
