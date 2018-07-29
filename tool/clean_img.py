import tkinter as tk
from PIL import Image, ImageTk
import os,shutil

def show_next():
    global cur_img_id
    cur_img_id=cur_img_id+1
    if cur_img_id>=len(image_list):
        return False
    im = Image.open(img_dir+'/'+image_list[cur_img_id])
    (x, y) = im.size
    x_s = 1000
    y_s = int(y * x_s / x)
    if y_s>700:
        y_s=700
        x_s=int(x*y_s/y)
    out = im.resize((x_s, y_s), Image.ANTIALIAS)
    bm = ImageTk.PhotoImage(out)
    label2.configure(image=bm)
    label2.bm=bm
    img_c.config(text=str(cur_img_id+1) + '/' + str(len(image_list)))
    return True

def call(event):
    if event.keysym=='q':
        show_next()
    elif event.keysym=='w':
        if tar_dir=="":
            os.remove(img_dir+'/'+image_list[cur_img_id])
        else:
            shutil.move(img_dir+'/'+image_list[cur_img_id], tar_dir+'/positive'+'/'+image_list[cur_img_id])
        show_next()
    elif event.keysym == 'e':
        if tar_dir=="":
            os.remove(img_dir+'/'+image_list[cur_img_id])
        else:
            shutil.move(img_dir+'/'+image_list[cur_img_id], tar_dir+'/negative'+'/'+image_list[cur_img_id])
        show_next()


def go():
    global image_list
    global img_dir
    global tar_dir
    img_dir = img_dir_t.get()
    tar_dir = tar_dir_t.get()
    top.focus_set()
    if tar_dir!="":
        if not os.path.exists(tar_dir + '/positive'):
            os.mkdir(tar_dir + '/positive')
        if not os.path.exists(tar_dir + '/negative'):
            os.mkdir(tar_dir + '/negative')
    image_list = os.listdir(img_dir)
    img_c.config(text='0'+'/'+str(len(image_list)))

logo='/home/leo/Desktop/loli.jpeg'
img_dir = ''
tar_dir = ''
image_list=[]
cur_img_id=-1

top = tk.Tk()
top.bind("<Key>", call)
top.title('label test')


img_dir_t=tk.Entry(top,width=100)
img_dir_t.pack()
tar_dir_t=tk.Entry(top,width=100)
tar_dir_t.pack()

button=tk.Button(top,text="load",command=go)
button.pack()

img_c=tk.Label(top, text='0/0')
img_c.pack()

im = Image.open(logo)
bm = ImageTk.PhotoImage(im)
label2 = tk.Label(top, image=bm, width=1000,height=0)
label2.bm = bm
label2.pack()
top.mainloop()







