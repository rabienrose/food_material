import os

img_root=u"D:\\raw_1\\菜花\\蒜蓉菜花\\"
label='01000'
imgs = os.listdir(img_root)
count=0
for i in range(len(imgs)):
    name_str='aa_'+label+'_'+str(count)+'.jpg'
    os.rename(img_root+imgs[i],img_root+name_str)
    count=count+1

