from flask import Flask, render_template
import os
ip_list=[
        ['172.30.8.119','leo','944340525'],
        ['172.30.6.111', 'chamo','1168518506'],
        ['172.30.8.168', 'leo', '1166831135'],
        ['172.30.6.165', 'yiming', '1196481021'],
        ['172.30.8.167', 'alpha', '1166703805'],
        ['172.30.8.135', 'remy', '1178905113'],
    ]

table_frame='<table border="1" cellpadding="5" cellspacing="1">{content}</table>'
table_header = '<tr>' \
                    '<th>IP</th>' \
                    '<th>User</th>' \
                    '<th>Team Id</th>' \
                    '<th>Idel CPU</th>' \
                    '<th>Free Mem</th>' \
                    '<th>Disk Usage</th>' \
                    '<th>GPU Mem</th>' \
                    '<th>GPU Usage</th>' \
                    '<th>GPU Temp</th>' \
               '</tr>'
table_item_templ = '<tr>' \
                        '<th>{ip}</th>' \
                        '<th>{user}</th>' \
                        '<th>{view}</th>' \
                        '<th>{cpu}</th>' \
                        '<th>{mem}</th>' \
                        '<th>{disk}</th>' \
                        '<th>{gpu_mem}</th>' \
                        '<th>{gpu}</th>' \
                        '<th>{gpu_temp}</th>' \
                   '</tr>'
app = Flask(__name__)

def getNumList(str):
    splited = str.split(" ")
    num_list=[]
    for item in splited:
        if item.replace(".",'').isdigit() or item.replace("W",'').isdigit() or item.replace("MiB",'').isdigit() or item.replace("C",'').isdigit():
            num_list.append(item)
    return num_list

def getStat(ip):
    table_item =table_item_templ.replace('{ip}', ip[0])
    table_item = table_item.replace('{user}', ip[1])
    table_item = table_item.replace('{view}', ip[2])

    #check if the pc is online
    re_str = os.popen('fping -c1 -t200 ' + ip[0]).read()
    if not 'loss' in re_str:
        table_item = table_item.replace('{cpu}', '')
        table_item = table_item.replace('{disk}', '')
        table_item = table_item.replace('{mem}', '')
        table_item = table_item.replace('{gpu}', '')
        table_item = table_item.replace('{gpu_mem}', '')
        table_item = table_item.replace('{gpu_temp}', '')
        return table_item

    # get disk, cpu infor
    login_str='sshpass -p 1 ssh '+ip[1]+'@'+ip[0]+' '
    re_str= os.popen(login_str+'"iostat"').read()
    lines= re_str.split("\n")
    if len(lines)>3:
        re_list = getNumList(lines[3])
        if len(re_list)>5:
            idle_cpu=re_list[5]
            table_item = table_item.replace('{cpu}', idle_cpu + '%')
        else:
            table_item = table_item.replace('{cpu}', '')
        use_harddisk=0
        re_list = getNumList(lines[6])
        if len(re_list) > 2:
            r_h1 = re_list[1]
            w_h1 = re_list[2]
            use_harddisk = float(r_h1) + float(w_h1)
            use_harddisk = int(use_harddisk / 1000)
            table_item = table_item.replace('{disk}', str(use_harddisk) + 'MiB/s')
        else:
            table_item = table_item.replace('{disk}', '')
        re_list = getNumList(lines[7])
        if len(re_list) > 2:
            r_h2 = re_list[1]
            w_h2 = re_list[2]
            use_harddisk = use_harddisk + float(r_h2) + float(w_h2)
            use_harddisk = int(use_harddisk / 1000)
            table_item = table_item.replace('{disk}', str(use_harddisk) + 'MiB/s')
        else:
            table_item = table_item.replace('{disk}', '')
    else:
        table_item = table_item.replace('{cpu}', '')
        table_item = table_item.replace('{disk}', '')

    # get mem infor
    re_str= os.popen(login_str+'"vmstat"').read()
    lines= re_str.split("\n")
    if len(lines)>2:
        re_list = getNumList(lines[2])
        if len(re_list) > 3:
            mem_use=float(re_list[3])
            mem_use=int(mem_use/1024)
            mem_use=str(mem_use)+'MiB'
            table_item =table_item.replace('{mem}', mem_use)
        else:
            table_item = table_item.replace('{mem}', '')
    else:
        table_item = table_item.replace('{mem}', '')

    #get gpu info
    re_str= os.popen(login_str+'"nvidia-smi"').read()
    if re_str=="":
        table_item = table_item.replace('{gpu}', '')
        table_item = table_item.replace('{gpu_mem}', '')
        table_item = table_item.replace('{gpu_temp}', '')
    else:
        lines= re_str.split("\n")
        re_list = getNumList(lines[8])
        GPU_w=re_list[1]+'/'+re_list[2]
        GPU_mem=re_list[3]+'/'+re_list[4]
        table_item =table_item.replace('{gpu}', GPU_w)
        table_item=table_item.replace('{gpu_mem}', GPU_mem)
        table_item = table_item.replace('{gpu_temp}', re_list[0])
    return table_item

@app.route('/')
def user():
    html_str = table_header
    for ip in ip_list:
        html_item=getStat(ip)
        html_str=html_str+html_item
    html_content=table_frame.replace('{content}',html_str)
    return render_template('main.html', content=html_content)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', '8000'))
    app.run(host='0.0.0.0', port=port)
