import re
import requests
for i in range(10):
    tem=str(i*60)
    url='http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=上汤娃娃菜&pn='+tem+'&gsm=0'
    #url='https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=%E5%B0%8F%E9%BB%84%E4%BA%BA&pn='+tem+'&gsm=0'
    html=requests.get(url).text
    pic_url=re.findall('"objURL":"(.*?)",',html,re.S)
    n=i*60
    for each in pic_url:
        print(each)
        try:
            pic=requests.get(each,timeout=800)
        except requests.exceptions.ConnectionError:
            print('打不开！哼')
            continue
        string='.\\wawacai2\\'+str(n)+'.jpg'
        fp=open(string,'wb')
        fp.write(pic.content)
        fp.close()
        n=n+1