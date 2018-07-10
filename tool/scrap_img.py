import re
import requests
key_word='家常菜'
for i in range(30):
    tem=str(i*60)
    url='http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word='+key_word+'&pn='+tem+'&gsm=0'
    #url='https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=%E5%B0%8F%E9%BB%84%E4%BA%BA&pn='+tem+'&gsm=0'
    #print(url)
    html=requests.get(url).text
    pic_url=re.findall('"objURL":"(.*?)",',html,re.S)
    n=100000+i*60
    for each in pic_url:
        print(n)
        #print(each)
        try:
            pic = requests.get(each, timeout=3)
        except requests.exceptions.ConnectTimeout:
            print('打不开！哼')
            continue
        except requests.exceptions.Timeout:
            print('打不开！哼')
            continue
        except requests.exceptions.ConnectionError:
            print('打不开！哼')
            continue
        except requests.exceptions.TooManyRedirects:
            print('打不开！哼')
            continue
        string='./img/re2/'+key_word+'_'+str(n)+'.jpg'
        fp=open(string,'wb')
        fp.write(pic.content)
        fp.close()
        n=n+1