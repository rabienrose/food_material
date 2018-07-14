import requests
import json
import urllib.request
import random
# 封装成函数
def getSoGoImG(img_url, num, save_path):
    url = img_url
    path = save_path
    count=0
    for i in range(100):
        imgs = requests.get(url + '&start=' + str(i * 48) + '&reqType=ajax&tn=0&reqFrom=result')
        jd = json.loads(imgs.text)
        jd = jd['items']
        imgs_url = []
        for j in jd:
            imgs_url.append(j['thumbUrl'])
        for img_url in imgs_url:
            m=random.randint(1,100000)
            print('***** ' + str(m) + '.jpg *****' + ' Downloading...')
            try:
                urllib.request.urlretrieve(img_url, path+'/' + str(m) + '.jpg')
            except urllib.error.HTTPError as e:
                print('打不开！哼')
                continue
            except urllib.error.URLError as e:
                print('打不开！哼')
                continue
            except Exception as e:
                print('打不开！哼')
                continue
            count=count+1
            if count>num:
                break
        if count > num:
            break
    print('Download complete!')

if __name__ == '__main__':
    img_url = 'http://pic.sogou.com/ris?query=http%3A%2F%2Fimg02.sogoucdn.com%2Fapp%2Fa%2F100520146%2F5C1AB65F306FCD08E5CD42705A4A79A2&flag=1'
    save_path = '/home/leo/Downloads/chamo/try/'
    getSoGoImG(img_url, 20, save_path)
