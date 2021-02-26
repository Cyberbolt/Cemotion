'''
    用于文件下载
'''

import requests
import os
from urllib.request import urlopen, Request

import requests
from tqdm import tqdm

headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:60.0) Gecko/20100101 Firefox/60.0'}

def download_from_url(url, dst):
    """
    @param: url to download file
    @param: dst place to put the file
    :return: bool
    """
    # 获取文件长度
    try:
        file = Request(url,headers=headers)
        file_size = int(urlopen(file).info().get('Content-Length', -1))
    except Exception as e:
        print(e)
        print("错误，访问url: %s 异常" % url)
        raise 'file not exit'
        return False

    # 判断本地文件存在时
    if os.path.exists(dst):
        # 获取文件大小
        first_byte = os.path.getsize(dst)
    else:
        # 初始大小为0
        first_byte = 0

    # 判断大小一致，表示本地文件存在
    if first_byte >= file_size:
        # print("文件已经存在,无需下载")
        return file_size

    # #删除不完整的文件
    # try:
    #     os.remove(dst)
    # except:
    #     pass
    
    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(
        total=file_size, initial=first_byte,
        unit='B', unit_scale=True, desc=dst.split('/')[-1])

    # 访问url进行下载
    req = requests.get(url, headers=headers, stream=True)

    #循环下载
    files = []
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:
            files.append(chunk)
            pbar.update(1024)
        
    with(open(dst, 'ab')) as f:
        for chunk in files:
            f.write(chunk)

    pbar.close()
    return True