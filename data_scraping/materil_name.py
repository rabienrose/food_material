material_list=[
    ["青椒"],
    ["木耳"],
    ["鸡蛋"],
    ["豆腐"],
    ["虾仁"],
    ["黄瓜"],
    ["排骨"]
]

def get_matarray_num(mat_array, mat_name):
    '''
    根据食材的名称获取食材的编号
    :param mat_array: 食材数组
    :param mat_name: 食材的名称
    :return: 食材的编号
    '''
    for i in mat_array:
        if mat_name in i:
            return mat_array.index(i)
    return None

