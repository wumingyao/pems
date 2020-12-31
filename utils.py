import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import queue

from math import radians, cos, sin, asin, sqrt

import datetime
import time
import re
from config import *


def string2timestamp(strValue):
    d = datetime.datetime.strptime(strValue, "%m/%d/%Y %H:%M:%S")
    t = d.timetuple()
    timeStamp = int(time.mktime(t))
    return timeStamp


# 1440751417.283 --> '2015-08-28 16:43:37.283'
def timestamp2string(timeStamp):
    d = datetime.datetime.fromtimestamp(timeStamp)
    str = d.strftime("%Y-%m-%d %H:%M:%S")
    return str


# 通过时间戳返回时间片
def getTemporalInterval(timestamp):
    # input:timestamp = 2019-1-25 09:35:27
    # return: 25, 9, 3
    str = timestamp2string(timestamp)
    timestamp = str.replace('/', '-')
    t = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    weekday = t.weekday()
    hour = t.hour
    minute = t.minute
    date = re.search(r"(\d{4}-\d{1,2}-\d{1,2})", timestamp).group(0)
    day = t.day
    # date = str(t.year) + '-' + str(t.month) + '-' + str(t.day)
    slice = (minute - 1) // SLICE
    return date, day, weekday, hour, slice + hour * 6  # day是一个月中的第几天


# 获得前一天
def get_day(date):
    """
    :param date: 2019-01-02
    :return: 2019-01-01
    """
    date = date + ' 00:00:00'
    timeStamp = string2timestamp(date) - 24 * 60 * 60
    string = timestamp2string(timeStamp)
    t = datetime.datetime.strptime(string, '%Y-%m-%d %H:%M:%S')
    return str(t).split()[0]


# 获得前一周对应天
def get_week(date):
    """
    :param date: 2019-01-02
    :return: 2018-12-26
    """
    date = date + ' 00:00:00'
    timeStamp = string2timestamp(date) - 7 * 24 * 60 * 60
    string = timestamp2string(timeStamp)
    t = datetime.datetime.strptime(string, '%Y-%m-%d %H:%M:%S')
    return str(t).split()[0]


# 判断该日期是一年中的第几天
def day_of_year(y, m, d):
    sday = datetime.date(y, m, d)
    count = sday - datetime.date(sday.year - 1, 12, 31)  # 减去上一年最后一天
    return count.days


# 计算某一年的天数
def days_of_year(n):
    if n % 4 == 0 and n % 100 != 0 or n % 400 == 0:
        return 366
    else:
        return 365


# 公式计算两点间距离（km）
def geodistance(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000
    distance = round(distance / 1000, 3)
    return distance


def ReadExcelFile(fileNamePath):
    with open(fileNamePath, 'rb') as f:
        sheet = pd.read_excel(f)
        return sheet


def ReadCSVFile(fileNamePath):
    with open(fileNamePath, 'rb') as f:
        sheet = pd.read_csv(f)
        return sheet


def stackedBar(df, title, ylabel, y_max, width, Name_bar, gap, save2File, myfont=None):
    '''
    堆叠柱状图
    :param df: 带有header的dataframe,有多少列就有多少个柱子
    :param title: 标题
    :param ylabel:
    :param y_max:
    :param width:
    :param Name_bar: 图例名称
    :param gap:
    :param save2File: 保存路径
    :param myfont:
    :return:
    '''
    # plt.rcParams['font.sans-serif'] = [u'SimHei']
    # plt.rcParams['axes.unicode_minus'] = False

    # 设置标题
    plt.title(title, fontproperties=myfont)
    # X坐标的文字设置和间隔设置
    ind = np.arange(len(list(df.columns.values.tolist())))
    plt.xticks(ind, df.columns.values, fontproperties=myfont, rotation=90)
    # plt.yticks(np.arange(0, y_max, gap))  # 0到y_max 间隔20
    plt.ylabel(ylabel, fontproperties=myfont)
    width = width  # 设置条形图一个长条的宽度
    color = ['pink', 'salmon', 'c', 'm', 'y',
             'k', 'darkorange', 'lightgreen', 'plum', 'tan',
             'khaki', 'g', 'r', 'skyblue', 'lawngreen', 'b']
    data = np.array(df)
    s = np.sum(data, axis=0)
    for i in range(len(data)):
        plt.bar(ind, data[i] / s, width=width, bottom=np.sum(data[:i], axis=0), color=color[i])
    plt.legend(labels=Name_bar, bbox_to_anchor=(1.05, 0), loc=3, prop=myfont, borderaxespad=0)
    plt.savefig(save2File + '.jpg', dpi=200, bbox_inches='tight', fontproperties=myfont)
    plt.show()


def bit_product_sum(x, y):
    return sum([item[0] * item[1] for item in zip(x, y)])


def cosine_similarity(x, y, norm=False):
    """ 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)

    # method 1
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内


def bfs(adj, start):
    visited = set()
    visited.add(start)
    q = queue.Queue()
    q.put(start)  # 把起始点放入队列
    while not q.empty():
        u = q.get()
        # print(u)
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                q.put(v)
    return list(visited)


# graph = {1: [4, 2], 2: [3, 4], 3: [4], 4: [5]}
# bfs(graph, 1)

if __name__ == '__main__':
    print(geodistance(124, 23, 122, 35))
