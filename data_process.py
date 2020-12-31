import os
import pandas as pd
from utils import geodistance, string2timestamp, cosine_similarity, bfs
import numpy as np
import math
from config import *

from collections import Counter


def station_merge(dir):
    '''
    将pems各个地区的站点数据合并成一个文件
    :param dir: 站点数据的根目录
    :return:
    '''
    files = os.listdir(dir)
    data = []
    for f in files:
        csvfile = os.path.join(dir, f)
        df = pd.read_csv(csvfile, sep='\t', error_bad_lines=False)
        data.append(df)
    df = pd.concat(data, axis=0)
    pass


def incident_merge(dir):
    '''
    将pems2017/01/01-2017/06/30的事故合并成一个文件
    :param dir:
    :return:
    '''
    files = os.listdir(dir)
    data = []
    for f in files:
        csvfile = os.path.join(dir, f)
        df = pd.read_csv(csvfile, names=(
            'Incident_ID', 'CC_Code', 'Incident_Number', 'Timestamp', 'Description', 'Location', 'Area', 'Zoom_Map',
            'TB_xy',
            'Latitude', 'Longtitude', 'District', 'Country_FIPS_ID', 'City_FIPS_ID', 'Freeway_Number',
            'Freeway_Direction',
            'State_Postmile', 'Absolute_Postmile',
            'Severity', 'Duration'), sep=',')
        data.append(df)
    df = pd.concat(data, axis=0)
    df.to_csv(os.path.join(dir, 'incidient_20170101to20170630_merge.csv'), index=False)


def get_Bay_Area_incident(incident_merge, station_bay_total, station_bay_target):
    '''
    获得pems-bay公开数据集匹配的事故
    :param incident_merge: 2017/01/01-2017/06/30加州总的事故数据文件：csv
    :param station_bay_total: Bay Area 地区所有的sensors数据集文件：txt
    :param station_bay_target: pems-bay公开数据集中所使用的sensor文件：csv
    :return:
    '''
    df_incident_merge = pd.read_csv(incident_merge)
    df_station_bay_total = pd.read_csv(station_bay_total, sep='\t')
    df_station_bay_target = pd.read_csv(station_bay_target, names=('ID', 'lat', 'lng'))

    # 筛选目标sensors
    df_station = df_station_bay_total[df_station_bay_total['ID'].isin(list(df_station_bay_target['ID']))]
    df_station.to_csv('sensors_pems_bay.csv', index=False)

    # 筛选出Bay Area 的incidents
    df_incident_Bay = df_incident_merge[df_incident_merge['District'] == 4]
    sensors = []
    dis = []
    for index1, row in df_incident_Bay.iterrows():
        # 1.确定事故发生的公路
        Freeway_Number = row['Freeway_Number']
        Freeway_Direction = row['Freeway_Direction']
        # 2.找出在该公路上的sensors
        df_sensors = df_station[(df_station['Fwy'] == Freeway_Number) & (df_station['Dir'] == Freeway_Direction)]
        # 3.确定离该事故最近的sensor
        min_dis = float("+inf")
        min_dis_sensor = None
        for index2, r in df_sensors.iterrows():
            d = geodistance(df_incident_Bay['Longtitude'][index1], df_incident_Bay['Latitude'][index1],
                            df_sensors['Longtitude'][index2], df_sensors['Latitude'][index2])
            if d <= min_dis:
                min_dis = d
                min_dis_sensor = df_sensors['ID'][index2]
        sensors.append(min_dis_sensor)
        dis.append(min_dis)
    df_incident_Bay['sensor'] = sensors
    df_incident_Bay['dis'] = dis
    df_incident_Bay = df_incident_Bay[~(df_incident_Bay['sensor'].isnull())]
    df_incident_Bay.to_csv('/public/lhy/wmy/dataset/Pems/incidents/incident_pems-bay.csv', index=False)
    pass


def get_POI_accident(POI_dataset, sensor):
    '''
    POI_dataset是带有详细POI的数据，sensor是传感器数据集
    通过距离，找出sensor附近的POI
    :param POI_dataset:
    :param incident:
    :return:
    '''
    df_sensor = pd.read_csv(sensor)
    df_POI_dataset = pd.read_csv(POI_dataset)
    Bay_cities = ['San Francisco', 'San Mateo', 'Palo Alto', 'San Jose', 'Gilroy', 'Fremont', 'Hayward', 'Oakland',
                  'Berkeley', 'Richmond', 'Concord', 'Vallejo', 'Fairfield', 'Napa', 'Novato', 'San Rafael']

    df_POI_dataset = df_POI_dataset[df_POI_dataset['City'].isin(Bay_cities)]
    min_dis_list = []
    Amenity = []
    Bump = []
    Crossing = []
    Give_Way = []
    Junction = []
    No_Exit = []
    Railway = []
    Roundabout = []
    Station = []
    Stop = []
    Traffic_Calming = []
    Traffic_Signal = []
    Turning_Loop = []
    for i in range(len(df_sensor)):
        min_dis = float("+inf")
        for j in range(len(df_POI_dataset)):
            d = geodistance(df_sensor.iloc[i]['Longtitude'], df_sensor.iloc[i]['Latitude'],
                            df_POI_dataset.iloc[j]['Start_Lng'], df_POI_dataset.iloc[j]['Start_Lat'])
            if d < min_dis:
                min_dis = d
                indexj = j
        min_dis_list.append(min_dis)
        Amenity.append(df_POI_dataset.iloc[indexj]['Amenity'])
        Bump.append(df_POI_dataset.iloc[indexj]['Bump'])
        Crossing.append(df_POI_dataset.iloc[indexj]['Crossing'])
        Give_Way.append(df_POI_dataset.iloc[indexj]['Give_Way'])
        Junction.append(df_POI_dataset.iloc[indexj]['Junction'])
        No_Exit.append(df_POI_dataset.iloc[indexj]['No_Exit'])
        Railway.append(df_POI_dataset.iloc[indexj]['Railway'])
        Roundabout.append(df_POI_dataset.iloc[indexj]['Roundabout'])
        Station.append(df_POI_dataset.iloc[indexj]['Station'])
        Stop.append(df_POI_dataset.iloc[indexj]['Stop'])
        Traffic_Calming.append(df_POI_dataset.iloc[indexj]['Traffic_Calming'])
        Traffic_Signal.append(df_POI_dataset.iloc[indexj]['Traffic_Signal'])
        Turning_Loop.append(df_POI_dataset.iloc[indexj]['Turning_Loop'])
    df_sensor['POI_dis'] = min_dis_list
    df_sensor['Amenity'] = Amenity
    df_sensor['Bump'] = Bump
    df_sensor['Crossing'] = Crossing
    df_sensor['Give_Way'] = Give_Way
    df_sensor['Junction'] = Junction
    df_sensor['No_Exit'] = No_Exit
    df_sensor['Railway'] = Railway
    df_sensor['Roundabout'] = Roundabout
    df_sensor['Station'] = Station
    df_sensor['Stop'] = Stop
    df_sensor['Traffic_Calming'] = Traffic_Calming
    df_sensor['Traffic_Signal'] = Traffic_Signal
    df_sensor['Turning_Loop'] = Turning_Loop

    df_sensor.sort_values('POI_dis')
    df_sensor.to_csv('/public/lhy/wmy/dataset/Pems/station/sensors_pems_bay_POI.csv', index=False)
    pass


def get_staion_POI():
    sensor_POI = pd.read_csv('/public/lhy/wmy/dataset/Pems/station/sensors_pems_bay_POI.csv')
    sensor_bay = pd.read_csv('/public/lhy/wmy/dataset/Pems/station/graph_sensor_locations_bay.csv',
                             names=('ID', 'lat', 'lng'))
    graph_sensor_POI_bay = pd.DataFrame()
    df_POI_dataset = pd.read_csv('/public/lhy/wmy/dataset/Accident/US_Accidents_June20.csv')
    # Bay_cities = ['San Francisco', 'San Mateo', 'Palo Alto', 'San Jose', 'Gilroy', 'Fremont', 'Hayward', 'Oakland',
    #               'Berkeley', 'Richmond', 'Concord', 'Vallejo', 'Fairfield', 'Napa', 'Novato', 'San Rafael']
    #
    # df_POI_dataset = df_POI_dataset[df_POI_dataset['City'].isin(Bay_cities)]
    ID = []
    min_dis_list = []
    Amenity = []
    Bump = []
    Crossing = []
    Give_Way = []
    Junction = []
    No_Exit = []
    Railway = []
    Roundabout = []
    Station = []
    Stop = []
    Traffic_Calming = []
    Traffic_Signal = []
    Turning_Loop = []
    for i in range(len(sensor_bay)):
        if sensor_bay.iloc[i]['ID'] not in list(sensor_POI['ID']):
            min_dis = float("+inf")
            for j in range(len(df_POI_dataset)):
                d = geodistance(sensor_bay.iloc[i]['lat'], sensor_bay.iloc[i]['lng'],
                                df_POI_dataset.iloc[j]['Start_Lng'], df_POI_dataset.iloc[j]['Start_Lat'])
                if d < min_dis:
                    min_dis = d
                    indexj = j
            ID.append(sensor_bay.iloc[i]['ID'])
            min_dis_list.append(min_dis)
            Amenity.append(df_POI_dataset.iloc[indexj]['Amenity'])
            Bump.append(df_POI_dataset.iloc[indexj]['Bump'])
            Crossing.append(df_POI_dataset.iloc[indexj]['Crossing'])
            Give_Way.append(df_POI_dataset.iloc[indexj]['Give_Way'])
            Junction.append(df_POI_dataset.iloc[indexj]['Junction'])
            No_Exit.append(df_POI_dataset.iloc[indexj]['No_Exit'])
            Railway.append(df_POI_dataset.iloc[indexj]['Railway'])
            Roundabout.append(df_POI_dataset.iloc[indexj]['Roundabout'])
            Station.append(df_POI_dataset.iloc[indexj]['Station'])
            Stop.append(df_POI_dataset.iloc[indexj]['Stop'])
            Traffic_Calming.append(df_POI_dataset.iloc[indexj]['Traffic_Calming'])
            Traffic_Signal.append(df_POI_dataset.iloc[indexj]['Traffic_Signal'])
            Turning_Loop.append(df_POI_dataset.iloc[indexj]['Turning_Loop'])
    graph_sensor_POI_bay['ID'] = ID
    graph_sensor_POI_bay['POI_dis'] = min_dis_list
    graph_sensor_POI_bay['Amenity'] = Amenity
    graph_sensor_POI_bay['Bump'] = Bump
    graph_sensor_POI_bay['Crossing'] = Crossing
    graph_sensor_POI_bay['Give_Way'] = Give_Way
    graph_sensor_POI_bay['Junction'] = Junction
    graph_sensor_POI_bay['No_Exit'] = No_Exit
    graph_sensor_POI_bay['Railway'] = Railway
    graph_sensor_POI_bay['Roundabout'] = Roundabout
    graph_sensor_POI_bay['Station'] = Station
    graph_sensor_POI_bay['Stop'] = Stop
    graph_sensor_POI_bay['Traffic_Calming'] = Traffic_Calming
    graph_sensor_POI_bay['Traffic_Signal'] = Traffic_Signal
    graph_sensor_POI_bay['Turning_Loop'] = Turning_Loop
    df = sensor_POI[
        ['ID', 'POI_dis', 'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout',
         'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']]
    graph_sensor_POI_bay = pd.concat([df, graph_sensor_POI_bay])
    graph_sensor_POI_bay.to_csv('/public/lhy/wmy/dataset/Pems/station/graph_sensor_POI_bay.csv')


def get_slice(timestamp, duration):
    d = string2timestamp(timestamp) + duration * 60 - string2timestamp("01/01/2017 00:00:00")
    slice = math.ceil(d / (SLICE * 60))
    return slice


def gen_accident(accident_file, sensorid_file):
    # get_slice("01/01/2017 00:05:20")
    df_accident = pd.read_csv(accident_file)
    df_accident['sensor'] = list(map(str, list(map(int, list(df_accident['sensor'])))))
    df_accident = df_accident.dropna(subset=['Timestamp'])
    df_accident['Duration'].fillna(0, inplace=True)
    with open(sensorid_file, 'r') as f:
        sensors = f.read()
        sensors = sensors.split(',')
    data_accident = np.zeros((52116, 325, 1))
    for index, row in df_accident.iterrows():
        slice_start = get_slice(df_accident['Timestamp'][index], 0)
        slice_end = get_slice(df_accident['Timestamp'][index], df_accident['Duration'][index])
        sensorid = df_accident['sensor'][index]
        id = sensors.index(sensorid)
        for slice in range(slice_start, slice_end + 1):
            if slice > 52116:
                continue
            data_accident[slice - 1, id, 0] += 1
    np.save('/public/lhy/wmy/dataset/Pems/incidents/accident.npy', data_accident)

    pass


def gen_sensor_attr(sensor_bay_file, sensor_id):
    df_sensors_bay = pd.read_csv(sensor_bay_file, sep='\t', error_bad_lines=False)
    with open(sensor_id, 'r') as f:
        sensors = f.read()
        sensors = sensors.split(',')
    df_sensors_bay['ID'] = list(map(str, list(df_sensors_bay['ID'])))
    df_sensors_bay = df_sensors_bay[df_sensors_bay['ID'].isin(sensors)]
    df_sensors_bay = df_sensors_bay.dropna(axis=1, how='any')
    # ID = df_sensors_bay['ID']
    df_sensors_bay = df_sensors_bay.drop(['State_PM', 'Name', 'User_ID_1', 'User_ID_2', 'User_ID_3', 'User_ID_4'],
                                         axis=1)

    # df_sensors_bay.insert(0, 'ID', ID)
    df_sensors_bay.to_csv('/public/lhy/wmy/dataset/Pems/station/graph_sensor_attr_bay.csv', index=False)


def get_sensor_static_info_one_hot(POI_file, Attr_file):
    df_poi = pd.read_csv(POI_file)
    df_poi = df_poi.replace(True, 1)
    df_poi = df_poi.replace(False, 0)
    df_attr = pd.read_csv(Attr_file)
    ID = df_attr['ID']
    df_attr.drop(['ID', 'District', 'County'], axis=1)
    df_attr['Fwy'] = list(map(str, list(df_attr['Fwy'])))
    df_attr = pd.get_dummies(df_attr)
    df_attr['ID'] = ID
    df_sensors_static_info = pd.merge(df_attr, df_poi, on='ID')
    return df_sensors_static_info


def search_subgraph(sensor, adj, N_sub):
    '''
    :param sensor: sensor在idlist中的序列号
    :param adj: 全局adj
    :return: list,len=N_sub
    '''
    res = bfs(adj, sensor)
    if len(res) > N_sub:
        res = res[:N_sub]
    res.sort()
    return res


def get_subAdj(sensorid_file, POI_file, Attr_file, adj, traffic_file, accident_file, N_sub=10):
    '''
    计算每个点的小图邻接矩阵
    :param sensorid_file:txt,sensorsid的顺序
    :param POI_file: csv,poi信息
    :param Attr_file: csv,sensor属性信息
    :param adj: npy,全局邻接矩阵，大图
    :param traffic_file: npz,流量矩阵
    :param accident_file: npy,事故矩阵
    :return: (N,T,N_sub,N_sub)
    '''
    with open(sensorid_file, 'r') as f:
        sensors = f.read()
        sensors = sensors.split(',')
    df_sensors = pd.DataFrame()
    df_sensors['ID'] = sensors
    df_static = get_sensor_static_info_one_hot(POI_file, Attr_file)
    df_static['ID'] = df_static['ID'].astype('str')
    data_static = np.array(pd.merge(df_sensors, df_static, on='ID').drop('ID', axis=1))
    adj = np.load(adj)
    traffic = np.load(traffic_file)['data']
    range_traffic = traffic[1:] - traffic[:-1]
    range_traffic = np.concatenate((np.zeros((1, traffic.shape[1], traffic.shape[2])), range_traffic), axis=0)

    accident = np.load(accident_file)
    A_sub = np.zeros((len(traffic), len(sensors), N_sub, N_sub))
    adj_dict = {}
    for i in range(len(adj)):
        adj_dict[i] = [j for j, v in enumerate(adj[i]) if v >= 1]
    # sensors_sub = sensors[65 * (nk - 1):65 * nk]
    for s in sensors:
        print(sensors.index(s))
        # 对于每个sensor
        sindex = sensors.index(s)
        sublist = search_subgraph(sindex, adj_dict, N_sub)
        for t in range(len(traffic)):
            # 对于每个时刻
            # 计算sublist之间的相似度
            a = np.zeros((N_sub, N_sub))
            for k in sublist:
                for m in sublist:
                    if k == m:
                        continue
                    vector_sensor1 = list(data_static[k])
                    # vector_sensor1.append(accident[t, k, 0])
                    vector_sensor1.append(range_traffic[t, k, 0])
                    vector_sensor2 = list(data_static[m])
                    # vector_sensor2.append(accident[t, m, 0])
                    vector_sensor2.append(range_traffic[t, m, 0])
                    a[sublist.index(k)][sublist.index(m)] = cosine_similarity(vector_sensor1, vector_sensor2)
            A_sub[t, sindex, :, :] = a
    np.save('/public/lhy/wmy/myMRA-GCN/data/PEMSBAY/adj_sub_' + str(N_sub) + '.npy', A_sub)
    print("Dnoe")
    pass


def gen_edge_info(distances_bay_filename, sensorid_filename):
    with open(sensorid_filename, 'r') as f:
        sensors = f.read()
        sensors = sensors.split(',')
    df_dist = pd.read_csv(distances_bay_filename, names=['sensor_start', 'sensor_end', 'distance'])
    df_dist = df_dist[df_dist['distance'] > 0]
    df_dist = df_dist.sort_values('distance')
    df_dist['edgeId'] = [i for i in range(len(df_dist))]
    df_dist['index_start'] = df_dist.apply(lambda x: sensors.index(str(int(x['sensor_start']))), axis=1)
    df_dist['index_end'] = df_dist.apply(lambda x: sensors.index(str(int(x['sensor_end']))), axis=1)
    df_dist.to_csv("/public/lhy/wmy/dataset/Pems/edge/edgeinfo.csv", index=False)


def gen_adj_edge(edgeinfo_filename):
    edgeinfo = pd.read_csv(edgeinfo_filename)
    adj = np.zeros((len(edgeinfo), len(edgeinfo)))
    df_merge = pd.merge(edgeinfo, edgeinfo, left_on='index_end', right_on='index_start')
    for i in range(len(df_merge)):
        adj[int(df_merge.iloc[i]['edgeId_x']), int(df_merge.iloc[i]['edgeId_y'])] = 1
    np.save('/public/lhy/wmy/dataset/Pems/edge/adj_edge.npy', adj)


#     edgeId,tgs_start,tgs_end,index_start,index_end
if __name__ == '__main__':
    get_staion_POI()
    # get_POI_accident('/public/lhy/wmy/dataset/Accident/US_Accidents_June20.csv',
    #                  '/public/lhy/wmy/dataset/Pems/station/sensors_pems_bay.csv')
    #
    # # incident_merge('/public/lhy/wmy/dataset/Pems/incidents')
    # get_Bay_Area_incident('/public/lhy/wmy/dataset/Pems/incidents/incidient_20170101to20170630_merge.csv',
    #                       '/public/lhy/wmy/dataset/Pems/station/d04_text_meta_2020_08_20.txt',
    #                       '/public/lhy/wmy/dataset/Pems/station/graph_sensor_locations_bay.csv')
    # gen_accident('/public/lhy/wmy/dataset/Pems/incidents/incident_pems-bay.csv',
    #              '/public/lhy/wmy/dataset/Pems/station/graph_sensor_ids.txt')
    # gen_sensor_attr('/public/lhy/wmy/dataset/Pems/station/station_meta/d04_text_meta_2017_01_04.txt',
    #                 '/public/lhy/wmy/dataset/Pems/station/graph_sensor_ids.txt')
    # get_sensor_static_info_one_hot('/public/lhy/wmy/dataset/Pems/station/graph_sensor_POI_bay.csv',
    #                                '/public/lhy/wmy/dataset/Pems/station/graph_sensor_attr_bay.csv')
    # import argparse
    #
    # parser = argparse.ArgumentParser(description='manual to this script')
    # parser.add_argument("--batch", type=int, default=1)
    # args = parser.parse_args()
    # print(args.batch)
    # get_subAdj('/public/lhy/wmy/dataset/Pems/station/graph_sensor_ids.txt',
    #            '/public/lhy/wmy/dataset/Pems/station/graph_sensor_POI_bay.csv',
    #            '/public/lhy/wmy/dataset/Pems/station/graph_sensor_attr_bay.csv',
    #            '/public/lhy/wmy/ASTGCN-r-pytorch/data/PEMSBAY/adj.npy',
    #            '/public/lhy/wmy/dataset/Pems/traffic/pems-bay.npz',
    #            '/public/lhy/wmy/dataset/Pems/incidents/accident.npy', N_sub=5)
    # gen_edge_info('/public/lhy/wmy/dataset/Pems/edge/distances_bay_2017.csv',
    #               '/public/lhy/wmy/dataset/Pems/station/graph_sensor_ids.txt')
    # gen_adj_edge('/public/lhy/wmy/dataset/Pems/edge/edgeinfo.csv')
    # dir = '/public/lhy/wmy/dataset/Pems/train/sub_A'
    # files = os.listdir(dir)
    # A_sub = np.zeros((52116, 325, 10, 10))
    # for i in range(len(files)):
    #     a = np.load(os.path.join(dir, files[i]))
    #     A_sub[:, 65 * i:65 * (i + 1), :, :] = a[:, :65, :, :]
    # np.save(os.path.join(dir, 'adj_sub.npy'), A_sub)
    # flow = np.load('/public/lhy/wmy/myMRA-GCN/data/PEMSBAY/pems-bay.npy').flatten()
    # print(np.sum(flow))
    pass
