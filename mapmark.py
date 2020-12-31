import folium
import pandas as pd
import numpy as np
from utils import ReadExcelFile, stackedBar


def accident(incident_Bay,
             station_bay_target):
    m = folium.Map(
        location=[37.3478, -121.9771],
        tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}&ltype=6',
        attr='AutoNavi'
    )
    m.add_child(folium.LatLngPopup())  # 显示鼠标点击点经纬度
    df_incident_Bay = pd.read_csv(incident_Bay)
    df_station_bay_target = pd.read_csv(station_bay_target)
    for i in range(len(df_station_bay_target)):
        folium.Marker([df_station_bay_target.iloc[i]['Latitude'], df_station_bay_target.iloc[i]['Longtitude']]).add_to(
            m)
    for i in range(len(df_incident_Bay)):
        if df_incident_Bay.iloc[i]['dis'] < 1:
            folium.CircleMarker([df_incident_Bay.iloc[i]['Latitude'], df_incident_Bay.iloc[i]['Longtitude']],
                                color='Red',
                                fill=True,
                                fill_color='Red').add_to(m)

    m.save('pems-bay_accident.html')


if __name__ == '__main__':
    accident = pd.read_csv('/public/lhy/wmy/dataset/Accident/accident_Pems-bay.csv')
    incident = pd.read_csv('/public/lhy/wmy/dataset/Pems/incidents/incident_pems-bay.csv')
    pass
    # accident('incident_pems-bay.csv',
    #          'sensors_pems_bay.csv')
