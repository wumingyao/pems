import pandas as pd

if __name__ == '__main__':
    df_incidents = pd.read_csv('/public/lhy/wmy/dataset/Pems/all_text_chp_incidents_month_2017_01.txt', names=(
        'Incident_ID', 'CC_Code', 'Incident_Number', 'Timestamp', 'Description', 'Location', 'Area', 'Zoom_Map',
        'TB_xy',
        'Latitude', 'Longtitude', 'District', 'Country_FIPS_ID', 'City_FIPS_ID', 'Freeway_Number', 'Freeway_Direction',
        'State_Postmile', 'Absolute_Postmile',
        'Severity', 'Duration'), sep=',')
    df_incidents = df_incidents[df_incidents['District'] == 4].sort_values('Freeway_Number')
    # df_traffic = pd.read_csv('/public/lhy/wmy/dataset/Pems/d03_text_station_5min_2020_01_01.txt', names=(
    # 'Timestamp', 'Station', 'District', 'Freeway', 'Direction_of_Travel', 'Lane_Type', 'Station_Length', 'Samples',
    # '%Observed', 'Total_Flow', 'Avg_Occupancy', 'Avg_Speed', 'Lane_N_Samples', 'Lane_N_Flow', 'Lane_N_Avg_Occ',
    # 'Lane_N_Avg_Speed', 'Lane_N_Observed'), sep=',')
    df_station = pd.read_csv('/public/lhy/wmy/dataset/Pems/station/d04_text_meta_2020_08_20.txt', sep='\t').sort_values(
        'Fwy')
    pass
