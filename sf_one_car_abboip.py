# 本脚本试图处理./cabspottingdata下的原生数据


import pandas as pd
import geopandas as gpd
import movingpandas as mpd
import folium
from folium.plugins import TimestampedGeoJson, AntPath
from datetime import datetime
import numpy as np

# from utils import get_border

def main():

    # step. 读txt并处理为df
    taxi_id = 'abboip'
    df = pd.read_csv(f"./cabspottingdata/new_{taxi_id}.txt", header=None, sep=" ")
    df.columns = ['latitude', 'longitude', 'occupancy', 't']
    df.pop('occupancy')  # drop无关列occupancy
    df.insert(0, 'id', [taxi_id for _ in range(df.shape[0])])  # 插入新列：id

    print('get df, columns=[latitude, longitude, t]')

    # step. 提取某个时间范围的数据
    df.t = df.t.apply(lambda x: datetime.fromtimestamp(x))  # 时间戳转datetime
    # tmp = df['t'].apply(lambda x: datetime.date(x))
    # tmp.value_counts()
    # 2008-05-21    1505
    # 2008-05-20    1316
    # 2008-06-04    1294
    # 2008-06-05    1230
    # 2008-05-18    1208
    # ...
    chosen_index = df.t.apply(lambda x: (x.month == 5 and x.day == 18))  # option：仅保留一天的数据
    df = df[chosen_index]
    df = df.sort_values(by=['t'], ascending=[True])  # 按t升序排序


    print('now df columns=[latitude, longitude, id], index=t')

    # step. 根据df创建gdf和mpd.trajs
    df = df.set_index('t')  # 以t为index
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4236)
    trajs = mpd.TrajectoryCollection(gdf, 'id')

    # step. 画图
    start_point = trajs.trajectories[0].get_start_location()
    m = folium.Map(location=[start_point.y, start_point.x], tiles="cartodbpositron", zoom_start=14)  # 经纬度反向
    m.add_child(folium.LatLngPopup())
    minimap = folium.plugins.MiniMap()
    m.add_child(minimap)
    folium.TileLayer('OpenStreetMap').add_to(m)

    for index, traj in enumerate(trajs.trajectories):
        name = f"Taxi {traj.df['id'].iloc[0]}"  # 轨迹名称
        print(traj.df.shape)
        randr = lambda: np.random.randint(0, 255)
        color = '#%02X%02X%02X' % (randr(), randr(), randr())  
        # line
        geo_col = traj.to_point_gdf().geometry
        xy = [[y, x] for x, y in zip(geo_col.x, geo_col.y)]
        f1 = folium.FeatureGroup(name)
        AntPath(locations=xy, color=color, weight=3, opacity=0.7, dash_array=[20, 30]).add_to(f1)
        f1.add_to(m)
        # break  # 只画一条轨迹

    folium.LayerControl().add_to(m)

    m.get_root().render()
    m.get_root().save("taxi_test_page.html")

    print(1)


if __name__ == '__main__':
    main()
