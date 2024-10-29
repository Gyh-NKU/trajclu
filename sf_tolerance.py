# 相比sf.py的改进：
# 1. 设置tolerance。为什么在时间对齐前检查toletance呢？逻辑是如果如果对齐前能通过，对齐后更能通过。所以这种筛选不可能得到异常数据
# 2. 更小的有效范围，仅选中核心城区. 重新设置网格数，并换算无人机以18m/s的速度，一个timestamp移动多少格子
# 3. 对齐时间戳到1519894800~1519898400



import pandas as pd
import geopandas as gpd
import movingpandas as mpd
import folium
from folium.plugins import TimestampedGeoJson, AntPath
from datetime import datetime
import numpy as np
import osmnx as ox
import networkx as nx
import warnings
from shapely.geometry import Point
from movingpandas.geometry_utils import measure_distance_geodesic

from utils import get_border, get_datetime_chime, folium_draw, check_tolerance, x_into_region, y_into_region

warnings.simplefilter("ignore")

def gen_raw_data():
    # step. 获取id列表
    fp = open('./cabspottingdata/_cabs.txt', 'r')
    lines = fp.readlines()
    id_list = [line.split("\"")[1] for line in lines]

    # step. 读所有txt，并处理为df
    raw_df = pd.DataFrame()
    s = 1
    for id in id_list:
        df = pd.read_csv(f"./cabspottingdata/new_{id}.txt", header=None, sep=" ")
        df.columns = ['latitude', 'longitude', 'occupancy', 't']
        df.pop('occupancy')  # drop无关列occupancy
        df.insert(0, 'id', [id for _ in range(df.shape[0])])  # 插入新列：id
        raw_df = pd.concat([raw_df, df], axis=0)  # 拼接

        print('Finished merging {}/{}'.format(s, len(id_list)))
        s += 1

    raw_df = raw_df.sort_values(by=['id', 't'], ascending=[True, True])  # 按id和t升序排序
    raw_df = raw_df.set_index('t')  # 以t为index

    print('get raw_df, columns=[latitude, longitude, id], index=t')

    # step. 将包含所有车的原始数据写入./data/sf_tolerance/raw_data.csv
    raw_df.to_csv('./data/sf_tolerance/raw_data.csv')


def select_region(df, lower_left, upper_right):
    chosen_index = df.longitude.apply(lambda x: x > lower_left[0] and x < upper_right[0])
    df = df[chosen_index]
    chosen_index = df.latitude.apply(lambda x: x > lower_left[1] and x < upper_right[1])
    df = df[chosen_index]
    return df


def select_hour(groupDf, best_hour):
    # step. 得到该车的top列表
    hours = groupDf.t.apply(lambda x: x - (x % 3600))
    hours = hours.apply(lambda x: datetime.fromtimestamp(x))  # 时间戳转datetime
    top = list(hours.value_counts().index)  # 该车数据最多的小时的排序列表

    # step. 依次检查 选出该车既数据丰富又通过tolerance检验的小时
    groupDf.t = groupDf.t.apply(lambda x: datetime.fromtimestamp(x))  # 时间戳转datetime
    if best_hour in top[0:10]:  # case1.全局最佳小时在前10名，检查该小时
        chosen_hour = best_hour
        chosen_index = groupDf.t.apply(lambda x: (x.month == chosen_hour.month and
                                              x.day == chosen_hour.day and x.hour == chosen_hour.hour))
        candDf = groupDf[chosen_index]  # 候选子表
        if check_tolerance(candDf):  # 通过tolerance检验
            print('for car {}, chosen_hour = {}'.format(list(groupDf.id)[0], chosen_hour))
            print('该小时是最佳小时')
            candDf.pop('id')  # 删除id属性，因为groupBy后会以id为一级索引
            return candDf

    for i in range(len(top)):  # case2.全局最佳小时不在前10名，或全局最佳小时未通过tolerance检验，则在top依次检查该车数据多的小时
        chosen_hour = top[i]
        chosen_index = groupDf.t.apply(lambda x: (x.month == chosen_hour.month and
                                                  x.day == chosen_hour.day and x.hour == chosen_hour.hour))
        candDf = groupDf[chosen_index]  # 候选子表
        if check_tolerance(candDf):  # 通过tolerance检验
            print('for car {}, chosen_hour = {}'.format(list(groupDf.id)[0], chosen_hour))
            print('该小时是该车的top{}'.format(i))
            candDf.pop('id')  # 删除id属性，因为groupBy后会以id为一级索引
            return candDf

    print('for car {}, 该车没有不瞬移的小时...'.format(list(groupDf.id)[0]))
    return pd.DataFrame()  # 上述两种情况都未选出子表，则返回空表



def round15(x):
    # 商15的余数为1~7时下舍，为8~14时上入
    r = x % 15
    x = x - r if r <= 7 else x + (15 - r)
    return x


def slot_and_align_in_period(groupDf):
    groupDf.t = groupDf.t.apply(lambda x: datetime.timestamp(x))  # datetime转时间戳
    groupDf.t = groupDf.t.apply(round15)  # 近似为15s的倍数
    groupDf.t = groupDf.t.apply(lambda x: datetime.fromtimestamp(x))  # 时间戳转datetime

    startDateTime = list(groupDf.t)[0]
    startHour = get_datetime_chime(startDateTime, step=0, rounding_level='hour')
    endHour = get_datetime_chime(startDateTime, step=1, rounding_level='hour')
    # dateSpan = pd.date_range('20080518090000', '20080518100000', freq='15s')  # 这里要改 适配每辆车的具体时间范围
    dateSpan = pd.date_range(startHour, endHour, freq='15s')
    dateSpan = pd.DataFrame({'t': dateSpan})

    slottedGroupDf = pd.merge(dateSpan, groupDf, how="left", on="t")
    # 删除在t上的重复值，仅保留第一次出现重复值的行
    slottedGroupDf.drop_duplicates(subset=['t'], keep='first', inplace=True)
    slottedGroupDf.pop('id')  # 删除id属性，因为groupBy后会以id为一级索引

    print('Finished slotting a car')
    return slottedGroupDf


def fill_nan_by_shortest_path(groupDf, G, nodes_gdf, edges_gdf):
    longitudes = list(groupDf.longitude)
    latitudes = list(groupDf.latitude)

    # 1. 获取非空元素的下标
    nan_index = np.where(np.isnan(latitudes))[0]
    nonan_index = [i for i in range(len(latitudes)) if i not in nan_index]
    # nonan_index = [1, 5, 8, 10, 13,...,237]

    # 2. 填充头和尾部缺失值
    # 头部。 第一个非nan下标为1时，需填充1个头部缺失值
    longitudes[0: nonan_index[0]] = [longitudes[nonan_index[0]] for _ in range(nonan_index[0])]
    latitudes[0: nonan_index[0]] = [latitudes[nonan_index[0]] for _ in range(nonan_index[0])]
    # 尾部。 最后一个非nan下标为239时，需填充1个缺失值
    longitudes[nonan_index[-1] + 1:] = [longitudes[nonan_index[-1]] for _ in
                                        range(len(latitudes) - nonan_index[-1] - 1)]
    latitudes[nonan_index[-1] + 1:] = [latitudes[nonan_index[-1]] for _ in range(len(latitudes) - nonan_index[-1] - 1)]

    def get_shortest_path(origin_point, destination_point):
        origin_node = ox.get_nearest_node(G, origin_point)  # 得到距离源位置最近的节点
        destination_node = ox.get_nearest_node(G, destination_point)
        route = nx.shortest_path(G, origin_node, destination_node, weight='length')  # 计算最短路
        return route

    def route_total_length(edges_gdf, route):
        totalLength = 0
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            totalLength += edges_gdf.loc[(u, v, 0)].geometry.length
        return totalLength

    def get_lat_lon_of_quantile(edges_gdf, route, totalLength, quantile):
        # quantile: 分位点，0.5即中点，0.33即距起点更近的三分位点
        resLength = quantile * totalLength  # 起点到分位点的长度
        x, y = G.nodes[route[-1]]['x'], G.nodes[route[-1]]['y']  # 下述循环未对x,y定值时的兜底定值，即路径上最后一个点
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            curEdgeLength = edges_gdf.loc[(u, v, 0)].geometry.length
            if resLength > curEdgeLength:  # 遍历下一条边
                resLength -= curEdgeLength
            else:  # 分位点在当前边上
                u_x, u_y = G.nodes[u]['x'], G.nodes[u]['y']
                v_x, v_y = G.nodes[v]['x'], G.nodes[v]['y']
                x = u_x + (resLength / curEdgeLength) * (v_x - u_x)
                y = u_y + (resLength / curEdgeLength) * (v_y - u_y)
                break

        assert x != -1 and y != -1
        return x, y  # 经度，纬度

    # 3. 填充中间缺失值
    for i in range(len(nonan_index) - 1):  # 遍历需填充的区间
        k = nonan_index[i + 1] - nonan_index[i] - 1  # 当前区间待填充的缺失值个数
        if k == 0: continue
        ori = (latitudes[nonan_index[i]], longitudes[nonan_index[i]])
        dst = (latitudes[nonan_index[i + 1]], longitudes[nonan_index[i + 1]])
        try:
            route = get_shortest_path(ori, dst)
            totalLength = route_total_length(edges_gdf, route)
            # 最短路总长度 UnboundLocalError: local variable 'route' referenced before assignment

            fill_lon = []
            fill_lat = []
            for j in range(1, k + 1):  # 遍历每个待填充的缺失值
                lon, lat = get_lat_lon_of_quantile(edges_gdf, route, totalLength, j / (k + 1))
                fill_lon.append(lon)
                fill_lat.append(lat)
            # 填充区间
            longitudes[nonan_index[i] + 1: nonan_index[i + 1]] = fill_lon
            latitudes[nonan_index[i] + 1: nonan_index[i + 1]] = fill_lat

        except:  # 若get_shortest_path无法找到最短路，或出现其他意外情况，则用源点填充缺失值
            print('try语句出现异常，可能是无法找到最短路')
            longitudes[nonan_index[i] + 1: nonan_index[i + 1]] = [ori[1] for _ in range(k)]
            latitudes[nonan_index[i] + 1: nonan_index[i + 1]] = [ori[0] for _ in range(k)]


    groupDf.longitude = longitudes
    groupDf.latitude = latitudes


    print('填充完毕一辆车的缺失值！')
    print('填充后在241条数据中，有{}条非空数据'.format(
        241 - groupDf.latitude.isnull().sum())
    )
    return groupDf


def count_nonan(groupDf):
    print('在241条数据中，有{}条非空数据'.format(
        241 - groupDf.latitude.isnull().sum())
    )

def get_same_time_span(groupDf):
    # 对齐到和purdue相同的时间戳
    groupDf.t = pd.date_range(datetime.fromtimestamp(1519894800), datetime.fromtimestamp(1519898400), freq='15s')
    return groupDf

def delete_last(groupDf):
    groupDf = groupDf.iloc[0:groupDf.shape[0]-1]
    groupDf.pop('id')
    return groupDf

def func(groupDf, id_list):
    if list(groupDf.id)[0] in id_list:
        return groupDf


def main():

    # 设置地图基本信息（粗略缩小至城区范围）
    lower_left = [-122.4620, 37.7441]
    upper_right = [-122.3829, 37.8137]

    '''
    # 一、根据原始txt文件生成原始数据。若已经得到raw_data则跳过
    # gen_raw_data()
    '''

    # 二、预处理。若已经得到processed_data则跳过
    raw_df = pd.read_csv('./data/sf_tolerance/raw_data.csv')
    raw_df.set_index('t')
    # step. 限定区域
    print("raw_df.shape before select region:", raw_df.shape)
    raw_df = select_region(raw_df, lower_left, upper_right)
    print("raw_df.shape after select region:", raw_df.shape)

    # step.在原始数据上选出一个数据最丰富的小时
    # method1 人为指定2018.5.18 9~10小时
    # raw_df.t = raw_df.t.apply(lambda x: datetime.fromtimestamp(x))  # 时间戳转datetime
    # chosen_index = raw_df.t.apply(lambda x: (x.month == 5 and x.day == 18 and x.hour == 9))  # option：仅保留一天的数据
    # raw_df = raw_df[chosen_index]
    # method2
    # 1） 选出在所有车中数据量最多的小时best_hour
    hours = raw_df.t.apply(lambda x: x - (x % 3600))
    hours = hours.apply(lambda x: datetime.fromtimestamp(x))  # 时间戳转datetime
    best_hour = hours.value_counts().index[0]
    # 2） 对于每辆车，若best_hour in top10，则该车选用best_hour;否则选用该车自己数据量最多的小时。
    raw_df = raw_df.groupby('id').apply(select_hour, best_hour)  # best_hour是参数
    raw_df = raw_df.reset_index()  # 分组操作后，取消二级索引
    raw_df.pop('level_1')  # 删除多余的level_1
    print('Finished selecting hour')

    # step. 分组+等差时间轴，每辆车得到241条数据
    raw_df = raw_df.groupby('id').apply(slot_and_align_in_period)
    raw_df = raw_df.reset_index()  # 分组操作后，取消二级索引
    raw_df.pop('level_1')  # 删除多余的level_1
    # 手动查看每辆车在时间轴上非空数据的条数。
    raw_df.groupby('id').apply(count_nonan)
    raw_df.to_csv('./data/sf_tolerance/processed_data.csv')  # 保存数据
    print('Finished slotting and aligning hour')

    
    # 三、补全缺失值，得到processed_fillna_data。预估时间4.5h
    raw_df = pd.read_csv('./data/sf_tolerance/processed_data.csv')
    raw_df.set_index('t')

    # step. 使用osmnx，下载旧金山市的路网（若已经得到map.graphml则跳过）
    # G = ox.graph_from_place('San Francisco, California, USA', which_result=2)  # 暂不明确which_result参数的含义
    # ox.save_graphml(G, "./data/sf_tolerance/map.graphml")
    G = ox.load_graphml('./data/sf_tolerance/map.graphml')
    # ox.plot_graph(G)
    # step. 补全缺失值。
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G, nodes=True, edges=True)  # 不变代码外提
    raw_df = raw_df.groupby('id').apply(fill_nan_by_shortest_path, G, nodes_gdf, edges_gdf)
    # step. 对于超出矩形范围的缺失值，将其放入矩形范围（7.13 debug新增）
    raw_df.longitude = raw_df.longitude.apply(x_into_region, lower_left, upper_right)
    raw_df.latitude = raw_df.latitude.apply(y_into_region, lower_left, upper_right)
    raw_df.to_csv('./data/sf_tolerance/processed_fillna_data.csv')  # 保存数据
    print('Finished filling nan by shortest path')

    '''
    # 四、根据processed_train_data，绘制前端
    raw_df = pd.read_csv('./data/sf_tolerance/processed_fillna_data.csv', index_col=0)
    raw_df.pop('Unnamed: 0.1')  #
    raw_df.pop('geometry')

    # 根据raw_df，创建gdf和mpd.trajs，并使用folium画图
    folium_draw(lower_left, upper_right, raw_df)  # TODO 调用画图过程
    

    # 五、网格化，并将所有车的时间统一为2018.5.18 15~16。得到最终processed_train_data
    # raw_df = pd.read_csv('./data/sf_tolerance/processed_fillna_data.csv', index_col=0)
    nlon = 7910
    nlat = 6960
    max_distance_x = measure_distance_geodesic(Point(lower_left[0], lower_left[1]),
                                               Point(upper_right[0], lower_left[1]))
    max_distance_y = measure_distance_geodesic(Point(lower_left[0], lower_left[1]),
                                               Point(lower_left[0], upper_right[1]))

    longitude_boundaries = np.linspace(lower_left[0], upper_right[0], nlon + 1)  # 经度
    latitude_boundaries = np.linspace(lower_left[1], upper_right[1], nlat + 1)  # 纬度
    raw_df['x'] = pd.cut(raw_df.longitude, bins=longitude_boundaries,
                         labels=range(longitude_boundaries.shape[0] - 1),  # labels是输出的结果
                         right=False).astype(float)
    raw_df['x_distance'] = raw_df['x'] * max_distance_x / nlon + max_distance_x / nlon / 2  # 网格化后，到左下角点的距离
    raw_df['y'] = pd.cut(raw_df.latitude, bins=latitude_boundaries,
                         labels=range(latitude_boundaries.shape[0] - 1),
                         right=False).astype(float)
    raw_df['y_distance'] = raw_df['y'] * max_distance_y / nlat + max_distance_x / nlat / 2  # 网格化后，到左下角点的距离

    # 时间统一
    raw_df = raw_df.groupby('id').apply(get_same_time_span)
    raw_df.t = raw_df.t.apply(lambda x: datetime.timestamp(x))
    raw_df.rename(columns={'t': 'timestamp'}, inplace=True)  # 修改列名:t -> timestamp
    raw_df.pop('geometry')
    raw_df.to_csv('./data/sf_tolerance/processed_train_data.csv')

    print(1)
    
    
    # 额外添加:仅保留每个车的前半个小时,并将id变为0,1,2...
    # step.
    raw_df = pd.read_csv('./data/sf_tolerance/processed_train_data.csv')
    raw_df.timestamp = raw_df.timestamp.apply(lambda x: datetime.fromtimestamp(x))
    chosen_index = raw_df.timestamp.apply(lambda x: (x.minute<30 or x.minute==30 and x.second==0))  # option：仅保留半个小时的数据
    raw_df = raw_df[chosen_index]
    raw_df.timestamp = raw_df.timestamp.apply(lambda x: datetime.timestamp(x))

    raw_df = raw_df.groupby('id').apply(delete_last)
    raw_df = raw_df.reset_index()
    raw_df.pop('level_1')

    # step. range索引是id在id_list中的下标
    fp = open('./cabspottingdata/_cabs.txt', 'r')
    lines = fp.readlines()
    id_list = [line.split("\"")[1] for line in lines]
    raw_df.id = raw_df.id.apply(lambda x: id_list.index(x))
    raw_df.to_csv('./data/sf_tolerance/processed_train_half_data.csv')
    print(1)
    '''

    # step. 随机抽样100辆车,并将id变为0,1,2...
    raw_df = pd.read_csv('data/sf_tolerance/processed_train_half_data.csv')

    # 1.抽样
    fp = open('./cabspottingdata/_cabs.txt', 'r')
    lines = fp.readlines()
    id_list = [line.split("\"")[1] for line in lines]
    import random
    list1 = random.sample(range(536), 100)
    raw_df = raw_df.groupby('id').apply(func, list1)
    raw_df = raw_df[raw_df['id'].notna()]
    raw_df.id = raw_df.id.apply(lambda x: int(x))


    # 2. 变为range索引，不再取下标
    ids = list(raw_df.id.value_counts().index)
    dict1 = dict(zip(ids, range(len(ids))))  # {143:0, 144:1, ... }
    raw_df.id = raw_df.id.apply(lambda x: dict1[x])
    raw_df = raw_df.sort_values(by='id')
    raw_df.pop("Unnamed: 0")
    raw_df.to_csv('./data/sf_tolerance/processed_train_half_100_data.csv')
    print(1)


if __name__ == '__main__':
    main()
