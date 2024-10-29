import pandas as pd
import geopandas as gpd
import movingpandas as mpd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from shapely.geometry import Point
from datetime import datetime, timedelta

# 创建示例轨迹数据
data = {
    "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
    "time": [
        datetime(2021, 1, 1, 8, 0),
        datetime(2021, 1, 1, 8, 10),
        datetime(2021, 1, 1, 8, 20),
        datetime(2021, 1, 1, 8, 0),
        datetime(2021, 1, 1, 8, 10),
        datetime(2021, 1, 1, 8, 20),
        datetime(2021, 1, 1, 8, 0),
        datetime(2021, 1, 1, 8, 10),
        datetime(2021, 1, 1, 8, 20)
    ],
    "x": [0, 1, 2, 0, 1, 2, 1, 2, 3],
    "y": [0, 1, 2, 0, 1, 2, 1, 2, 3],
}
df = pd.DataFrame(data)
df["geometry"] = df.apply(lambda row: Point(row["x"], row["y"]), axis=1)
gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
gdf.set_index("time", inplace=True)

# 使用MovingPandas创建轨迹集合
trajs = mpd.TrajectoryCollection(gdf, "id", t="time")


