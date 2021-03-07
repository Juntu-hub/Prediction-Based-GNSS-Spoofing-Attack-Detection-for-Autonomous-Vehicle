import os
import numpy as np
import pandas as pd
import pyproj
from scipy.interpolate import make_interp_spline
import math
import matplotlib.pyplot as plt
# a = 6378137
# b = 6356752.3142
# esq = 6.69437999014 * 0.001
# e1sq = 6.73949674228 * 0.001
# def ecef2geodetic(ecef, radians=False):
#   """
#   Convert ECEF coordinates to geodetic using ferrari's method
#   """
#   # Save shape and export column
#   ecef = np.atleast_1d(ecef)
#   input_shape = ecef.shape
#   ecef = np.atleast_2d(ecef)
#   x, y, z = ecef[:, 0], ecef[:, 1], ecef[:, 2]
#
#   ratio = 1.0 if radians else (180.0 / np.pi)
#
#   # Conver from ECEF to geodetic using Ferrari's methods
#   # https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#Ferrari.27s_solution
#   r = np.sqrt(x * x + y * y)
#   Esq = a * a - b * b
#   F = 54 * b * b * z * z
#   G = r * r + (1 - esq) * z * z - esq * Esq
#   C = (esq * esq * F * r * r) / (pow(G, 3))
#   S = np.cbrt(1 + C + np.sqrt(C * C + 2 * C))
#   P = F / (3 * pow((S + 1 / S + 1), 2) * G * G)
#   Q = np.sqrt(1 + 2 * esq * esq * P)
#   r_0 =  -(P * esq * r) / (1 + Q) + np.sqrt(0.5 * a * a*(1 + 1.0 / Q) - \
#         P * (1 - esq) * z * z / (Q * (1 + Q)) - 0.5 * P * r * r)
#   U = np.sqrt(pow((r - esq * r_0), 2) + z * z)
#   V = np.sqrt(pow((r - esq * r_0), 2) + (1 - esq) * z * z)
#   Z_0 = b * b * z / (a * V)
#   h = U * (1 - b * b / (a * V))
#   lat = ratio*np.arctan((z + e1sq * Z_0) / r)
#   lon = ratio*np.arctan2(y, x)
#
#   # stack the new columns and return to the original shape
#   geodetic = np.column_stack((lat, lon, h))
#   return geodetic.reshape(input_shape)
# 在字符串指定位置的插入字符
def str_insert(str_origin, pos, str_add):
    str_list = list(str_origin)    # 字符串转list
    str_list.insert(pos, str_add)  # 在指定位置插入字符串
    str_out = ''.join(str_list)    # 空字符连接
    return  str_out

# 数组去重
def unique(old_list):
    newList = []
    # 判断相邻时间是否相等
    if np.any(old_list[1:] == old_list[:-1]):
        for x in old_list:
            if x in newList:
                # 若相等，则加上一个微小的数使其不等
                x = x + 0.005
            newList.append(x)
        return np.array(newList)
    else: return old_list

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    column_names = ['lats', 'lons', 'CAN_speeds', 'steering_angles', 'acceleration_forward']
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('%s(t-%d)' % (j, i)) for j in column_names]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('%s(t)' % (j)) for j in column_names]
        else:
            names += [('%s(t+%d)' % (j, i)) for j in column_names]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

EARTH_REDIUS = 6378.137

def rad(d):
    return d * math.pi / 180.0
def getDistance(lat1, lng1, lat2, lng2):
    # 对数组取元素做运算
    res = []
    for i in range(len(lat1)):
        radLat1 = rad(lat1[i])
        radLat2 = rad(lat2[i])
        a = radLat1 - radLat2
        b = rad(lng1[i]) - rad(lng2[i])
        s = 2 * math.asin(math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(radLat1) * math.cos(radLat2) * math.pow(
            math.sin(b / 2), 2)))
        s = s * EARTH_REDIUS * 1000
        res.append(s)
    return res

if __name__ == '__main__':

    # 用于GNSS坐标转化
    position_transformer = pyproj.Transformer.from_crs(
                {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
                {"proj": 'latlong', "ellps": 'WGS84', "datum": 'WGS84'},
            )
    dataset_directory = 'D:\comma2k19'
    chunk_set = []
    for chunk in os.listdir(dataset_directory):
        # 忽略生成的csv文件
        if ".csv" in chunk:
            continue
        # 如果序号为单个时在前补零，以便后面排序
        if len(chunk) == 7:
            used_name = chunk
            chunk = str_insert(chunk,6,'0')
            os.rename(os.path.join(dataset_directory, used_name), os.path.join(dataset_directory, chunk))
        chunk_set.append(os.path.join(dataset_directory, chunk))
    # 将序号小的片段放在前面
    chunk_set.sort()
    # 选一个chunk来训练（200分钟）
    chunk_index = 0
    route_set = []
    for route_id in os.listdir(chunk_set[chunk_index]):
        # 忽略生成的csv文件
        if ".csv" in route_id:
            continue
        route_set.append(os.path.join(chunk_set[chunk_index], route_id))
    segment_set = []
    # 选一个路段训练
    route_index = 9
    for segment in os.listdir(route_set[route_index]):
        # 如果序号为单个时在前补零，以便后面排序
        if len(segment) == 1:
            used_name = segment
            segment = '0'+segment
            os.rename(os.path.join(route_set[route_index], used_name),os.path.join(route_set[route_index], segment))
        segment_set.append(os.path.join(route_set[route_index], segment))
    # 将序号小的片段放在前面
    segment_set.sort()
    times = []
    lons = []
    lats = []
    orientations = []
    CAN_speeds = []
    steering_angles = []
    acceleration_forward = []
    for main_dir in segment_set:
        # 导入GNSS的时间和位置(pose)并将位置转化为经纬度
        temp_GNSS_time = np.load(main_dir + '\\global_pose\\frame_times')
        times = np.append(times, temp_GNSS_time)
        # 打印每一段的长度
        print(len(temp_GNSS_time))
        positions = np.load(main_dir + '\\global_pose\\frame_positions')
        positions = position_transformer.transform(positions[:, 0], positions[:, 1], positions[:, 2], radians=False)
        lats = np.append(lats, positions[1])
        lons = np.append(lons, positions[0])
        # Conver from ECEF to geodetic using Ferrari's methods
        # positions = ecef2geodetic(positions)
        # lats = np.append(lats, positions[:, 0])
        # lons = np.append(lons, positions[:, 1])
        # 暂时不用orientation
        # orientation = np.load(main_dir + '\\global_pose\\frame_orientations')
        # orientations = np.append(orientations, np.load(main_dir + '\\global_pose\\frame_orientations'))
        temp_CAN_times = np.load(main_dir + '\\processed_log\\CAN\\speed\\t')
        # 确保时间无重复值
        temp_CAN_speed_times = unique(temp_CAN_times)
        # 对CAN数据按照GNSS参考时间插值
        temp_CAN_speeds = make_interp_spline(temp_CAN_speed_times, np.load(main_dir + '\\processed_log\\CAN\\speed\\value'))(temp_GNSS_time).flatten()
        CAN_speeds = np.append(CAN_speeds, temp_CAN_speeds)
        # CAN_angles_times和CAN_speed_times有时不一致
        temp_CAN_angles_times = np.load(main_dir + '\\processed_log\\CAN\\steering_angle\\t')
        temp_steering_angles = np.load(main_dir + '\\processed_log\\CAN\\steering_angle\\value')
        temp_CAN_angles_times = unique(temp_CAN_angles_times)
        temp_steering_angles = make_interp_spline(temp_CAN_angles_times, temp_steering_angles)(temp_GNSS_time)
        steering_angles = np.append(steering_angles, temp_steering_angles)
        # 对IMU数据按照GNSS参考时间插值
        temp_IMU_times = np.load(main_dir + '\\processed_log\\IMU\\accelerometer\\t')
        temp_acceleration_forward = make_interp_spline(temp_IMU_times, np.load(main_dir +
                                '\\processed_log\\IMU\\accelerometer\\value')[:, 0])(temp_GNSS_time)
        acceleration_forward = np.append(acceleration_forward, temp_acceleration_forward)

    DataSet = list(zip(times, lats, lons, CAN_speeds, steering_angles, acceleration_forward))
    column_names = ['times', 'lats', 'lons', 'CAN_speeds', 'steering_angles', 'acceleration_forward']
    df = pd.DataFrame(data=DataSet, columns=column_names)
    times = df['times'].values
    df = df.set_index(['times'], drop=True)
    values = df.values.astype('float64')
    # 转为监督学习问题
    reframed = series_to_supervised(values, 1, 1)
    # 计算距离
    lons_t = reframed['lons(t)'].values
    lats_t = reframed['lats(t)'].values
    distance = np.array(getDistance(lats[:-1], lons[:-1], lats_t, lons_t))
    # drop columns we don't want to predict including（CAN_speed,steering_angel, acceleration_forward)
    reframed.drop(reframed.columns[[0, 1, 5, 6, 7, 8, 9]], axis=1, inplace=True)
    # 时间和计算的距离添加到数据集
    reframed['distance'] = distance
    reframed['times'] = times[: -1]
    # for i in distance:
    #     if i > 100:
    #         print(i)
    plt.plot(times[:-1], distance)
    plt.xlabel('Boot time (s)', fontsize=18)
    plt.ylabel('Distance travelled during single timestamp (m) ', fontsize=12)
    plt.show()
    # 将合并的数据集保存到.csv文件中
    reframed.to_csv(route_set[route_index]+".csv", index=False, sep=',')


