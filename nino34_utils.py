"""
nino34_utils.py — Nino3.4 区域坐标工具（60S-60N 数据版）
================================================
提供统一的 Nino3.4 区域索引查找，适配 60S-60N 经纬度网格。
所有需要定位 Nino3.4 区域的模块（Loss、Model、utils）统一调用此文件。

Nino3.4 定义：5S-5N, 170W-120W
  纬度范围: [-5, 5]
  经度范围: [190, 240]  (0-360 系统)
           或 [-170, -120] (±180 系统)
"""

import numpy as np


def find_nino34_indices(lat: np.ndarray, lon: np.ndarray):
    """
    根据实际经纬度坐标，查找 Nino3.4 区域的格点索引范围。

    Parameters
    ----------
    lat : np.ndarray, shape (H,)  纬度数组（度）
    lon : np.ndarray, shape (W,)  经度数组（度）

    Returns
    -------
    lat_s, lat_e, lon_s, lon_e : int
        Nino3.4 区域在网格中的切片索引 [lat_s:lat_e, lon_s:lon_e]
    """
    # 纬度范围: 5S-5N
    lat_mask = (lat >= -5) & (lat <= 5)
    lat_indices = np.where(lat_mask)[0]
    if len(lat_indices) == 0:
        raise ValueError(f"No latitude points found in Nino3.4 range [-5, 5]. "
                         f"Lat range: [{lat.min():.1f}, {lat.max():.1f}]")
    lat_s = int(lat_indices[0])
    lat_e = int(lat_indices[-1]) + 1

    # 经度范围: 170W-120W
    # 处理 0-360 和 ±180 两种经度系统
    if np.any(lon > 180):
        # 0-360 系统: 170W=190, 120W=240
        lon_mask = (lon >= 190) & (lon <= 240)
    else:
        # ±180 系统: 170W=-170, 120W=-120
        lon_mask = (lon >= -170) & (lon <= -120)

    lon_indices = np.where(lon_mask)[0]
    if len(lon_indices) == 0:
        raise ValueError(f"No longitude points found in Nino3.4 range. "
                         f"Lon range: [{lon.min():.1f}, {lon.max():.1f}]")
    lon_s = int(lon_indices[0])
    lon_e = int(lon_indices[-1]) + 1

    return lat_s, lat_e, lon_s, lon_e


def nino34_mask_bool(lat: np.ndarray, lon: np.ndarray):
    """
    返回布尔掩码形式的 Nino3.4 区域。

    Returns
    -------
    lat_mask : np.ndarray, shape (H,), dtype=bool
    lon_mask : np.ndarray, shape (W,), dtype=bool
    """
    lat_mask = (lat >= -5) & (lat <= 5)
    if np.any(lon > 180):
        lon_mask = (lon >= 190) & (lon <= 240)
    else:
        lon_mask = (lon >= -170) & (lon <= -120)
    return lat_mask, lon_mask
