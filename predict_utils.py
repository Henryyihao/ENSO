# predict_utils.py — Nino3.4 标量输出版
"""
纯未来预测可视化工具 (标量版)
用于 --stage predict: 没有真实数据, 只展示 Nino3.4 标量预测结果
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_future_nino34(nino_pred, future_dates, save_dir, title_suffix="",
                       obs_nino34=None, obs_dates=None):
    """
    nino_pred   : (T,) Nino3.4 标量预测 (物理量 deg C)
    future_dates: list of pd.Timestamp
    obs_nino34  : (T_obs,) 真实 Nino3.4 (可选)
    obs_dates   : list of pd.Timestamp (可选)
    """
    has_obs = (obs_nino34 is not None and obs_dates is not None
               and len(obs_nino34) > 0 and len(obs_dates) > 0)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)

    ax.fill_between(future_dates, 0.5, 3.0, alpha=0.08, color='red', label='El Nino (>0.5 deg C)')
    ax.fill_between(future_dates, -3.0, -0.5, alpha=0.08, color='blue', label='La Nina (<-0.5 deg C)')

    if has_obs:
        ax.plot(obs_dates, obs_nino34, color='#2C3E50', linewidth=2.0,
                marker='s', markersize=4.5, zorder=4,
                label='Observed Nino3.4', linestyle='-')
        ax.axvline(obs_dates[-1], color='gray', linestyle=':', linewidth=1.2,
                   alpha=0.8, label=f'Obs end ({obs_dates[-1].strftime("%Y-%m")})')

    ax.plot(future_dates, nino_pred, color='#C0392B', linewidth=2.2,
            marker='o', markersize=5, zorder=5, label='Predicted Nino3.4')

    for d, v in zip(future_dates, nino_pred):
        if not np.isnan(v):
            ax.annotate(f'{v:.2f}', (d, v), textcoords='offset points',
                        xytext=(0, 8), fontsize=7, ha='center', color='#C0392B')

    ax.axhline(0.5, color='red', linestyle='--', linewidth=1.0, alpha=0.6)
    ax.axhline(-0.5, color='blue', linestyle='--', linewidth=1.0, alpha=0.6)
    ax.axhline(0.0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('Forecast Month', fontsize=11)
    ax.set_ylabel('Nino3.4 Index (deg C)', fontsize=11)
    title = f'Future Nino3.4 Forecast'
    if title_suffix:
        title += f' -- {title_suffix}'
    ax.set_title(title, fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate(rotation=30)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)

    all_vals = list(nino_pred[~np.isnan(nino_pred)])
    if has_obs:
        all_vals += list(obs_nino34[~np.isnan(obs_nino34)])
    y_min = min(min(all_vals) - 0.5, -1.5) if all_vals else -1.5
    y_max = max(max(all_vals) + 0.5, 1.5) if all_vals else 1.5
    ax.set_ylim(y_min, y_max)

    save_path = os.path.join(save_dir, 'future_nino34_forecast.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  [Predict] Nino3.4 time series saved -> {save_path}")


def save_future_forecast_csv(nino_pred, future_dates, save_dir):
    """保存预测 CSV"""
    df = pd.DataFrame({
        'date': [d.strftime('%Y-%m') for d in future_dates],
        'lead_month': list(range(1, len(future_dates) + 1)),
        'nino34_pred': nino_pred,
    })
    csv_path = os.path.join(save_dir, 'future_nino34_forecast.csv')
    df.to_csv(csv_path, index=False)
    print(f"  [Predict] Nino3.4 CSV saved -> {csv_path}")


def run_all_predict_plots(nino_pred, future_dates, save_dir,
                           title_suffix="",
                           obs_nino34=None, obs_dates=None):
    """
    统一入口 (标量版)
    nino_pred   : (T,) 物理量 Nino3.4 预测
    future_dates: list of pd.Timestamp
    """
    os.makedirs(save_dir, exist_ok=True)

    plot_future_nino34(nino_pred, future_dates, save_dir,
                       title_suffix=title_suffix,
                       obs_nino34=obs_nino34, obs_dates=obs_dates)

    save_future_forecast_csv(nino_pred, future_dates, save_dir)

    print(f"\n[Predict] All outputs saved to: {save_dir}")
