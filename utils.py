# utils.py — Nino3.4 标量输出版
"""
仅保留与标量 Nino3.4 输出兼容的功能:
  - EarlyStopping
  - Nino3.4 技巧衰减曲线 (ACC / RMSE vs lead time)
  - Nino3.4 时间序列对比图
  - Nino3.4 季节性 SPB 热力图
  - Nino3.4 极端事件案例
  - 功率谱分析
  - Nino3.4 lead correlation
  - 季节-提前期热力图
  - 保存 CSV
"""

import torch
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from scipy.ndimage import gaussian_filter
from scipy.signal import welch
from matplotlib.ticker import MaxNLocator

INIT_MONTH = {'RechargeLatentNet', 'ENSOFormer', 'ROSEN'}


class EarlyStopping:
    def __init__(self, patience=7, verbose=True, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model, args, stats=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, args, stats)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'  [EarlyStopping] No improvement. Counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, args, stats)
            self.counter = 0
            if self.verbose:
                print(f'  [EarlyStopping] Validation loss decreased. Counter reset to 0/{self.patience}')

    def save_checkpoint(self, val_loss, model, args, stats):
        torch.save({
            'args': args,
            'model_state_dict': model.state_dict(),
            'stats': stats
        }, self.path)
        self.val_loss_min = val_loss


# ─────────────────────────────────────────────────────────────
# 反归一化 Nino3.4 标量
# ─────────────────────────────────────────────────────────────
def denorm_nino34(arr, stats):
    """将归一化的 Nino3.4 标量反归一化为物理量 (°C)"""
    mean = stats.get('nino34_mean', 0.0)
    std  = stats.get('nino34_std', 1.0)
    return arr * std + mean


# ─────────────────────────────────────────────────────────────
# 核心图: Nino3.4 技巧衰减 (ACC + RMSE vs Lead Time)
# ─────────────────────────────────────────────────────────────
def evaluate_nino34_skill_decay(all_preds, all_trues, stats, save_dir, args):
    """
    all_preds: (N, T_out) 归一化
    all_trues: (N, T_out) 归一化
    """
    print("\n--- Generating Nino3.4 Skill Decay Plots ---")
    os.makedirs(save_dir, exist_ok=True)

    preds = denorm_nino34(all_preds, stats)
    trues = denorm_nino34(all_trues, stats)

    T = preds.shape[1]
    x_axis = np.arange(1, T + 1)

    accs, rmses = [], []
    for t in range(T):
        p = preds[:, t]
        tr = trues[:, t]
        valid = ~np.isnan(p) & ~np.isnan(tr)
        if np.sum(valid) > 1 and np.std(p[valid]) > 1e-6 and np.std(tr[valid]) > 1e-6:
            accs.append(np.corrcoef(p[valid], tr[valid])[0, 1])
        else:
            accs.append(np.nan)
        rmses.append(np.sqrt(np.nanmean((p[valid] - tr[valid])**2)) if np.sum(valid) > 0 else np.nan)

    # 保存 CSV
    df = pd.DataFrame({'Lead_Time': x_axis, 'ACC': accs, 'RMSE': rmses})
    df.to_csv(os.path.join(save_dir, 'nino34_skill_decay.csv'), index=False)

    # 绘图
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xlim(x_axis[0] - 0.5, x_axis[-1] + 0.5)

    color = 'tab:red'
    ax1.set_xlabel('Forecast Lead (Months)', fontsize=11)
    ax1.set_ylabel('Nino3.4 Correlation (ACC)', color=color, fontsize=11)
    ax1.plot(x_axis, accs, color=color, marker='o', linewidth=2, label='Nino3.4 ACC')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0.0, 1.0)
    ax1.axhline(0.5, color='gray', linestyle=':', linewidth=1.5)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Nino3.4 RMSE (deg C)', color=color, fontsize=11)
    ax2.plot(x_axis, rmses, color=color, marker='s', linestyle='--', label='Nino3.4 RMSE')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Nino3.4 Index Prediction Skill Decay', fontsize=12)
    plt.savefig(os.path.join(save_dir, 'nino34_skill_decay.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved nino34_skill_decay.png")


# ─────────────────────────────────────────────────────────────
# Nino3.4 时间序列对比图
# ─────────────────────────────────────────────────────────────
def save_nino34_to_csv_and_plot(all_preds, all_trues, stats, save_dir,
                                 output_len, test_times, args):
    """
    all_preds, all_trues: (N, T_out) 归一化
    test_times: DatetimeIndex 或 list, 长度 >= N
    """
    print("\n--- Saving Nino3.4 CSV and Time Series Plot ---")
    os.makedirs(save_dir, exist_ok=True)

    preds = denorm_nino34(all_preds, stats)
    trues = denorm_nino34(all_trues, stats)
    N, T = preds.shape

    input_len = getattr(args, 'input_len', 12)

    # 保存各 lead 的 CSV
    for lead in range(T):
        rows = []
        for i in range(N):
            if test_times is not None and i < len(test_times):
                base = pd.Timestamp(test_times[i])
                target_date = base + pd.DateOffset(months=input_len + lead)
                date_str = target_date.strftime('%Y-%m')
            else:
                date_str = f"sample_{i}"
            rows.append({
                'date': date_str,
                'lead_month': lead + 1,
                'nino34_pred': preds[i, lead],
                'nino34_true': trues[i, lead],
            })
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(save_dir, f'nino34_lead{lead+1}.csv'), index=False)

    # 绘制几个关键 lead 的时间序列对比图
    key_leads = [1, 3, 6, 12, 18, 24]
    key_leads = [l for l in key_leads if l <= T]

    for lead in key_leads:
        lead_idx = lead - 1
        p = preds[:, lead_idx]
        t = trues[:, lead_idx]

        if test_times is not None and len(test_times) >= N:
            dates = [pd.Timestamp(test_times[i]) + pd.DateOffset(months=input_len + lead_idx)
                     for i in range(N)]
        else:
            dates = list(range(N))

        fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
        ax.plot(dates, t, 'k-', linewidth=1.5, label='Observed', alpha=0.8)
        ax.plot(dates, p, 'r-', linewidth=1.2, label=f'Predicted (Lead {lead}m)', alpha=0.8)
        ax.fill_between(dates, 0.5, ax.get_ylim()[1] if ax.get_ylim()[1] > 0.5 else 2.0,
                         alpha=0.05, color='red')
        ax.fill_between(dates, -0.5, ax.get_ylim()[0] if ax.get_ylim()[0] < -0.5 else -2.0,
                         alpha=0.05, color='blue')
        ax.axhline(0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axhline(-0.5, color='blue', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_ylabel('Nino3.4 Index (deg C)', fontsize=10)
        ax.set_title(f'Nino3.4 Prediction vs Observation (Lead = {lead} months)', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        if isinstance(dates[0], pd.Timestamp):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            fig.autofmt_xdate(rotation=30)
        plt.savefig(os.path.join(save_dir, f'nino34_timeseries_lead{lead}.png'),
                    bbox_inches='tight', dpi=150)
        plt.close()

    print(f"  Saved Nino3.4 CSV and time series plots for leads {key_leads}")


# ─────────────────────────────────────────────────────────────
# Nino3.4 lead correlation 多模型对比
# ─────────────────────────────────────────────────────────────
def plot_nino34_lead_correlation(all_preds_dict, all_trues, stats,
                                  save_dir, args, test_times=None):
    """
    all_preds_dict: {model_name: (N, T_out)} 归一化
    all_trues: (N, T_out) 归一化
    """
    print("\n--- Plotting Nino3.4 Lead Correlation ---")
    os.makedirs(save_dir, exist_ok=True)
    trues = denorm_nino34(all_trues, stats)
    T = trues.shape[1]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    colors = ['#C0392B', '#2980B9', '#27AE60', '#8E44AD', '#E67E22']

    for idx, (name, preds_raw) in enumerate(all_preds_dict.items()):
        preds = denorm_nino34(preds_raw, stats)
        accs = []
        for t in range(T):
            p, tr = preds[:, t], trues[:, t]
            valid = ~np.isnan(p) & ~np.isnan(tr)
            if np.sum(valid) > 1 and np.std(p[valid]) > 1e-6:
                accs.append(np.corrcoef(p[valid], tr[valid])[0, 1])
            else:
                accs.append(np.nan)
        ax.plot(range(1, T+1), accs, marker='o', linewidth=2,
                color=colors[idx % len(colors)], label=name, markersize=4)

    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1.5)
    ax.set_xlabel('Forecast Lead (Months)', fontsize=11)
    ax.set_ylabel('Correlation (ACC)', fontsize=11)
    ax.set_title('Nino3.4 Correlation vs Lead Time', fontsize=12)
    ax.set_ylim(0.0, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.savefig(os.path.join(save_dir, 'nino34_lead_correlation.png'),
                bbox_inches='tight', dpi=150)
    plt.close()
    print("  Saved nino34_lead_correlation.png")


# ─────────────────────────────────────────────────────────────
# 季节-提前期 ACC 热力图 (SPB)
# ─────────────────────────────────────────────────────────────
def plot_seasonal_lead_heatmap(all_preds, all_trues, stats, save_dir,
                                args, test_times):
    """SPB 季节性热力图: X=Lead, Y=Target Calendar Month, Color=ACC"""
    print("\n--- Generating Seasonal Lead Heatmap ---")
    if test_times is None or len(test_times) == 0:
        print("  [Warning] test_times missing, skipping.")
        return

    os.makedirs(save_dir, exist_ok=True)
    preds = denorm_nino34(all_preds, stats)
    trues = denorm_nino34(all_trues, stats)
    N, T = preds.shape
    input_len = getattr(args, 'input_len', 12)

    acc_matrix = np.full((12, T), np.nan)
    sample_cnt = np.zeros((12, T), dtype=int)

    for t in range(T):
        month_preds = {m: [] for m in range(1, 13)}
        month_trues = {m: [] for m in range(1, 13)}

        for b in range(N):
            if b >= len(test_times):
                break
            try:
                base_ts = pd.Timestamp(test_times[b])
            except Exception:
                continue
            months_ahead = input_len + t
            target_month = ((base_ts.month - 1 + months_ahead) % 12) + 1
            month_preds[target_month].append(preds[b, t])
            month_trues[target_month].append(trues[b, t])

        for m in range(1, 13):
            n = len(month_preds[m])
            sample_cnt[m-1, t] = n
            if n >= 2:
                p_arr = np.array(month_preds[m])
                t_arr = np.array(month_trues[m])
                if np.std(p_arr) > 1e-6 and np.std(t_arr) > 1e-6:
                    acc_matrix[m-1, t] = np.corrcoef(p_arr, t_arr)[0, 1]

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    lead_edges = np.arange(0.5, T + 1.5)
    month_edges = np.arange(0.5, 13.5)
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color='lightgray')

    im = ax.pcolormesh(lead_edges, month_edges, acc_matrix,
                        cmap=cmap, vmin=0.0, vmax=1.0, shading='flat')

    valid_cells = np.sum(~np.isnan(acc_matrix))
    if valid_cells > 0:
        X, Y = np.meshgrid(np.arange(1, T+1), np.arange(1, 13))
        smooth = acc_matrix.copy()
        if np.any(~np.isnan(smooth)):
            fill_val = np.nanmean(smooth)
            smooth[np.isnan(smooth)] = fill_val
            smooth = gaussian_filter(smooth, sigma=0.6)
            smooth[np.isnan(acc_matrix)] = np.nan
        try:
            cs = ax.contour(X, Y, smooth, levels=[0.4, 0.5, 0.6, 0.7, 0.8],
                            colors=['gray'], linewidths=0.8, alpha=0.7)
            ax.clabel(cs, fmt='%.1f', fontsize=7, inline=True)
        except Exception:
            pass

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.025)
    cbar.set_label('Correlation Skill (ACC)', fontsize=11)

    ax.set_yticks(np.arange(1, 13))
    ax.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=9)
    ax.set_xticks(np.arange(1, T+1, max(1, T//12)))
    ax.set_xlabel('Prediction Lead (months)', fontsize=11)
    ax.set_ylabel('Target Calendar Month', fontsize=11)
    ax.set_title('Seasonality and Lead-time Performance (SPB)', fontsize=13, pad=10)
    ax.set_xlim(0.5, T+0.5)
    ax.set_ylim(0.5, 12.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'seasonal_lead_heatmap.png'),
                bbox_inches='tight', dpi=150)
    plt.close()
    print("  Saved seasonal_lead_heatmap.png")


# ─────────────────────────────────────────────────────────────
# 极端事件案例
# ─────────────────────────────────────────────────────────────
def plot_extreme_event_case_study(all_preds, all_trues, stats, save_dir,
                                   test_times, args):
    """找到最强 El Nino 事件，展示不同 lead 的预测"""
    print("\n--- Plotting Extreme Event Case Study ---")
    os.makedirs(save_dir, exist_ok=True)

    preds = denorm_nino34(all_preds, stats)
    trues = denorm_nino34(all_trues, stats)
    N, T = trues.shape
    input_len = getattr(args, 'input_len', 12)

    # 最远 lead 时真值最大的样本
    target_lead = T - 1
    max_b_idx = np.argmax(trues[:, target_lead])

    seq_len = 24
    start_b = max(0, max_b_idx - seq_len // 2)
    end_b = min(N, max_b_idx + seq_len // 2)

    real_traj = trues[start_b:end_b, target_lead]
    pred_traj_long = preds[start_b:end_b, target_lead]
    mid_lead = T // 2
    pred_traj_mid = preds[start_b:end_b, mid_lead]

    dates = []
    for b in range(start_b, end_b):
        if test_times is not None and b < len(test_times):
            base = pd.Timestamp(test_times[b])
            dates.append(base + pd.DateOffset(months=input_len + target_lead))
        else:
            dates.append(b)

    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    ax.plot(dates, real_traj, 'k-', linewidth=2.5, label='Observation')
    ax.plot(dates, pred_traj_mid, 'g--', linewidth=1.5, marker='s', markersize=4,
            label=f'Lead {mid_lead+1} Prediction')
    ax.plot(dates, pred_traj_long, 'r-.', linewidth=2.0, marker='o', markersize=4,
            label=f'Lead {T} Prediction')

    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.7)
    ax.set_ylabel('Nino 3.4 Index (deg C)')
    ax.set_title(f'Extreme Event Tracking (Peak Sample ID: {max_b_idx})', fontsize=12)
    if isinstance(dates[0], pd.Timestamp):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate()
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(save_dir, 'extreme_event_case_study.png'), bbox_inches='tight')
    plt.close()
    print("  Saved extreme_event_case_study.png")


# ─────────────────────────────────────────────────────────────
# 功率谱分析
# ─────────────────────────────────────────────────────────────
def plot_power_spectrum(all_preds, all_trues, stats, save_dir):
    """Nino3.4 功率谱对比"""
    print("\n--- Plotting Power Spectrum ---")
    os.makedirs(save_dir, exist_ok=True)

    preds = denorm_nino34(all_preds, stats)
    trues = denorm_nino34(all_trues, stats)

    # 取第一个 lead 的时间序列做功率谱
    for lead_idx, lead_name in [(0, 'Lead1'), (5, 'Lead6'), (11, 'Lead12')]:
        if lead_idx >= preds.shape[1]:
            continue
        p_ts = preds[:, lead_idx]
        t_ts = trues[:, lead_idx]

        valid = ~np.isnan(p_ts) & ~np.isnan(t_ts)
        if np.sum(valid) < 24:
            continue

        nperseg = min(64, np.sum(valid) // 2)
        if nperseg < 8:
            continue

        f_p, pxx_p = welch(p_ts[valid], fs=12, nperseg=nperseg)
        f_t, pxx_t = welch(t_ts[valid], fs=12, nperseg=nperseg)

        fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
        ax.semilogy(1/f_t[1:], pxx_t[1:], 'k-', linewidth=1.5, label='Observed')
        ax.semilogy(1/f_p[1:], pxx_p[1:], 'r--', linewidth=1.5, label='Predicted')
        ax.set_xlabel('Period (years)', fontsize=10)
        ax.set_ylabel('Power Spectral Density', fontsize=10)
        ax.set_title(f'Power Spectrum ({lead_name})', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xlim(0, 10)
        plt.savefig(os.path.join(save_dir, f'power_spectrum_{lead_name}.png'),
                    bbox_inches='tight', dpi=150)
        plt.close()

    print("  Saved power_spectrum plots")
