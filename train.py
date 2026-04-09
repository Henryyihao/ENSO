# train.py — Nino3.4 标量输出版 (v2-AttnRes)
"""
Nino3.4 Index Scalar Prediction Training Entry

修改说明 (相对 v1):
  1. 默认 patience 提高到 50 (原 30, shell 脚本设为 5 导致欠训练)
  2. 损失函数参数可配置 (lead_decay, spb_weight, corr_weight)
  3. 学习率 warmup 缩短为 3 个 epoch (加速早期收敛)
  4. 增加余弦退火的最小学习率比例
  5. 增大默认 d_model 为 96

Pipeline:
  1. Train: CMIP6 数据
  2. Val:   CMIP6 尾部验证
  3. Test:  OBS 盲测
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os
import time
import random

from models import get_model_dict
from dataset import ENSODataset
from Nino34Loss import Nino34Loss
from predict_utils import run_all_predict_plots
from utils import (EarlyStopping, evaluate_nino34_skill_decay,
                   save_nino34_to_csv_and_plot, plot_extreme_event_case_study,
                   plot_power_spectrum, plot_nino34_lead_correlation,
                   plot_seasonal_lead_heatmap, denorm_nino34)
from nino34_utils import find_nino34_indices

PHYS_MODELS = {'RechargeLatentNet', 'CTEFNet'}
INIT_MONTH = {'RechargeLatentNet'}

ALL_VAR_NAMES = ['sst', 'hc', 'mld', 'sss', 'slp', 'tauu', 'tauv']
OCEAN_VARS = {'sst', 'hc', 'mld', 'sss'}
ATM_VARS = {'slp', 'tauu', 'tauv'}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def fix_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_variables(var_str: str) -> list:
    var_list = [v.strip().lower() for v in var_str.split(',')]
    for v in var_list:
        if v not in ALL_VAR_NAMES:
            raise ValueError(f"Unknown variable '{v}'. Valid: {ALL_VAR_NAMES}")
    if 'sst' not in var_list:
        raise ValueError("SST must always be included in variables.")
    return var_list


def build_var_config(var_names: list) -> dict:
    return {
        'var_names': var_names,
        'n_vars': len(var_names),
        'sst_idx': var_names.index('sst'),
    }


def da_to_numpy(da) -> np.ndarray:
    arr = np.array(da.values, dtype=np.float32)
    if arr.ndim == 4:
        if arr.shape[0] == 1 and arr.shape[1] > 1:
            arr = arr[0]
        else:
            arr = arr[:, 0, :, :]
    if arr.ndim != 3:
        raise ValueError(f"da_to_numpy: cannot handle shape={arr.shape}")
    return arr


def model_forward(model, inputs, model_name, init_month=None):
    if model_name in PHYS_MODELS:
        if model_name in INIT_MONTH:
            preds, phys_loss = model(inputs, init_month)
        else:
            preds, phys_loss = model(inputs)
        return preds, phys_loss
    preds = model(inputs)
    return preds, None


def build_criterion(args):
    """构建标量 Nino3.4 损失"""
    lead_decay  = getattr(args, 'lead_decay', 0.3)
    spb_weight  = getattr(args, 'spb_weight', 1.5)
    corr_weight = getattr(args, 'corr_weight', 0.1)
    return Nino34Loss(
        lead_decay=lead_decay,
        spb_weight=spb_weight,
        corr_weight=corr_weight,
    ).to(args.device)


def compute_loss(crit, preds, y, phys_loss, init_month):
    """
    preds: (B, T_out) 标量
    y: (B, T_out) 标量
    """
    loss = crit(preds, y, init_month=init_month)
    if phys_loss is not None:
        loss = loss + phys_loss
    return loss


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_vars_from_ds(ds, var_names):
    loaded = [ds[v] for v in var_names if v in ds]
    missing = [v for v in var_names if v not in ds]
    if missing:
        print(f"  [Warning] Missing variables in nc: {missing}")
    names = [v.name for v in loaded]
    shape = loaded[0].shape if loaded else "N/A"
    print(f"  -> Loaded variables: {names}, shape: {shape}")
    return loaded


def load_cmip6(path, var_names):
    ds = xr.open_dataset(path)
    print(f"  CMIP6 dims: {dict(ds.sizes)}")

    model_dim = next((d for d in ds.dims if d in ('model', 'source_id', 'member_id')), None)

    if model_dim is not None:
        n_models = ds.sizes[model_dim]
        T_per_model = ds.sizes['time']
        lat = ds['lat'].values
        lon = ds['lon'].values
        print(f"  Model dim '{model_dim}': {n_models} models x {T_per_model} months")

        arrays = []
        for v in var_names:
            if v not in ds:
                print(f"  [Warning] Variable '{v}' not in CMIP6 dataset, skipping")
                continue
            vals = np.array(ds[v].values, dtype=np.float32)
            vals = vals.reshape(-1, vals.shape[-2], vals.shape[-1])
            arrays.append(vals)

        segments = [(i * T_per_model, (i + 1) * T_per_model) for i in range(n_models)]
    else:
        da_list = _load_vars_from_ds(ds, var_names)
        arrays = [da_to_numpy(da) for da in da_list]
        lat = ds['lat'].values
        lon = ds['lon'].values
        T_total = arrays[0].shape[0]

        n_models_attr = ds.attrs.get('n_models', None)
        if n_models_attr is not None:
            n_models = int(n_models_attr)
            if T_total % n_models == 0:
                T_per = T_total // n_models
                segments = [(i * T_per, (i + 1) * T_per) for i in range(n_models)]
            else:
                segments = [(0, T_total)]
        else:
            segments = [(0, T_total)]
            print(f"  No model dim found, treating as single continuous series ({T_total} months)")

    print(f"  CMIP6 arrays: {len(arrays)} vars, total_time={arrays[0].shape[0]}, "
          f"spatial={arrays[0].shape[1]}x{arrays[0].shape[2]}")
    return arrays, segments, lat, lon


def load_obs(path, var_names):
    ds = xr.open_dataset(path)
    da_list = _load_vars_from_ds(ds, var_names)
    arrays = [da_to_numpy(da) for da in da_list]
    lat = ds['lat'].values
    lon = ds['lon'].values
    time_da = da_list[0]
    print(f"  OBS time range: {str(time_da.time.values[0])[:7]} ~ "
          f"{str(time_da.time.values[-1])[:7]}  ({len(time_da.time)} months)")
    return arrays, lat, lon, time_da


def slice_obs(arrays, time_da, start_str, end_str):
    times = pd.to_datetime([str(t)[:10] for t in time_da.time.values])
    t_start = pd.Timestamp(start_str) if start_str else times[0]
    t_end = pd.Timestamp(end_str) if end_str else times[-1]
    mask = (times >= t_start) & (times <= t_end)
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        raise ValueError(f"Time slice [{start_str}, {end_str}] has no matches")
    s, e = idxs[0], idxs[-1] + 1
    return [arr[s:e] for arr in arrays], times[s:e]


def split_cmip6_segments(segments, val_months_per_model=24, window=36):
    min_val = max(val_months_per_model, window)
    if min_val != val_months_per_model:
        print(f"  [Info] val_months raised from {val_months_per_model} to {min_val}")
        val_months_per_model = min_val

    train_segs, val_segs = [], []
    n_tr, n_vl = 0, 0

    for (s, e) in segments:
        length = e - s
        if length <= val_months_per_model + window:
            train_segs.append((s, e))
            n_tr += length
        else:
            split_point = e - val_months_per_model
            train_segs.append((s, split_point))
            val_segs.append((split_point, e))
            n_tr += split_point - s
            n_vl += val_months_per_model

    if not val_segs:
        print("  [Warning] No model long enough for val split")
        longest = max(range(len(segments)), key=lambda i: segments[i][1] - segments[i][0])
        s, e = segments[longest]
        split_point = e - val_months_per_model
        train_segs[longest] = (s, split_point)
        val_segs.append((split_point, e))
        n_tr = sum(te - ts for ts, te in train_segs)
        n_vl = val_months_per_model

    print(f"  CMIP6 train/val split: train {n_tr} months, val {n_vl} months")
    return train_segs, val_segs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    fix_seed(args.seed)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.visual_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    var_names = parse_variables(args.variables)
    var_config = build_var_config(var_names)
    args.var_config = var_config

    print(f"\n{'='*60}")
    print(f"  Stage: {args.stage.upper()}  |  Model: {args.model_name}  |  Device: {device}")
    print(f"  Variables ({len(var_names)}): {var_names}")
    print(f"  Input: {args.input_len}m -> Output: {args.output_len} Nino3.4 scalars")
    print(f"  OBS range: {args.obs_start} ~ {args.obs_end}")
    print(f"  Loss: lead_decay={args.lead_decay} spb_weight={args.spb_weight} "
          f"corr_weight={args.corr_weight}")
    print(f"{'='*60}\n")

    exp_name = args.model_name
    best_path = os.path.join(args.save_dir, f"zeroshot_{exp_name}.pth")

    # -----------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------
    if args.stage in ('train', 'test'):
        print(f"[Data] Loading CMIP6: {args.cmip_path}")
        cmip_arr, cmip_segs, lat, lon = load_cmip6(args.cmip_path, var_names)

        window = args.input_len + args.output_len
        train_segs, val_segs = split_cmip6_segments(
            cmip_segs, val_months_per_model=args.val_months_per_model, window=window)

        print(f"\n[Data] Loading OBS: {args.obs_path}")
        obs_arr, lat_obs, lon_obs, time_da = load_obs(args.obs_path, var_names)
        test_arr, test_times = slice_obs(obs_arr, time_da, args.obs_start, args.obs_end)
        print(f"  Test set (OBS): {test_arr[0].shape[0]} months")

        args.lat_coords = lat
        args.lon_coords = lon

        print(f"\n[Dataset] Building...")
        tr_ds = ENSODataset(
            cmip_arr, args.input_len, args.output_len,
            lat=lat, lon=lon,
            is_train=True, stats=None, segments=train_segs,
            var_names=var_names, start_month=0)
        stats = tr_ds.get_stats()
        vl_ds = ENSODataset(
            cmip_arr, args.input_len, args.output_len,
            lat=lat, lon=lon,
            is_train=False, stats=stats, segments=val_segs,
            var_names=var_names, start_month=0)
        te_ds = ENSODataset(
            test_arr, args.input_len, args.output_len,
            lat=lat_obs, lon=lon_obs,
            is_train=False, stats=stats, segments=None,
            var_names=var_names, start_month=0)

        tr_loader = DataLoader(
            tr_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True)
        vl_loader = DataLoader(
            vl_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)
        te_loader = DataLoader(
            te_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)

        args.input_dim = tr_ds.n_vars
        args.img_height, args.img_width = tr_ds.spatial_shape

        print(f"  Train: {len(tr_ds)}, Val: {len(vl_ds)}, Test: {len(te_ds)}")
        print(f"  Variables: {tr_ds.n_vars}, Spatial: {args.img_height}x{args.img_width}")

    elif args.stage == 'predict':
        print(f"[Data] Loading OBS: {args.obs_path}")
        all_arr, lat, lon, time_da = load_obs(args.obs_path, var_names)
        if args.predict_input_end:
            input_end = pd.Timestamp(args.predict_input_end)
        else:
            times_pd = pd.to_datetime([str(t)[:10] for t in time_da.time.values])
            input_end = times_pd[-1]
        input_start = input_end - pd.DateOffset(months=args.input_len - 1)
        predict_arr, _ = slice_obs(
            all_arr, time_da,
            input_start.strftime('%Y-%m-%d'),
            input_end.strftime('%Y-%m-%d'))
        args.lat_coords = lat
        args.lon_coords = lon
        args.input_dim = len(predict_arr)
        args.img_height = predict_arr[0].shape[1]
        args.img_width = predict_arr[0].shape[2]

    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------
    print(f"\n[Model] Building {args.model_name}...")
    model_dict = get_model_dict()
    model = model_dict[args.model_name](args).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params/1e6:.2f} M")

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------
    if args.stage == 'train':
        crit = build_criterion(args)
        val_crit = build_criterion(args)

        opt = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        warmup_epochs = 3
        def lr_lambda(ep):
            if ep < warmup_epochs:
                return (ep + 1) / warmup_epochs
            pct = (ep - warmup_epochs) / max(1, args.epochs - warmup_epochs)
            return 0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * pct))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        stopper = EarlyStopping(patience=args.patience, verbose=True, path=best_path)

        print(f"\n{'~'*55}")
        print(f"  [TRAIN]  lr={args.learning_rate:.1e}  epochs={args.epochs}"
              f"  patience={args.patience}")
        print(f"{'~'*55}")

        global_step = 0
        for epoch in range(args.epochs):
            lr_now = opt.param_groups[0]['lr']
            model.train()
            tr_loss = 0.0

            for x, y, init_month in tqdm(tr_loader, desc=f"  E{epoch+1:03d} Train", leave=False):
                x, y, init_month = x.to(device), y.to(device), init_month.to(device)
                opt.zero_grad()
                preds, phys = model_forward(model, x, args.model_name, init_month)
                loss = compute_loss(crit, preds, y, phys, init_month)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tr_loss += loss.item()
                global_step += 1
            avg_tr = tr_loss / len(tr_loader)

            model.eval()
            vl_loss = 0.0
            with torch.no_grad():
                for x, y, init_month in vl_loader:
                    x, y, init_month = x.to(device), y.to(device), init_month.to(device)
                    preds, _ = model_forward(model, x, args.model_name, init_month)
                    loss = val_crit(preds, y, init_month=init_month)
                    vl_loss += loss.item()
            avg_vl = vl_loss / len(vl_loader)

            sched.step()
            print(f"  E{epoch+1:03d} [lr={lr_now:.2e}]  Train={avg_tr:.4f}  Val={avg_vl:.4f}")
            stopper(avg_vl, model, args, stats=stats)

            if stopper.early_stop:
                print("  Early stopping triggered.")
                break

        print(f"\n  Best model saved: {best_path}")

    # -----------------------------------------------------------------
    # Test
    # -----------------------------------------------------------------
    if args.stage in ('train', 'test'):
        print(f"\n[Test] OBS blind evaluation ({args.obs_start} ~ {args.obs_end})...")

        load_path = args.load_model_path or best_path
        ckpt = torch.load(load_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        stats = ckpt.get('stats', {})
        print(f"  Loaded model from: {load_path}")

        model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for x, y, init_month in tqdm(te_loader, desc="  Inference"):
                x, y, init_month = x.to(device), y.to(device), init_month.to(device)
                preds, _ = model_forward(model, x, args.model_name, init_month)
                all_preds.append(preds.cpu())
                all_trues.append(y.cpu())

        all_preds = torch.cat(all_preds, 0).numpy()
        all_trues = torch.cat(all_trues, 0).numpy()

        vis_dir = os.path.join(args.visual_dir, f"zeroshot_{exp_name}_{timestamp}")
        os.makedirs(vis_dir, exist_ok=True)

        evaluate_nino34_skill_decay(all_preds, all_trues, stats, vis_dir, args)
        save_nino34_to_csv_and_plot(all_preds, all_trues, stats, vis_dir,
                                     args.output_len, test_times, args)
        plot_nino34_lead_correlation(
            all_preds_dict={args.model_name: all_preds},
            all_trues=all_trues, stats=stats,
            save_dir=vis_dir, args=args, test_times=test_times)
        plot_seasonal_lead_heatmap(all_preds, all_trues, stats, vis_dir, args, test_times)
        plot_extreme_event_case_study(all_preds, all_trues, stats, vis_dir, test_times, args)
        plot_power_spectrum(all_preds, all_trues, stats, vis_dir)

        print(f"\n  Results saved to: {vis_dir}")

    # -----------------------------------------------------------------
    # Predict
    # -----------------------------------------------------------------
    if args.stage == 'predict':
        print("\n[Predict] Rolling future prediction...")

        load_path = args.load_model_path or os.path.join(
            args.save_dir, f"zeroshot_{exp_name}.pth")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model not found: {load_path}")

        ckpt = torch.load(load_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        stats = ckpt.get('stats', {})

        C = len(predict_arr)
        H, W = predict_arr[0].shape[1], predict_arr[0].shape[2]
        inp_np = np.zeros((args.input_len, C, H, W), dtype=np.float32)
        for c, (vn, arr) in enumerate(zip(var_names, predict_arr)):
            mu = float(stats.get(f'{vn}_mean', 0.0))
            sig = max(float(stats.get(f'{vn}_std', 1.0)), 1e-6)
            inp_np[:, c] = np.nan_to_num((arr - mu) / sig, nan=0.0)

        inp_tensor = torch.from_numpy(inp_np).unsqueeze(0).to(device)

        n_pred = min(args.predict_months, args.output_len)
        model.eval()
        with torch.no_grad():
            preds_out, _ = model_forward(model, inp_tensor, args.model_name)
        pred_norm = preds_out[0, :n_pred].cpu().numpy()

        nino_mean = stats.get('nino34_mean', 0.0)
        nino_std = stats.get('nino34_std', 1.0)
        pred_physical = pred_norm * nino_std + nino_mean

        base_date = (pd.Timestamp(args.predict_input_end)
                     if args.predict_input_end
                     else pd.Timestamp(str(time_da.time.values[-1])[:10]))
        future_dates = [base_date + pd.DateOffset(months=m + 1) for m in range(n_pred)]
        print(f"  Predict range: {future_dates[0].strftime('%Y-%m')} -> "
              f"{future_dates[-1].strftime('%Y-%m')}")

        obs_nino34 = None
        obs_dates_plot = None
        try:
            times_all = pd.to_datetime([str(t)[:10] for t in time_da.time.values])
            last_t = times_all[-1]
            if future_dates[0] <= last_t:
                ov_end = min(future_dates[-1], last_t)
                mask = (times_all >= future_dates[0]) & (times_all <= ov_end)
                idxs = np.where(mask)[0]
                if len(idxs) > 0:
                    sst_idx = var_names.index('sst')
                    sst_obs = all_arr[sst_idx][idxs]
                    lat_s, lat_e, lon_s, lon_e = find_nino34_indices(lat, lon)
                    lat_region = lat[lat_s:lat_e]
                    weights = np.cos(np.deg2rad(lat_region))[:, np.newaxis]
                    w_sum = float(weights.sum() * (lon_e - lon_s))
                    obs_nino34 = np.array([
                        np.nansum(sst_obs[t, lat_s:lat_e, lon_s:lon_e] * weights) / w_sum
                        for t in range(len(idxs))])
                    obs_dates_plot = [times_all[i] for i in idxs]
        except Exception as e:
            print(f"  [Warning] Failed to extract obs Nino3.4: {e}")

        dir_name = f"predict-{exp_name}-init{base_date.strftime('%Y-%m')}-lead{n_pred}m"
        vis_dir = os.path.join(args.visual_dir, dir_name)
        if os.path.exists(vis_dir):
            vis_dir = f"{vis_dir}_{timestamp}"
        os.makedirs(vis_dir, exist_ok=True)

        run_all_predict_plots(
            nino_pred=pred_physical,
            future_dates=future_dates,
            save_dir=vis_dir,
            title_suffix=f"init {base_date.strftime('%Y-%m')}, lead 1-{n_pred}m",
            obs_nino34=obs_nino34,
            obs_dates=obs_dates_plot)

        print(f"\n  Prediction results saved to: {vis_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Nino3.4 Scalar Prediction (12 months input -> 24 scalars output)')

    p.add_argument('--stage', required=True,
                   choices=['train', 'test', 'predict'])

    p.add_argument('--cmip_path', default='./processed_ssta_data/cmip6_fixed_complete.nc')
    p.add_argument('--obs_path', default='./processed_ssta_data/obs_fixed.nc')

    p.add_argument('--variables', default='sst,hc,tauu,slp,tauv,mld,sss')
    p.add_argument('--obs_start', default='1980-01-01')
    p.add_argument('--obs_end', default='2021-12-31')
    p.add_argument('--val_months_per_model', type=int, default=60)

    p.add_argument('--save_dir', default='./checkpoints/')
    p.add_argument('--visual_dir', default='./results/')
    p.add_argument('--load_model_path', default=None)

    p.add_argument('--predict_input_end', default=None)
    p.add_argument('--predict_months', type=int, default=24)

    p.add_argument('--model_name', default='RechargeLatentNet')
    p.add_argument('--input_len', type=int, default=12)
    p.add_argument('--output_len', type=int, default=24)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--learning_rate', type=float, default=5e-4)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=2025)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--d_model', type=int, default=96)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--d_var', type=int, default=48)

    # 损失函数参数
    p.add_argument('--lead_decay', type=float, default=0.3,
                   help='Lead time weight decay (0=uniform, >0=short leads weigh more)')
    p.add_argument('--spb_weight', type=float, default=1.5,
                   help='Spring Predictability Barrier loss weight multiplier')
    p.add_argument('--corr_weight', type=float, default=0.1,
                   help='Pearson correlation loss weight (0=disable)')

    p.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 128, 64])
    p.add_argument('--kernel_size', type=int, nargs='+', default=[5, 5])
    p.add_argument('--n_layers', type=int, default=4)

    args = p.parse_args()
    main(args)
