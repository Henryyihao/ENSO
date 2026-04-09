"""
Nino34Loss — SPB-Aware Nino3.4 标量损失 (v2)
==============================================
修复说明:
  1. lead 权重反转: 短 lead 权重 >= 长 lead (原版相反, 导致模型坍缩到均值)
  2. 新增 Pearson 相关系数损失: 直接优化 ACC 指标
  3. SPB 权重可调且默认降低至 1.5 (原 2.0 在早期训练不稳定)

用法:
    criterion = Nino34Loss(spb_weight=1.5, corr_weight=0.1)
    loss = criterion(pred, target, init_month=init_month)
"""

import torch
import torch.nn as nn


class Nino34Loss(nn.Module):
    """
    SPB-Aware Nino3.4 标量损失 + 相关系数损失

    Parameters
    ----------
    lead_decay  : float  短 lead 加权衰减系数, 0=均匀, >0=短 lead 更重要
    spb_weight  : float  春季目标月的额外损失倍率
    corr_weight : float  Pearson 相关损失权重 (0=不用, >0=开启)
    """

    def __init__(self, lead_decay: float = 0.3, spb_weight: float = 1.5,
                 corr_weight: float = 0.1):
        super().__init__()
        self.lead_decay  = lead_decay
        self.spb_weight  = spb_weight
        self.corr_weight = corr_weight

    def _lead_weight_tensor(self, T_out, device):
        """短 lead 权重 >= 长 lead 权重 (修复原版反向错误)"""
        leads = torch.arange(1, T_out + 1, dtype=torch.float32, device=device)
        if self.lead_decay <= 0:
            # 均匀权重
            return torch.ones(T_out, device=device)
        # 线性衰减: lead 1 权重最高, lead T_out 权重最低
        w = 1.0 - self.lead_decay * (leads - 1) / (T_out - 1)
        w = w / w.mean()  # 归一化使均值 = 1
        return w

    def _spb_weight_tensor(self, init_month, T_out, device):
        lead_idx = torch.arange(T_out, device=device)
        target_months = (init_month.unsqueeze(1) + lead_idx.unsqueeze(0)) % 12
        # 春季: Mar(2), Apr(3), May(4)
        is_spring = (target_months >= 2) & (target_months <= 4)
        w_spb = torch.where(
            is_spring,
            torch.full_like(is_spring, float(self.spb_weight), dtype=torch.float32),
            torch.ones_like(is_spring, dtype=torch.float32),
        )
        return w_spb

    def _pearson_corr_loss(self, pred, target):
        """
        Pearson 相关系数损失 (每个 lead 独立计算, 取均值)
        直接优化 ACC 指标, 解决 MSE 导致模型预测均值的问题
        """
        B, T = pred.shape
        if B < 4:
            return torch.tensor(0.0, device=pred.device)

        # 每个 lead 独立计算相关系数
        p_mean = pred.mean(dim=0, keepdim=True)   # (1, T)
        t_mean = target.mean(dim=0, keepdim=True)  # (1, T)
        p_c = pred - p_mean
        t_c = target - t_mean

        cov = (p_c * t_c).mean(dim=0)             # (T,)
        p_std = p_c.pow(2).mean(dim=0).sqrt().clamp(min=1e-6)
        t_std = t_c.pow(2).mean(dim=0).sqrt().clamp(min=1e-6)

        corr = cov / (p_std * t_std)               # (T,)
        # 短 lead 相关性更重要
        lead_w = self._lead_weight_tensor(T, pred.device)
        return 1.0 - (corr * lead_w).sum() / lead_w.sum()

    def forward(self, pred, target, init_month=None, step=0):
        """
        pred       : (B, T_out)  预测的 Nino3.4 标量
        target     : (B, T_out)  真实的 Nino3.4 标量
        init_month : (B,) LongTensor  初始月份 (0=Jan)
        """
        B, T_out = pred.shape
        device = pred.device

        # lead 权重 (短 lead 更重要)
        w_lead = self._lead_weight_tensor(T_out, device)

        # SPB 权重
        if init_month is not None:
            w_spb = self._spb_weight_tensor(init_month, T_out, device)
        else:
            w_spb = torch.ones(B, T_out, device=device)

        w = w_lead.unsqueeze(0) * w_spb

        # MSE 损失
        sq_err = (pred - target).pow(2)
        mse_loss = (sq_err * w).mean()

        # 相关系数损失
        if self.corr_weight > 0 and B >= 4:
            corr_loss = self._pearson_corr_loss(pred, target)
            total_loss = mse_loss + self.corr_weight * corr_loss
        else:
            total_loss = mse_loss

        return total_loss
