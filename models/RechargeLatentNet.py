"""
RechargeLatentNet v3 — ELSO (Efficient Lead-aware Spatial Oscillator)
======================================================================
设计目标:
  1. 解决v2显存爆炸: ConvGRU自回归→并行Transformer, 节省3-5x显存
  2. 提升>20月预测: HC专用Recharge Oscillator + 长期记忆路径
  3. 改善春季预测障碍: 月份条件化注意力 + Lead解码器显式月份建模
  4. 提高训练速度: 无串行依赖, ~3x加速

物理依据（基于相关矩阵分析）:
  SST-HC: 0.77    -> HC是最强单一预测因子（recharged oscillator主体）
  SST-SLP: -0.65  -> Walker环流反相关（短期预测信号）
  HC-MLD: 0.76    -> 海洋混合层与热含量耦合（中期信号）
  SSS/TAUV        -> 弱预测因子（噪声多，长窗口平滑）

接口: 与v2完全兼容
  forward(x, init_month=None) -> (preds, phys_loss)
  x:        (B, T_in, C, H, W)
  preds:    (B, T_out)   Nino3.4标量
  phys_loss: scalar
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================================
# 基础工具
# ==========================================================================

def cyclic_month_enc(month_idx: torch.Tensor, d: int) -> torch.Tensor:
    """月份循环编码  (B,) LongTensor [0-11] -> (B, d) float"""
    t = month_idx.float() * (2 * math.pi / 12)
    k = torch.arange(1, d // 2 + 1, device=month_idx.device, dtype=torch.float32)
    angles = t.unsqueeze(1) * k.unsqueeze(0)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)[:, :d]


# ==========================================================================
# 物理时间尺度配置（基于可预测性图分析）
# ==========================================================================

VAR_TIMESCALES = {
    'sst':  (3,  12),   # 快变（6月内有效预测），长窗口捕捉背景态
    'hc':   (6,  24),   # recharged oscillator核心，需要最长历史
    'mld':  (3,   9),   # 与HC耦合，中等时间尺度
    'sss':  (6,  18),   # 弱预测因子，长窗口平滑噪声
    'slp':  (2,   6),   # 大气快变，短时间尺度
    'tauu': (2,   6),   # 纬向风应力，ENSO同步快变
    'tauv': (2,   6),   # 经向风应力，最弱预测因子
}

# 关键海洋区域（比例坐标: lat南->北, lon 0->360）
OCEAN_REGIONS = {
    'nino34': (0.42, 0.58, 0.53, 0.75),   # 5S-5N, 190-240E
    'wpac':   (0.42, 0.58, 0.33, 0.47),   # 5S-5N, 120-170E（暖池）
    'epac':   (0.42, 0.58, 0.75, 0.92),   # 5S-5N, 270-330E（东太）
    'npac':   (0.60, 0.72, 0.42, 0.75),   # 20-40N, PDO区域
}


# ==========================================================================
# 模块1: 关键海洋区域池化
# ==========================================================================

class RegionPool(nn.Module):
    """
    将空间特征图 (B, d, H, W) 池化到预定义海洋区域 (B, n_regions, d)
    相比全局池化: 保留更多空间信息
    相比全卷积展开: 节省 (H*W)/(n_regions) 倍显存
    """

    def __init__(self, H: int, W: int, region_names: list):
        super().__init__()
        self.n_regions = len(region_names)
        self._slices = []
        for rn in region_names:
            lf0, lf1, wf0, wf1 = OCEAN_REGIONS[rn]
            h0 = max(0, int(lf0 * H))
            h1 = min(H, max(h0 + 1, int(lf1 * H)))
            w0 = max(0, int(wf0 * W))
            w1 = min(W, max(w0 + 1, int(wf1 * W)))
            self._slices.append((h0, h1, w0, w1))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """(B, d, H, W) -> (B, n_regions, d)"""
        return torch.stack(
            [feat[:, :, h0:h1, w0:w1].mean(dim=[-2, -1])
             for h0, h1, w0, w1 in self._slices],
            dim=1
        )


# ==========================================================================
# 模块2: 单变量多时间尺度编码器
# ==========================================================================

class VarEncoder(nn.Module):
    """
    单变量双时间尺度空间编码器
    短分支->快变信号（Walker环流、ENSO快变）
    长分支->背景态（ENSO成熟/转型信号）
    可学习门控融合，适应不同ENSO阶段
    """

    def __init__(self, T_avail: int, d: int,
                 short_T: int = 3, long_T: int = 12,
                 dropout: float = 0.1):
        super().__init__()
        self.short_T = min(short_T, T_avail)
        self.long_T  = min(long_T,  T_avail)
        d2 = d // 2

        def _branch(k_t, ch_out):
            return nn.Sequential(
                nn.Conv3d(1, ch_out, (k_t, 3, 3), padding=(0, 1, 1)),
                nn.GELU(),
                nn.GroupNorm(min(4, ch_out), ch_out),
                nn.Dropout3d(dropout),
                nn.Conv3d(ch_out, ch_out, (1, 3, 3), padding=(0, 1, 1)),
                nn.GELU(),
                nn.GroupNorm(min(4, ch_out), ch_out),
            )

        self.short_branch = _branch(self.short_T, d2)
        self.long_branch  = _branch(self.long_T,  d2)

        # 融合两分支
        self.fusion = nn.Sequential(
            nn.Conv2d(d2 * 2, d, 1),
            nn.GroupNorm(min(4, d), d),
            nn.GELU(),
            nn.Dropout2d(dropout)
        )

    def forward(self, x_var: torch.Tensor) -> torch.Tensor:
        """x_var: (B, 1, T, H, W) -> (B, d, H, W)"""
        h_s = self.short_branch(x_var[:, :, -self.short_T:]).squeeze(2)
        h_l = self.long_branch(x_var[:, :, -self.long_T:]).squeeze(2)
        return self.fusion(torch.cat([h_s, h_l], dim=1))


# ==========================================================================
# 模块3: HC Recharge Oscillator（核心物理模块）
# ==========================================================================

class HCRechargeOscillator(nn.Module):
    """
    热含量（HC）充放电振荡器专用编码器
    ----------------------------------------
    物理机制 (Jin 1997 Recharged Oscillator):
      充电阶段: 西风异常 -> 暖水聚集西太 -> 温跃层加深
      放电阶段: 暖水东传 -> 温跃层上升 -> El Nino爆发
      -> HC是>12月预测的主要信号源

    三时间尺度:
      短(6月):  当前ENSO相位
      中(12月): 年际变化（最近ENSO循环）
      长(24月): 多年周期（预测>20月的关键！）

    充放电状态:
      西太(充电区)和Nino3.4(当前相位)区域HC均值差
      -> 充放电势 -> 长期预测调制信号
    """

    def __init__(self, T: int, d: int, H: int, W: int, dropout: float = 0.1):
        super().__init__()
        self.T_s = min(6,  T)
        self.T_m = min(12, T)
        self.T_l = min(24, T)
        d3 = max(d // 3, 8)

        def _branch(k_t):
            return nn.Sequential(
                nn.Conv3d(1, d3, (k_t, 3, 3), padding=(0, 1, 1)),
                nn.GELU(),
                nn.GroupNorm(min(4, d3), d3),
                nn.Dropout3d(dropout),
                nn.Conv3d(d3, d3, (1, 3, 3), padding=(0, 1, 1)),
                nn.GELU(),
                nn.GroupNorm(min(4, d3), d3),
            )

        self.short_enc = _branch(self.T_s)
        self.mid_enc   = _branch(self.T_m)
        self.long_enc  = _branch(self.T_l)

        # 三分支融合 -> d维空间特征
        self.fusion = nn.Sequential(
            nn.Conv2d(d3 * 3, d, 1),
            nn.GroupNorm(min(4, d), d),
            nn.GELU(),
            nn.Dropout2d(dropout)
        )

        # 关键海洋区域切片（预计算）
        H_ = max(H, 1); W_ = max(W, 1)
        # 西太暖池（充电区）
        self._wh = slice(max(0, int(0.42*H_)), min(H_, max(int(0.42*H_)+1, int(0.58*H_))))
        self._ww = slice(max(0, int(0.33*W_)), min(W_, max(int(0.33*W_)+1, int(0.47*W_))))
        # Nino3.4（当前相位）
        self._nh = slice(max(0, int(0.42*H_)), min(H_, max(int(0.42*H_)+1, int(0.58*H_))))
        self._nw = slice(max(0, int(0.53*W_)), min(W_, max(int(0.53*W_)+1, int(0.75*W_))))

        # 充放电状态 -> d维向量
        self.charge_proj = nn.Sequential(
            nn.Linear(2, 64), nn.GELU(), nn.Linear(64, d)
        )

    def forward(self, x_hc: torch.Tensor):
        """
        x_hc: (B, 1, T, H, W)
        returns:
          feat:         (B, d, H, W)  空间特征
          charge_state: (B, d)        充放电状态（长期记忆）
        """
        h_s = self.short_enc(x_hc[:, :, -self.T_s:]).squeeze(2)
        h_m = self.mid_enc(x_hc[:, :, -self.T_m:]).squeeze(2)
        h_l = self.long_enc(x_hc[:, :, -self.T_l:]).squeeze(2)
        feat = self.fusion(torch.cat([h_s, h_m, h_l], dim=1))

        # 充放电状态（从标准化HC场提取区域均值）
        wpac_hc = x_hc[:, 0, -1, self._wh, self._ww].mean(dim=[-2, -1])  # (B,)
        n34_hc  = x_hc[:, 0, -1, self._nh, self._nw].mean(dim=[-2, -1])  # (B,)
        charge_state = self.charge_proj(
            torch.stack([wpac_hc, n34_hc], dim=1)  # (B, 2)
        )  # (B, d)

        return feat, charge_state


# ==========================================================================
# 模块4: 月份条件化跨变量注意力
# ==========================================================================

class MonthCondCrossVarAttn(nn.Module):
    """
    月份条件化的跨变量跨区域注意力
    --------------------------------
    物理动机:
      ENSO锁相: SST-HC耦合在秋季(9-11月)最强
      Walker环流: SLP-SST反相在冬季(12-2月)最强
      春季预测障碍(3-5月): 各变量预测贡献均弱
    -> 月份编码调制注意力，显式建模季节性差异
    """

    def __init__(self, d: int, d_model: int,
                 n_heads: int = 4, cyclic_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.cyclic_dim = cyclic_dim
        # 保证d可被n_heads整除
        n_heads = min(n_heads, d)
        while d % n_heads != 0 and n_heads > 1:
            n_heads -= 1

        self.month_bias = nn.Sequential(
            nn.Linear(cyclic_dim, d), nn.GELU()
        )
        self.attn  = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d * 4, d)
        )
        self.norm2    = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, d_model)

    def forward(self, tokens: torch.Tensor, month_idx: torch.Tensor) -> torch.Tensor:
        """tokens: (B, N, d), month_idx: (B,) -> (B, d_model)"""
        m_enc = cyclic_month_enc(month_idx, self.cyclic_dim)
        tokens = tokens + self.month_bias(m_enc).unsqueeze(1)

        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)
        tokens = self.norm2(tokens + self.ff(tokens))

        return self.out_proj(tokens.mean(dim=1))


# ==========================================================================
# 模块5: 并行 Lead Transformer Decoder
# ==========================================================================

class LeadTransformerDecoder(nn.Module):
    """
    并行 Lead-Time Transformer Decoder
    ------------------------------------
    关键: 所有T_out个lead同时计算，无自回归循环！

    显存对比 (B=32, d=96, H=24, W=72, T_out=24):
      v2 ConvGRU隐状态: ~24 * 32 * 96 * 24 * 72 * 4 bytes ~ 1.2 GB
      v3 Transformer激活: ~32 * 24 * 96 * 4 bytes ~ 7 MB
      节省约 180x 激活显存

    Memory设计:
      [全局上下文, HC充放电状态] 两个token
      -> 短期路径利用全局上下文
      -> 长期路径利用HC充放电状态

    双路径输出:
      lead 1-12:  主要走短期路径（SST驱动）
      lead 13-24: 主要走长期路径（HC recharge驱动）
    """

    def __init__(self, d_model: int, T_out: int = 24,
                 n_heads: int = 4, n_layers: int = 3,
                 cyclic_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.T_out     = T_out
        self.d_model   = d_model
        self.cyclic_dim = cyclic_dim

        # Lead嵌入（可学习）
        self.lead_emb  = nn.Embedding(T_out + 12, d_model)
        nn.init.normal_(self.lead_emb.weight, 0, 0.02)

        # 目标月份编码投影
        self.month_proj = nn.Linear(cyclic_dim, d_model)

        # HC充放电状态 -> 额外memory token
        self.charge_proj = nn.Linear(d_model, d_model)

        # Pre-LN Transformer Decoder（训练更稳定）
        decoder_layer = nn.TransformerDecoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_model * 4,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,    # Pre-LayerNorm
            activation      = 'gelu',
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # 双路径输出头（短期 + 长期）
        def _head():
            return nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1)
            )

        self.short_head = _head()
        self.long_head  = _head()

        # 可学习混合系数（logit空间）
        # 初始化: lead 1-T/2 偏短路径，lead T/2+1-T 偏长路径
        half = T_out // 2
        mix_init = torch.cat([
            torch.linspace(-3.0, 0.0, half),            # lead 1-12: sigmoid->0.05~0.50
            torch.linspace(0.0,  3.0, T_out - half),    # lead 13-24: sigmoid->0.50~0.95
        ])
        self.mix_logit = nn.Parameter(mix_init)

    def forward(self, context: torch.Tensor,
                charge_state: torch.Tensor,
                init_month: torch.Tensor) -> torch.Tensor:
        """
        context:      (B, d_model) 全局上下文
        charge_state: (B, d_model) HC充放电状态
        init_month:   (B,) LongTensor [0-11]
        -> preds: (B, T_out)
        """
        B      = context.shape[0]
        device = context.device
        leads  = torch.arange(self.T_out, device=device)

        # Lead查询: lead_emb + target_month_emb
        lead_e = self.lead_emb(leads).unsqueeze(0).expand(B, -1, -1)  # (B, T, d)

        tgt_months = (init_month.unsqueeze(1) + leads.unsqueeze(0)) % 12
        month_enc  = cyclic_month_enc(tgt_months.reshape(-1), self.cyclic_dim)
        month_e    = self.month_proj(month_enc).view(B, self.T_out, self.d_model)

        queries = lead_e + month_e  # (B, T_out, d_model)

        # Memory: [全局上下文, HC充放电状态]
        ctx_tok    = context.unsqueeze(1)                         # (B, 1, d)
        charge_tok = self.charge_proj(charge_state).unsqueeze(1)  # (B, 1, d)
        memory     = torch.cat([ctx_tok, charge_tok], dim=1)      # (B, 2, d)

        # 并行解码（所有lead同时！无自回归！）
        decoded = self.transformer(queries, memory)  # (B, T_out, d_model)

        # 双路径融合
        out_s = self.short_head(decoded).squeeze(-1)  # (B, T_out)
        out_l = self.long_head(decoded).squeeze(-1)   # (B, T_out)

        alpha = torch.sigmoid(self.mix_logit).unsqueeze(0)  # (1, T_out)
        return (1.0 - alpha) * out_s + alpha * out_l


# ==========================================================================
# 主编码器
# ==========================================================================

class MultiVarEncoder(nn.Module):
    """
    多变量联合编码器
    使用 ModuleDict 管理各变量编码器（避免 None 占位问题）
    """

    def __init__(self, T: int, d_var: int, d_model: int,
                 H: int, W: int, var_names: list,
                 n_regions: int = 4, cyclic_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.var_names = var_names
        self.has_hc    = 'hc' in var_names
        self.d_var     = d_var

        # 各变量编码器（ModuleDict，HC单独处理）
        enc_dict = {}
        for vn in var_names:
            if vn == 'hc':
                continue
            st, lt = VAR_TIMESCALES.get(vn, (3, 12))
            enc_dict[vn] = VarEncoder(T, d_var, st, lt, dropout)
        self.var_encoders = nn.ModuleDict(enc_dict)

        # HC专用模块
        if self.has_hc:
            self.hc_osc            = HCRechargeOscillator(T, d_var, H, W, dropout)
            self.charge_to_dmodel  = nn.Sequential(
                nn.Linear(d_var, d_model), nn.GELU(), nn.Linear(d_model, d_model)
            )
        else:
            self.null_charge = nn.Parameter(torch.zeros(d_model))

        # 区域池化
        region_names = ['nino34', 'wpac', 'epac', 'npac'][:n_regions]
        self.region_pool = RegionPool(H, W, region_names)

        # Token归一化
        self.token_norm = nn.LayerNorm(d_var)

        # 月份条件化跨变量注意力
        n_heads_attn = max(1, min(4, d_var // 8))
        self.cross_var_attn = MonthCondCrossVarAttn(
            d_var, d_model, n_heads=n_heads_attn,
            cyclic_dim=cyclic_dim, dropout=dropout
        )

    def forward(self, x: torch.Tensor, month_idx: torch.Tensor):
        """
        x: (B, C, T, H, W) 变量在第1维
        -> context: (B, d_model), charge_state: (B, d_model)
        """
        var_region_feats = []
        charge_state_raw = None

        for i, vn in enumerate(self.var_names):
            x_var = x[:, i:i+1]  # (B, 1, T, H, W)
            if vn == 'hc' and self.has_hc:
                feat_map, charge_state_raw = self.hc_osc(x_var)
            else:
                feat_map = self.var_encoders[vn](x_var)

            r = self.region_pool(feat_map)         # (B, n_regions, d_var)
            var_region_feats.append(self.token_norm(r))

        tokens  = torch.cat(var_region_feats, dim=1)  # (B, n_vars*n_regions, d_var)
        context = self.cross_var_attn(tokens, month_idx)

        if charge_state_raw is not None:
            charge_state = self.charge_to_dmodel(charge_state_raw)
        else:
            charge_state = self.null_charge.unsqueeze(0).expand(x.shape[0], -1)

        return context, charge_state


# ==========================================================================
# 主模型: RechargeLatentNet (v3 — ELSO)
# ==========================================================================

class RechargeLatentNet(nn.Module):
    """
    RechargeLatentNet v3 — ELSO

    推荐超参数配置:
      参数           v2推荐   v3推荐   说明
      ----------------------------------------------------------------
      d_model         96      96      保持一致
      d_var           N/A     48      新增: 变量编码维度（=d//2）
      n_regions       N/A      4      新增: 关键海洋区域数
      n_dec_layers    N/A      3      新增: Transformer解码层数
      n_heads         N/A      4      新增: 注意力头数
      input_len        36     36      可增至48捕捉更长ENSO历史
      output_len       24     24      保持
      batch_size       32     64      并行解码支持更大batch
      learning_rate   5e-4   3e-4    Transformer更稳定
      spb_weight       1.5    2.0    加强春季训练压力
      corr_weight      0.1    0.2    更强ACC优化
      lead_decay       0.3    0.15   减弱长lead惩罚(鼓励20+月预测)

    Shell脚本示例:
      python train.py --model_name RechargeLatentNet \\
        --d_model 96 --d_var 48 --n_regions 4 \\
        --n_dec_layers 3 --n_heads 4 \\
        --input_len 36 --output_len 24 \\
        --batch_size 64 --learning_rate 3e-4 \\
        --spb_weight 2.0 --corr_weight 0.2 --lead_decay 0.15
    """

    def __init__(self, configs):
        super().__init__()
        self.input_len  = configs.input_len
        self.output_len = configs.output_len
        self.d_model    = configs.d_model
        self.input_dim  = configs.input_dim

        T     = self.input_len
        T_out = self.output_len
        d     = self.d_model
        H     = getattr(configs, 'img_height',   24)
        W     = getattr(configs, 'img_width',    72)

        dropout    = getattr(configs, 'dropout',      0.1)
        cyclic_dim = getattr(configs, 'cyclic_dim',   16)
        n_regions  = getattr(configs, 'n_regions',     4)
        n_heads    = getattr(configs, 'n_heads',        4)
        n_layers   = getattr(configs, 'n_dec_layers',   3)
        d_var      = getattr(configs, 'd_var',  max(32, d // 2))

        # 变量名解析（兼容v2）
        var_names = getattr(configs, 'var_names', None)
        if var_names is None:
            vc = getattr(configs, 'var_config', None)
            var_names = (vc['var_names'] if (vc and 'var_names' in vc)
                         else ['sst', 'hc', 'tauu'])
        self.var_names = var_names

        # 编码器
        self.encoder = MultiVarEncoder(
            T=T, d_var=d_var, d_model=d,
            H=H, W=W, var_names=var_names,
            n_regions=n_regions, cyclic_dim=cyclic_dim, dropout=dropout,
        )

        # 并行Lead解码器（核心改进）
        self.decoder = LeadTransformerDecoder(
            d_model=d, T_out=T_out,
            n_heads=n_heads, n_layers=n_layers,
            cyclic_dim=cyclic_dim, dropout=dropout,
        )

        # 物理正则化系数
        self.phys_weight = nn.Parameter(torch.tensor(0.05))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, init_month: torch.Tensor = None):
        """
        x:          (B, T_in, C, H, W)
        init_month: (B,) LongTensor [0-11]，None时默认0
        -> (preds, phys_loss)
           preds: (B, T_out)  Nino3.4标量
        """
        B, T, C, H, W = x.shape
        device = x.device

        if init_month is None:
            init_month = torch.zeros(B, dtype=torch.long, device=device)

        # (B, T, C, H, W) -> (B, C, T, H, W)
        x_enc = x.permute(0, 2, 1, 3, 4).contiguous()

        # 编码
        context, charge_state = self.encoder(x_enc, init_month)

        # 并行解码（无自回归循环）
        preds = self.decoder(context, charge_state, init_month)

        # 物理正则化
        phys_loss = self._physics_loss(preds)

        return preds, phys_loss

    def _physics_loss(self, preds: torch.Tensor) -> torch.Tensor:
        """
        物理约束:
          1. 时序平滑: 相邻lead预测不剧烈跳变
          2. 幅度约束: 超过3sigma的预测给予软惩罚
        """
        smooth = ((preds[:, 1:] - preds[:, :-1]).pow(2).mean()
                  if preds.shape[1] > 1
                  else torch.tensor(0.0, device=preds.device))
        amplitude = F.relu(preds.abs() - 3.0).pow(2).mean()
        w = F.softplus(self.phys_weight)
        return w * (smooth * 0.1 + amplitude * 0.05)


# ==========================================================================
# 快速测试
# ==========================================================================

if __name__ == '__main__':
    torch.manual_seed(42)

    class Cfg:
        input_len    = 36
        output_len   = 24
        d_model      = 96
        d_var        = 48
        input_dim    = 7
        img_height   = 24
        img_width    = 72
        dropout      = 0.1
        cyclic_dim   = 16
        n_regions    = 4
        n_heads      = 4
        n_dec_layers = 3
        var_names    = ['sst', 'hc', 'mld', 'sss', 'slp', 'tauu', 'tauv']

    cfg   = Cfg()
    model = RechargeLatentNet(cfg)
    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {n_p/1e6:.3f} M")
    print("子模块参数分布:")
    for n, sm in model.named_children():
        p = sum(pp.numel() for pp in sm.parameters())
        print(f"  {n}: {p/1e3:.1f} K")

    # 前向传播
    B = 4
    x  = torch.randn(B, 36, 7, 24, 72)
    im = torch.tensor([0, 3, 6, 9])
    model.eval()
    with torch.no_grad():
        preds, phys = model(x, im)
    print(f"\n输入:    {x.shape}")
    print(f"输出:    {preds.shape}")
    print(f"物理损失: {phys.item():.6f}")

    # 梯度传播
    model.train()
    preds2, phys2 = model(x, im)
    (preds2.mean() + phys2).backward()
    print("梯度传播: OK")

    # 不同batch大小
    for bs in [16, 32, 64]:
        with torch.no_grad():
            p, _ = model(torch.randn(bs, 36, 7, 24, 72),
                         torch.randint(0, 12, (bs,)))
        print(f"  batch={bs}: {p.shape} OK")

    # 不同init_month验证SPB建模
    print("\n不同初始月份预测（验证季节性差异）:")
    months = ['Jan', 'Apr', 'Jul', 'Oct']
    for mi, mn in enumerate(months):
        with torch.no_grad():
            p, _ = model(x[:1], torch.tensor([mi * 3]))
        print(f"  init={mn}: pred_range=[{p.min():.3f}, {p.max():.3f}]"
              f"  lead1={p[0,0]:.3f}  lead24={p[0,-1]:.3f}")