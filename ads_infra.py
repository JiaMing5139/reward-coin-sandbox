"""
ads_infra.py —— 广告投放链路 (基础设施)

真实投放链路:
  库存(inventory) → 召回(recall K个候选) → [金币] → 预估(predict pCTR, 带误差)
                  → eCPM = bid × predict_pctr → rank 排序 → top-1 竞胜
                  → 二价计费(成交价 = 次高位 eCPM)

与 world 的关键区别:
  - 预估 pCTR 带【误差】(系统偏差 + 噪声), 用于竞价排序。
  - world 的真实 pCTR 才决定用户真点不点。
  两者不一致 → 可能把"真实高价值"广告排错位 → 拿不到曝光 → 收入损失。
  predict_noise=0 时退化为完美预估 (eCPM 排序用真值)。

广告是【持久实体】: 库存里每条广告有 ad_id, 整个 rollout 复用。
"""
import numpy as np

from world import sigmoid


# ======================================================================
# 库存: 一批有 id 的持久广告
# ======================================================================
def build_inventory(n_ads, rng, cfg):
    """生成固定广告库存。每条广告是持久实体, 有 ad_id 与真实属性。

    返回 dict of arrays, 第 i 个元素是 ad_id=i 的广告:
      ad_id / g_bid / g_ctr(真实点击质量) / ad_type(0自竞价 1联盟) / g_ext_ecpm(B类外生)
    """
    inv = {}
    inv["ad_id"]      = np.arange(n_ads)
    inv["g_bid"]      = rng.uniform(8.0, 20.0, n_ads)
    inv["g_ctr"]      = rng.uniform(0.05, 0.30, n_ads)
    inv["g_cat"]      = rng.integers(0, 10, n_ads).astype(float)
    inv["g_material"] = rng.integers(0, 4, n_ads).astype(float)
    if cfg["union_enabled"]:
        inv["ad_type"]    = (rng.uniform(0, 1, n_ads) < cfg["union_ratio"]).astype(float)
        inv["g_ext_ecpm"] = rng.uniform(cfg["union_ecpm_lo"], cfg["union_ecpm_hi"], n_ads)
    else:
        inv["ad_type"]    = np.zeros(n_ads)
        inv["g_ext_ecpm"] = np.zeros(n_ads)
    # 每条广告一个固定的"预估偏差"(模拟模型对不同广告系统性高估/低估)
    inv["pred_bias"]  = rng.normal(0, cfg["predict_bias_std"], n_ads)
    return inv


# ======================================================================
# 召回: 为每个用户请求从库存取 K 个候选
# ======================================================================
def recall(inv, n_req, K, rng):
    """对 n_req 个请求, 各随机召回 K 个候选广告。
    返回候选 ad_id 矩阵 (n_req, K)。(最小实现: 随机召回; 可扩展为按兴趣)
    """
    n_ads = len(inv["ad_id"])
    return rng.integers(0, n_ads, size=(n_req, K))


def gather(inv, cand_ids, key):
    """按候选 id 矩阵 (n_req, K) 取出某属性, 返回 (n_req, K)。"""
    return inv[key][cand_ids]


# ======================================================================
# 预估层 (带误差) —— 核心
# ======================================================================
class Predictor:
    """平台对 pCTR 的预估。= 真实 logit + 系统偏差(每广告固定) + 随机噪声。
    它【不知道】world 的真实参数, 用自己的(可能有偏的)参数估。
    这里为简化, 预估用与 world 相同的结构但叠加偏差/噪声来模拟"不准"。
    """

    # 预估层自己的参数 (理论上应与 world 不同; 简化为同结构 + 误差)
    COIN_TO_CTR = 0.06
    LTV_TO_CTR  = 0.40
    GQ_TO_CTR   = 6.0
    CTR_BIAS    = -2.2

    def __init__(self, cfg):
        self.noise_std = cfg["predict_noise"]

    def predict_pctr(self, feats, coins, pred_bias, rng):
        """预估点击率 (n,)。feats 各字段均为 (n,)。
        pred_bias: 每条候选广告的系统性偏差(来自库存)。
        """
        logit = (self.CTR_BIAS + self.LTV_TO_CTR * feats["u_ltv"]
                 + self.GQ_TO_CTR * feats["g_ctr"] + self.COIN_TO_CTR * coins)
        logit = logit + pred_bias                       # 系统偏差(高估/低估某些广告)
        if self.noise_std > 0:
            logit = logit + rng.normal(0, self.noise_std, len(coins))
        return sigmoid(logit)


# ======================================================================
# 竞价: 对召回候选算 eCPM, rank, top-1 竞胜, 二价计费
# ======================================================================
def auction(inv, cand_ids, user, coins, predictor, world, rng):
    """对每个请求的 K 个候选完成竞价, 返回胜出广告 + 成交价。

    参数:
      cand_ids: (n_req, K) 召回候选
      user:     dict, 各字段 (n_req,) —— 用户特征(请求级)
      coins:    (n_req,) 本请求拟发金币 (作用于所有候选的预估)
      world:    World 实例, 用于计算真实 pCTR (计费用)
    返回: winner dict (各字段 n_req,), 含真实特征 + won + clearing_price_true
    """
    n_req, K = cand_ids.shape
    # 取候选真实特征 (n_req, K)
    bid   = gather(inv, cand_ids, "g_bid")
    gctr  = gather(inv, cand_ids, "g_ctr")
    atype = gather(inv, cand_ids, "ad_type")
    ext   = gather(inv, cand_ids, "g_ext_ecpm")
    pbias = gather(inv, cand_ids, "pred_bias")

    coins_b = np.repeat(coins[:, None], K, axis=1)      # (n_req, K) 同请求同金币
    # 预估 pCTR (逐候选), 需展平调用 predictor
    feats_flat = {"u_ltv": np.repeat(user["u_ltv"][:, None], K, 1).ravel(),
                  "g_ctr": gctr.ravel()}
    pctr_pred = predictor.predict_pctr(feats_flat, coins_b.ravel(), pbias.ravel(), rng).reshape(n_req, K)

    # eCPM 预估: A 类 = bid × 预估pctr; B 类 = 外生 eCPM (与金币/预估无关)
    ecpm_pred = np.where(atype == 0, bid * pctr_pred, ext)   # (n_req, K)

    # rank: 选 eCPM_pred 最高者胜出; 成交价用【真实 eCPM】的次高位 (二价不受预估噪声污染)
    order = np.argsort(-ecpm_pred, axis=1)                   # 按预估排序
    win_idx = order[:, 0]
    rows = np.arange(n_req)

    # 真实 eCPM (用于计费, 不受预估误差污染)
    pctr_true = world.pctr_true(feats_flat, coins_b.ravel()).reshape(n_req, K)
    ecpm_true = np.where(atype == 0, bid * pctr_true, ext)
    second_ecpm_true = ecpm_true[rows, order[:, 1]] if K >= 2 else np.zeros(n_req)

    def pick(arr):
        return arr[rows, win_idx]

    winner = {
        "ad_id":      cand_ids[rows, win_idx],
        "g_bid":      pick(bid),
        "g_ctr":      pick(gctr),
        "ad_type":    pick(atype),
        "g_ext_ecpm": pick(ext),
        "won":        np.ones(n_req),          # 召回非空必有胜者(简化: 总有曝光机会)
        "clearing_price": second_ecpm_true,    # 二价成交价(用真实eCPM, 不受预估噪声污染)
        "win_ecpm":   pick(ecpm_pred),         # 胜出者的预估 eCPM (排序用)
    }
    return winner
