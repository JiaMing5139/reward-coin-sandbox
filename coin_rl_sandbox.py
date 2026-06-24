"""
Sandbox —— 上帝视角的真实环境模拟器 (激励广告金币定价)

设计要点:
  * 策略/模型看不到 Sandbox 内部参数, 只能通过 step() 拿反馈 (won/clicked/profit)。
  * 金币(coins)统一从【精排 CTR 模型】进入因果链:
        coins ─► pCTR(精排) ─► eCPM = bid×pCTR ─► 竞胜(vs win_threshold) ─► 曝光
        coins ─────────────────────────────────► 直接成本(发放金币)
    一条抬收入(经曝光+点击), 一条加成本, 二者拉扯 => 金币存在内点最优。
  * win_threshold 只是竞胜环节的一个量(竞争水位), 不再是金币的唯一入口。
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# 动作空间
COINS = np.arange(0, 21, 1.0)


class Sandbox:
    """真实世界。所有 true_* 参数都是上帝私有, 模型不可见。"""

    # ----- 上帝参数 (策略不可见) -----
    COIN_TO_CTR = 0.06     # 金币激励对 pCTR(logit) 的提升系数
    LTV_TO_CTR  = 0.40     # 高价值用户更愿意点
    GQ_TO_CTR   = 6.0      # 广告自身 ctr 质量的权重
    CTR_BIAS    = -2.2     # ctr logit 偏置
    WIN_SHARP   = 1.0      # 竞胜 sigmoid 陡峭度
    THRESH_BASE = 6.0      # 竞争水位基线 (够得着, 竞胜率有弹性)
    THRESH_PEAK = 2.0      # 晚高峰水位凸起幅度
    COIN_COST   = 0.5      # 金币面值 -> 真实成本折算 (1金币=0.5元成本)

    # ============ 精排 CTR 模型 (金币在此进入) ============
    def pctr(self, f, coins):
        """预估点击率。金币作为激励特征参与 CTR 预估。"""
        logit = (self.CTR_BIAS
                 + self.LTV_TO_CTR * f["u_ltv"]
                 + self.GQ_TO_CTR  * f["g_ctr"]      # 广告内在点击质量 (因果)
                 + self.COIN_TO_CTR * coins)
        return sigmoid(logit)

    # ============ eCPM = 出价 × pCTR ============
    def ecpm(self, f, coins):
        """eCPM 由出价与预估点击率相乘得到; 金币经 pCTR 间接抬高 eCPM。"""
        return f["g_bid"] * self.pctr(f, coins)

    # ============ 竞胜环节 ============
    def win_threshold(self, f):
        """竞争水位: 随时间日内波动 (20点晚高峰最激烈)。win_threshold 是其中之一。"""
        peak = self.THRESH_PEAK * np.exp(-((f["c_hour"] - 20.0) ** 2) / (2 * 2.5 ** 2))
        return self.THRESH_BASE + peak

    def win_prob(self, f, coins):
        return sigmoid(self.WIN_SHARP * (self.ecpm(f, coins) - self.win_threshold(f)))

    # ============ 期望利润 (解析, 给 oracle / 评估用) ============
    def expected_profit(self, f, coins):
        """E[利润] = 竞胜率 × ( 点击带来的收入 - 金币成本 )
                   = win_prob × ( pCTR × bid  -  COIN_COST × coins )
        曝光后: 期望点击收入 = pCTR×bid; 成本 = 发放金币×折算。"""
        wp = self.win_prob(f, coins)
        return wp * (self.pctr(f, coins) * f["g_bid"] - self.COIN_COST * coins)

    # ============ 真实采样一次 (给回放日志用) ============
    def step(self, f, coins, rng):
        """走一遍真实拍卖, 返回字典 (won/clicked/revenue/cost/profit)。"""
        wp = self.win_prob(f, coins)
        won = (rng.uniform(0, 1, len(coins)) < wp).astype(float)
        pc = self.pctr(f, coins)
        clicked = won * (rng.uniform(0, 1, len(coins)) < pc).astype(float)
        revenue = clicked * f["g_bid"]      # 广告主按点击付费
        cost = won * self.COIN_COST * coins  # 曝光即发放金币 (折算成本)
        profit = revenue - cost
        return {"won": won, "clicked": clicked,
                "revenue": revenue, "cost": cost, "profit": profit}


# ======================================================================
# 特征构造 (U 侧 + G 侧 + C 侧)。 [因果]=Sandbox 会用; [噪声]=干扰维
# ======================================================================
def sample_features(n, rng):
    f = {}
    # ---- U 侧 ----
    f["u_ltv"]       = rng.uniform(0.0, 2.0, n)             # [因果] -> ctr & revenue
    f["u_active7"]   = rng.integers(0, 8, n).astype(float)  # [噪声]
    f["u_sens"]      = rng.uniform(0.0, 1.0, n)             # [噪声]
    f["u_city_lv"]   = rng.integers(1, 6, n).astype(float)  # [噪声]
    f["u_device"]    = rng.integers(0, 3, n).astype(float)  # [噪声]
    f["u_hist_coin"] = rng.uniform(0.0, 15.0, n)           # [噪声]
    # ---- G 侧 ----
    f["g_bid"]       = rng.uniform(8.0, 20.0, n)           # [因果] 广告主出价 -> eCPM & revenue
    f["g_ctr"]       = rng.uniform(0.05, 0.30, n)          # [因果] 广告内在点击质量 -> pCTR
    f["g_cat"]       = rng.integers(0, 10, n).astype(float) # [噪声]
    f["g_hist_ecpm"] = rng.uniform(5.0, 12.0, n)           # [噪声]
    f["g_material"]  = rng.integers(0, 4, n).astype(float)  # [噪声]
    # ---- C 侧 ----
    f["c_hour"]      = rng.integers(0, 24, n).astype(float) # [因果] -> win_threshold
    f["c_weekend"]   = rng.integers(0, 2, n).astype(float)  # [噪声]
    f["c_slot"]      = rng.integers(0, 3, n).astype(float)  # [噪声]
    f["c_pacing"]    = rng.uniform(0.0, 1.0, n)            # [噪声]

    keys = list(f.keys())
    state = np.stack([f[k] for k in keys], axis=1)
    return f, state, keys


# ======================================================================
# 自检
# ======================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    box = Sandbox()

    # ---- 回放日志(训练) ----
    N_TRAIN = 60000
    f_tr, X_tr, KEYS = sample_features(N_TRAIN, rng)
    coins_tr = rng.choice(COINS, N_TRAIN)
    out_tr = box.step(f_tr, coins_tr, rng)

    # ---- 推理数据 ----
    N_INFER = 20000
    f_te, X_te, _ = sample_features(N_INFER, rng)

    print("=" * 64)
    print(f"特征维度 {len(KEYS)}: {KEYS}")
    print("=" * 64)
    print(f"[训练] X_tr={X_tr.shape}")
    print(f"   金币均值={coins_tr.mean():.2f}  竞胜率={out_tr['won'].mean():.1%}  "
          f"点击率={out_tr['clicked'].mean():.1%}")
    print(f"   收入均值={out_tr['revenue'].mean():.3f}  成本均值={out_tr['cost'].mean():.3f}  "
          f"利润均值={out_tr['profit'].mean():.3f}")
    print(f"[推理] X_te={X_te.shape} (无动作无反馈)")

    print("-" * 64)
    print("因果链 sanity check (固定 u_ltv=1, g_bid=14, g_ctr=0.18):")
    one = {"u_ltv": np.array([1.0]), "g_bid": np.array([14.0]),
           "g_ctr": np.array([0.18]), "c_hour": np.array([12.0])}
    print(f"  {'coins':>6}{'pCTR':>8}{'eCPM':>8}{'win_p':>8}{'E[利润]':>10}")
    for c in [0, 4, 8, 12, 16, 20]:
        cc = np.array([float(c)])
        print(f"  {c:>6}{box.pctr(one,cc)[0]:>8.3f}{box.ecpm(one,cc)[0]:>8.2f}"
              f"{box.win_prob(one,cc)[0]:>8.3f}{box.expected_profit(one,cc)[0]:>10.3f}")
    curve = np.array([box.expected_profit(one, np.array([c]))[0] for c in COINS])
    print(f"  => 最优金币={COINS[int(np.argmax(curve))]:.0f}, 峰值利润={curve.max():.3f}")

    print("-" * 64)
    print("时间对竞胜水位的影响:")
    for h in [3, 12, 20]:
        print(f"  hour={h:>2}  win_threshold={box.win_threshold({'c_hour':np.array([float(h)])})[0]:.2f}")
