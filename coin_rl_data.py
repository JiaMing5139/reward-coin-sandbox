"""
激励广告金币定价 RL —— 数据构造 (BASE, 只有自竞价广告)

本文件只负责构造数据, 不含任何模型。产出:
  1) 回放日志(训练数据): behavior policy 随机发金币 + 真实环境回填胜负/利润
  2) 推理数据: 只有特征(context), 待模型预测金币

特征分三侧, 每一维标注是否在真实环境里起因果作用:
  U 侧(用户) / G 侧(广告) / C 侧(上下文,含时间)
"""
import numpy as np

rng = np.random.default_rng(0)

# ----------------------------------------------------------------------
# 动作空间
# ----------------------------------------------------------------------
COINS = np.arange(0, 21, 1.0)          # 金币档位 0..20

# ======================================================================
# 1. 特征构造  (U 侧 + G 侧 + C 侧)
# ======================================================================
# 约定: 返回一个 dict, 每个 key 是一维特征 (shape=[n]); 另返回拼好的 state 矩阵。
# 注释里 [因果] = 真实环境会用它; [噪声] = 与利润无关, 考验模型抗干扰。

def sample_features(n):
    f = {}
    # ---------- U 侧: 用户 ----------
    f["u_ltv"]      = rng.uniform(0.0, 2.0, n)          # [因果] 用户价值, 决定广告收入
    f["u_active7"]  = rng.integers(0, 8, n).astype(float)  # [噪声] 近7天活跃天数
    f["u_sens"]     = rng.uniform(0.0, 1.0, n)          # [噪声] 体验敏感度(留作扩展)
    f["u_city_lv"]  = rng.integers(1, 6, n).astype(float)  # [噪声] 城市等级 1..5
    f["u_device"]   = rng.integers(0, 3, n).astype(float)  # [噪声] 设备档位 0/1/2
    f["u_hist_coin"]= rng.uniform(0.0, 15.0, n)         # [噪声] 历史人均金币

    # ---------- G 侧: 广告 ----------
    f["g_base"]     = rng.uniform(3.0, 7.0, n)          # [因果] 广告底价质量, 决定 eCPM
    f["g_ctr"]      = rng.uniform(0.0, 0.1, n)          # [噪声] 历史点击率
    f["g_cat"]      = rng.integers(0, 10, n).astype(float) # [噪声] 行业类目
    f["g_hist_ecpm"]= rng.uniform(5.0, 12.0, n)         # [噪声] 历史 eCPM
    f["g_material"] = rng.integers(0, 4, n).astype(float)  # [噪声] 素材类型

    # ---------- C 侧: 上下文 / 时间 ----------
    f["c_hour"]     = rng.integers(0, 24, n).astype(float) # [因果] 小时 -> 影响竞胜门槛(日内波动)
    f["c_weekend"]  = rng.integers(0, 2, n).astype(float)  # [噪声] 是否周末
    f["c_slot"]     = rng.integers(0, 3, n).astype(float)  # [噪声] 广告位 开屏/插屏/banner
    f["c_pacing"]   = rng.uniform(0.0, 1.0, n)          # [噪声] 预算消耗进度

    # 拼成 state 矩阵 (顺序固定)
    keys = list(f.keys())
    state = np.stack([f[k] for k in keys], axis=1)
    return f, state, keys

# ======================================================================
# 2. 真实环境 (线上实际发生的因果) —— 仅自竞价广告
# ======================================================================
COIN_TO_ECPM = 0.9
WIN_SHARP    = 0.6

def win_threshold(c_hour):
    """竞胜门槛随时间日内波动: 晚高峰(19-22点)竞争激烈, 门槛更高。"""
    base = 8.0
    peak = 2.0 * np.exp(-((c_hour - 20.0) ** 2) / (2 * 2.5 ** 2))  # 20点附近凸起
    return base + peak

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def true_ecpm(g_base, coins):
    return g_base + COIN_TO_ECPM * coins

def true_win_prob(g_base, coins, c_hour):
    return sigmoid(WIN_SHARP * (true_ecpm(g_base, coins) - win_threshold(c_hour)))

def true_revenue(u_ltv):
    return 6.0 + 1.5 * u_ltv

def true_expected_profit(f, coins):
    """期望利润 = 竞胜率 * (广告收入 - 金币成本)。f 是特征 dict。"""
    wp = true_win_prob(f["g_base"], coins, f["c_hour"])
    return wp * (true_revenue(f["u_ltv"]) - coins)

# ======================================================================
# 3. 回放日志 (训练数据): 随机发金币, 真实环境回填
# ======================================================================
N_TRAIN = 60000
f_tr, X_tr, KEYS = sample_features(N_TRAIN)
coins_tr  = rng.choice(COINS, N_TRAIN)                       # behavior policy
wp_tr     = true_win_prob(f_tr["g_base"], coins_tr, f_tr["c_hour"])
won_tr    = (rng.uniform(0, 1, N_TRAIN) < wp_tr).astype(float)
profit_tr = won_tr * (true_revenue(f_tr["u_ltv"]) - coins_tr)  # reward

# ======================================================================
# 4. 推理数据: 只有特征, 没有动作/反馈 (待模型预测金币)
# ======================================================================
N_INFER = 20000
f_te, X_te, _ = sample_features(N_INFER)

# ======================================================================
# 打印自检
# ======================================================================
if __name__ == "__main__":
    print("=" * 64)
    print(f"特征维度: {len(KEYS)}  ->  {KEYS}")
    print("=" * 64)
    print(f"[训练] 回放日志 X_tr={X_tr.shape}, coins/won/profit 各 {N_TRAIN}")
    print(f"   动作(金币)均值 = {coins_tr.mean():.2f}")
    print(f"   竞胜率        = {won_tr.mean():.1%}")
    print(f"   回放利润均值  = {profit_tr.mean():.3f}  (behavior 乱发, 通常亏)")
    print(f"[推理] X_te={X_te.shape} (无动作无反馈)")

    print("-" * 64)
    print("真实环境因果维度 sanity check:")
    # 时间对竞胜门槛的影响
    for h in [3, 12, 20]:
        print(f"   hour={h:>2}  竞胜门槛={win_threshold(np.array([float(h)]))[0]:.2f}")
    # 一个样本的金币->利润曲线
    one = {k: np.array([v]) for k, v in
           {"u_ltv":1.0, "g_base":5.0, "c_hour":12.0}.items()}
    curve = np.array([true_expected_profit(one, np.array([c]))[0] for c in COINS])
    best  = COINS[int(np.argmax(curve))]
    print(f"   样本(u_ltv=1,g_base=5,hour=12) 最优金币={best:.0f}, 峰值利润={curve.max():.2f}")

    print("-" * 64)
    print("训练数据前 3 行 (特征 | 金币 | 是否竞胜 | 利润):")
    for i in range(3):
        feat_str = ", ".join(f"{k}={X_tr[i,j]:.2f}" for j, k in enumerate(KEYS))
        print(f"  #{i}: {feat_str}")
        print(f"       coins={coins_tr[i]:.0f}  won={won_tr[i]:.0f}  profit={profit_tr[i]:.3f}")
