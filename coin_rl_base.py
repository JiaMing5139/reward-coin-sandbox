"""
激励广告金币定价 RL —— BASE 版 (只有自竞价广告, 不含联盟/穿山甲)

流程:
  1) 定义真实环境 (线上实际发生的因果)
  2) 用 behavior policy 随机发金币, 构造回放日志 (训练数据)
  3) 训练金币定价模型 (Q 表: 拟合 context+coins -> 真实回放利润)
  4) 构造推理数据, 用模型推策略, 与 oracle 对比
纯 numpy。
"""
import numpy as np

rng = np.random.default_rng(0)

# ----------------------------------------------------------------------
# 0. 动作空间
# ----------------------------------------------------------------------
COINS = np.arange(0, 21, 1.0)          # 金币档位 0..20

# ----------------------------------------------------------------------
# 1. 真实环境 (线上实际发生的事) —— 仅自竞价广告
# ----------------------------------------------------------------------
# state: u = 用户价值/体验敏感度 (影响广告收入), g = 广告底价质量
# action: coins
# 金币 -> 抬高 eCPM -> 提高竞胜率(曝光), 但金币本身是成本
COIN_TO_ECPM = 0.9                     # 金币撬动 eCPM 的系数
WIN_THRESH   = 8.0                     # 竞胜门槛(竞争环境)
WIN_SHARP    = 0.6

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def true_ecpm(g, coins):
    return g + COIN_TO_ECPM * coins

def true_win_prob(g, coins):
    return sigmoid(WIN_SHARP * (true_ecpm(g, coins) - WIN_THRESH))

def true_revenue(u):
    """胜出曝光后的广告收入(与用户价值相关), 与金币无关。"""
    return 6.0 + 1.5 * u

def true_expected_profit(u, g, coins):
    """期望利润 = 竞胜率 * (广告收入 - 金币成本)。"""
    return true_win_prob(g, coins) * (true_revenue(u) - coins)

# ----------------------------------------------------------------------
# 2. 构造回放日志 (训练数据)
#    behavior policy: 随机发金币 (线上探索), 真实环境回填胜负与利润
# ----------------------------------------------------------------------
def sample_context(n):
    u = rng.uniform(0.0, 2.0, n)
    g = rng.uniform(3.0, 7.0, n)
    return u, g

N_TRAIN = 60000
u_tr, g_tr = sample_context(N_TRAIN)
coins_tr   = rng.choice(COINS, N_TRAIN)                 # 随机探索动作
wp_tr      = true_win_prob(g_tr, coins_tr)
won_tr     = (rng.uniform(0, 1, N_TRAIN) < wp_tr).astype(float)
profit_tr  = won_tr * (true_revenue(u_tr) - coins_tr)   # 真实回放利润 (reward)

print("=" * 60)
print(f"回放日志(训练): N={N_TRAIN}")
print(f"  平均金币={coins_tr.mean():.1f}  竞胜率={won_tr.mean():.1%}  "
      f"平均利润={profit_tr.mean():.3f}")

# ----------------------------------------------------------------------
# 3. 训练模型: 分桶 Q 表  Q(u_bin, g_bin, coins) = 平均回放利润
#    (用桶平均, 能逼出金币的内点最优, 不受线性假设限制)
# ----------------------------------------------------------------------
U_BINS = np.linspace(0, 2, 5)      # 4 桶
G_BINS = np.linspace(3, 7, 5)      # 4 桶
def binidx(x, bins): return np.clip(np.digitize(x, bins) - 1, 0, len(bins) - 2)

nU, nG, nC = len(U_BINS) - 1, len(G_BINS) - 1, len(COINS)
Qsum = np.zeros((nU, nG, nC)); Qcnt = np.zeros((nU, nG, nC))
ui, gi = binidx(u_tr, U_BINS), binidx(g_tr, G_BINS)
ci = coins_tr.astype(int)
np.add.at(Qsum, (ui, gi, ci), profit_tr)
np.add.at(Qcnt, (ui, gi, ci), 1.0)
Qtable = Qsum / np.maximum(Qcnt, 1.0)      # 空桶记 0

# ----------------------------------------------------------------------
# 4. 构造推理数据, 推策略并与 oracle 对比
# ----------------------------------------------------------------------
N_INFER = 20000
u_te, g_te = sample_context(N_INFER)

# 模型策略: 在 Q 表里挑利润最高的金币档
ui_te, gi_te = binidx(u_te, U_BINS), binidx(g_te, G_BINS)
model_coin = COINS[np.argmax(Qtable[ui_te, gi_te], axis=1)]

# oracle 策略: 直接在真实环境里挑最优金币
oracle_q = np.stack([true_expected_profit(u_te, g_te, c) for c in COINS], axis=1)
oracle_coin = COINS[np.argmax(oracle_q, axis=1)]

# 两个策略在真实环境的利润
model_profit  = true_expected_profit(u_te, g_te, model_coin)
oracle_profit = true_expected_profit(u_te, g_te, oracle_coin)

print("=" * 60)
print(f"推理评估: N={N_INFER}")
print(f"  {'策略':<10}{'真实利润':>12}{'平均金币':>12}")
print("  " + "-" * 34)
print(f"  {'oracle':<10}{oracle_profit.mean():>12.3f}{oracle_coin.mean():>12.2f}")
print(f"  {'model':<10}{model_profit.mean():>12.3f}{model_coin.mean():>12.2f}")
print(f"  达到 oracle 的 {model_profit.mean()/oracle_profit.mean():.1%}")

# ----------------------------------------------------------------------
# 5. 看几个分桶上模型选的金币 vs oracle, 验证内点最优学对了
# ----------------------------------------------------------------------
print("=" * 60)
print("分桶最优金币对比 (model vs oracle):")
print(f"  {'u_bin':>6}{'g_bin':>6}{'model':>8}{'oracle':>8}")
for ub in range(nU):
    for gb in range(nG):
        uc = (U_BINS[ub] + U_BINS[ub+1]) / 2
        gc = (G_BINS[gb] + G_BINS[gb+1]) / 2
        m = COINS[np.argmax(Qtable[ub, gb])]
        o = COINS[np.argmax([true_expected_profit(uc, gc, c) for c in COINS])]
        flag = "" if m == o else "  <-- 差异"
        print(f"  {ub:>6}{gb:>6}{m:>8.0f}{o:>8.0f}{flag}")
