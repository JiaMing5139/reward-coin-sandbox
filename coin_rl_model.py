"""
最简模型 —— 能不能在 Sandbox 里学会"金币怎么出最好"

模型: 一个线性回归当 Q 函数 (纯 numpy, 最朴素)
  输入特征 = [标准化后的15维state, coins, coins^2]
  拟合目标 = 真实回放利润 profit
  推理     = 对每条样本扫 21 个金币档, 取预测利润最大的那个
加 coins^2 是为了让线性模型能表达"先升后降"的内点峰。

对照基线:
  oracle  = Sandbox 解析最优 (上界)
  random  = behavior policy (乱发金币, 训练数据的水平)
"""
import numpy as np
from coin_rl_sandbox import Sandbox, sample_features, COINS

rng = np.random.default_rng(0)
box = Sandbox()

# ----------------------------------------------------------------------
# 1. 造数据: 回放日志(训练) + 推理集
# ----------------------------------------------------------------------
N_TRAIN, N_INFER = 80000, 20000
f_tr, X_tr, KEYS = sample_features(N_TRAIN, rng)
coins_tr = rng.choice(COINS, N_TRAIN)                 # behavior: 随机发金币
out_tr   = box.step(f_tr, coins_tr, rng)
y_tr     = out_tr["profit"]                            # 真实回放利润 = reward

f_te, X_te, _ = sample_features(N_INFER, rng)

# ----------------------------------------------------------------------
# 2. 特征工程: 标准化 state, 拼上 coins 与 coins^2
# ----------------------------------------------------------------------
mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-9
def make_phi(X, coins):
    Xs = (X - mu) / sd
    c  = (coins / 20.0)[:, None]
    return np.concatenate([Xs, c, c**2, np.ones((len(coins), 1))], axis=1)

Phi_tr = make_phi(X_tr, coins_tr)

# ----------------------------------------------------------------------
# 3. 训练: ridge 闭式解  w = (ΦᵀΦ + λI)⁻¹ Φᵀy
# ----------------------------------------------------------------------
lam = 1.0
A = Phi_tr.T @ Phi_tr + lam * np.eye(Phi_tr.shape[1])
w = np.linalg.solve(A, Phi_tr.T @ y_tr)
train_mse = np.mean((Phi_tr @ w - y_tr) ** 2)
print(f"训练完成: 特征维={Phi_tr.shape[1]}, 训练MSE={train_mse:.4f}")

# ----------------------------------------------------------------------
# 4. 推理: 对每条样本扫 21 档金币, 取模型预测利润最大者
# ----------------------------------------------------------------------
def model_policy(X):
    n = len(X)
    best_c = np.zeros(n); best_q = np.full(n, -1e9)
    for c in COINS:
        q = make_phi(X, np.full(n, c)) @ w
        upd = q > best_q
        best_q = np.where(upd, q, best_q)
        best_c = np.where(upd, c, best_c)
    return best_c

def oracle_policy(f, n):
    best_c = np.zeros(n); best_q = np.full(n, -1e9)
    for c in COINS:
        q = box.expected_profit(f, np.full(n, c))
        upd = q > best_q
        best_q = np.where(upd, q, best_q)
        best_c = np.where(upd, c, best_c)
    return best_c

coin_model  = model_policy(X_te)
coin_oracle = oracle_policy(f_te, N_INFER)
coin_random = rng.choice(COINS, N_INFER)

# 真实环境里评估这三种策略的期望利润 (用解析 expected_profit, 无采样噪声)
p_model  = box.expected_profit(f_te, coin_model).mean()
p_oracle = box.expected_profit(f_te, coin_oracle).mean()
p_random = box.expected_profit(f_te, coin_random).mean()

print("=" * 56)
print(f"{'策略':<10}{'真实期望利润':>14}{'平均金币':>12}")
print("-" * 56)
print(f"{'random':<10}{p_random:>14.4f}{coin_random.mean():>12.2f}")
print(f"{'model':<10}{p_model:>14.4f}{coin_model.mean():>12.2f}")
print(f"{'oracle':<10}{p_oracle:>14.4f}{coin_oracle.mean():>12.2f}")
print("-" * 56)
gap = (p_model - p_random) / (p_oracle - p_random + 1e-9)
print(f"模型把 random->oracle 的差距填补了 {gap:.1%}")

# ----------------------------------------------------------------------
# 5. 模型选的金币 vs oracle, 看是否随上下文变化
# ----------------------------------------------------------------------
print("=" * 56)
print("抽 6 条样本看模型 vs oracle 的金币决策:")
print(f"{'u_ltv':>6}{'g_bid':>7}{'g_ctr':>7}{'hour':>6}{'model':>7}{'oracle':>8}")
for i in range(6):
    print(f"{f_te['u_ltv'][i]:>6.2f}{f_te['g_bid'][i]:>7.1f}{f_te['g_ctr'][i]:>7.2f}"
          f"{f_te['c_hour'][i]:>6.0f}{coin_model[i]:>7.0f}{coin_oracle[i]:>8.0f}")

mae = np.abs(coin_model - coin_oracle).mean()
print(f"\n模型与 oracle 金币决策的平均绝对差 = {mae:.2f} 档")
