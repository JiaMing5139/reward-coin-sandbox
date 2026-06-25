"""
Main —— 主流程: 金币定价 RL (序列决策 / 预算 / off-policy 版)

相比单步版, 这里体现三件真实化改造:
  (1) 序列决策: 一个 episode = 一个用户日(多步), 训练目标用【MC 折扣累计回报】,
      而非单步利润 → 模型学的是长期 Q, 会为"留住用户"而多发一点金币。
  (2) 预算:     每个用户日有金币预算(写在 obs 里的 budget_left/steps_left),
      模型据此做预算分配。
  (5) off-policy: 训练日志由【有偏的行为策略】产生(不是纯随机), 某些(状态,金币)
      组合从没出现 → 模型在分布外只能外推。用 random/biased 两种日志对比暴露危害。

完整流程:
  [步骤1] 采集回放日志: 用行为策略跑 episode, 记录 (obs, coins, 折扣回报)
  [步骤2] 特征工程: 标准化 obs + 拼 coins/coins^2; 目标=折扣回报
  [步骤3] 训练 Q 模型
  [步骤4] rollout 评估: 把各策略放进真环境跑 episode, 比累计利润
  [步骤5] 对照与诊断: random / behavior / myopic(短视真值) / model, 分广告类型

运行: python3 main.py
"""
import numpy as np

from sandbox import Sandbox, COINS, OBS_KEYS, CONFIG
from model import MLP

# ----------------------------------------------------------------------
# 全局配置
# ----------------------------------------------------------------------
N_TRAIN_USERS = 80000     # 采日志的用户日数 (每个产生多步样本)
N_EVAL_USERS  = 20000     # 评估用户日数
SEED = 0
GAMMA = CONFIG["gamma"]
BUDGET_IDX = OBS_KEYS.index("budget_left")


# ======================================================================
# 行为策略 (#5 off-policy 的核心): 决定训练日志怎么采
# ======================================================================
def behavior_policy(obs, rng, biased=True):
    """采集训练日志时用的策略。
    biased=False: 纯随机发金币 (理想探索, 覆盖所有动作)。
    biased=True : 有偏的"当前线上策略" —— 倾向按预算比例发,
                  且几乎不给低预算/末步的请求发高金币 →
                  制造【覆盖空洞】, 让模型在这些(状态,金币)上没有数据。
    """
    n = len(obs)
    if not biased:
        return rng.choice(COINS, n)
    # 有偏: 以 budget_left 为中心发金币, 加噪声, clip 到档位
    budget = obs[:, BUDGET_IDX]
    center = np.clip(budget * 0.25, 0, COINS.max())     # 预算多才敢发多
    noisy = center + rng.normal(0, 2.0, n)
    # 对齐到最近档位
    idx = np.clip(np.round(noisy / 2.0), 0, len(COINS) - 1).astype(int)
    return COINS[idx]


# ======================================================================
# 通用: 用某策略跑 episode, 收集轨迹 (用于采日志 + rollout 评估)
# ======================================================================
def run_episodes(box, n_users, policy_fn, rng, collect=False):
    """把 policy_fn 放进真环境跑 n_users 个用户日。

    policy_fn(obs) -> coins
    返回: 平均每用户日累计利润;
    collect=True 时还返回 transitions: (obs, coins, reward, next_obs, done)
    —— 即 TD/Q-learning 所需的 (s, a, r, s', done) 四元组(只保留真实发生的步)。
    """
    obs = box.reset(n_users)
    step_obs, step_coins, step_reward, step_next, step_done, step_active = [], [], [], [], [], []
    while box.alive.any():
        active = box.alive.copy()
        coins = policy_fn(obs)
        obs_now = obs
        obs, reward, done, info = box.step(coins)
        step_obs.append(obs_now)
        step_coins.append(np.asarray(coins, float))
        step_reward.append(reward)
        step_next.append(obs)              # s' (下一步观测)
        step_done.append(done)             # 该步后 episode 是否结束
        step_active.append(active)

    T = len(step_reward)
    rewards = np.stack(step_reward)
    actives = np.stack(step_active)
    total_profit = rewards.sum(0).mean()

    if not collect:
        return total_profit, (rewards, actives)

    # 摊平成 transition 样本 (s, a, r, s', done): 只保留真实发生(该步存活)的
    Xs, Cs, Rs, Ns, Ds = [], [], [], [], []
    for t in range(T):
        m = actives[t].astype(bool)
        Xs.append(step_obs[t][m])
        Cs.append(step_coins[t][m])
        Rs.append(rewards[t][m])
        Ns.append(step_next[t][m])
        Ds.append(step_done[t][m].astype(float))
    trans = {
        "obs":  np.concatenate(Xs),
        "coin": np.concatenate(Cs),
        "rew":  np.concatenate(Rs),
        "next": np.concatenate(Ns),
        "done": np.concatenate(Ds),
    }
    return total_profit, trans


def main():
    rng = np.random.default_rng(SEED)

    # ==================================================================
    # [步骤1] 采集回放日志 (off-policy: 用有偏行为策略跑 episode)
    # ==================================================================
    print("[步骤1] 采集回放日志 ...")
    box = Sandbox(seed=SEED)
    biased = CONFIG["behavior_biased"]
    print(f"  行为策略: {'有偏(线上策略,制造覆盖空洞)' if biased else '无偏(随机探索)'}")
    pol = lambda o: behavior_policy(o, rng, biased=biased)
    train_profit, tr = run_episodes(box, N_TRAIN_USERS, pol, rng, collect=True)
    obs_tr, coin_tr, rew_tr = tr["obs"], tr["coin"], tr["rew"]
    next_tr, done_tr = tr["next"], tr["done"]
    print(f"  采到 transition 数={len(obs_tr)} (来自 {N_TRAIN_USERS} 个用户日)")
    print(f"  行为策略每用户日累计利润={train_profit:.3f}")
    print(f"  即时利润 r 均值={rew_tr.mean():.3f} 范围=[{rew_tr.min():.2f},{rew_tr.max():.2f}]")

    # ==================================================================
    # [步骤2] 特征工程: 标准化 obs + 拼 coins/coins^2
    # ==================================================================
    print("[步骤2] 特征工程 ...")
    mu, sd = obs_tr.mean(0), obs_tr.std(0) + 1e-9

    def make_phi(X, coins):
        Xs = (X - mu) / sd
        c = (np.asarray(coins, float) / COINS.max())[:, None]
        return np.concatenate([Xs, c, c ** 2], axis=1)

    Phi_sa = make_phi(obs_tr, coin_tr)        # Q(s,a) 的输入 (固定不变)
    d_in = Phi_sa.shape[1]
    print(f"  特征维度={d_in} ({len(OBS_KEYS)} obs + coins + coins^2)")

    # 预算掩码: next 状态下每个金币档是否可发 (超预算的不能选)
    next_budget = next_tr[:, BUDGET_IDX]       # (M,)
    coin_ok = COINS[None, :] <= next_budget[:, None]   # (M, |COINS|)

    # ==================================================================
    # [步骤3] 训练 Q 模型 —— Fitted Q-Iteration (TD/Q-learning)
    #   目标 y = r + γ · max_a' Q(s', a')   (s' 终止则无未来项)
    #   未来项用【上一轮的模型】估(自举), 而非整条轨迹采样 → 方差远小于 MC。
    #   每一轮: 用当前 Q 算 TD 目标 → 重新拟合 → 迭代收敛。
    # ==================================================================
    print("[步骤3] 训练 Q 模型 (Fitted Q-Iteration) ...")
    model = MLP(d_in=d_in, seed=0)
    n_iter = 8
    for it in range(n_iter):
        if it == 0:
            # 第0轮: 没有 Q 可自举, 目标就是即时利润 (等价单步)
            target = rew_tr.copy()
        else:
            # 估 next 状态各档 Q, 受预算掩码, 取 max
            qn = np.full((len(obs_tr), len(COINS)), -1e9)
            for j, c in enumerate(COINS):
                qn[:, j] = model.predict(make_phi(next_tr, np.full(len(obs_tr), c)))
            qn = np.where(coin_ok, qn, -1e9)
            max_qn = qn.max(1)
            target = rew_tr + GAMMA * (1.0 - done_tr) * max_qn

        # 拟合 Q(s,a) -> target (标准化目标稳梯度)
        t_mu, t_sd = target.mean(), target.std() + 1e-9
        model.train(Phi_sa, (target - t_mu) / t_sd,
                    n_epoch=25, batch_sz=512, lr=3e-3, rng=rng, verbose=False)
        # 把标准化反算回去: 重新缩放最后一层输出, 让 predict 直接给真值尺度
        model.W2 *= t_sd; model.b2 = model.b2 * t_sd + t_mu
        pred = model.predict(Phi_sa)
        td_err = np.mean((pred - target) ** 2)
        print(f"  iter {it+1}/{n_iter}  TD目标均值={target.mean():.3f}  拟合MSE={td_err:.4f}")

    # 策略: 扫金币档取预测 Q 最大者 (受预算上限约束)
    def model_policy(obs):
        n = len(obs)
        best_c = np.zeros(n); best_q = np.full(n, -1e9)
        budget = obs[:, BUDGET_IDX]
        for c in COINS:
            q = model.predict(make_phi(obs, np.full(n, c)))
            q = np.where(c <= budget, q, -1e9)
            upd = q > best_q
            best_q = np.where(upd, q, best_q)
            best_c = np.where(upd, c, best_c)
        return best_c

    # 短视真值策略 (myopic): 每步用上帝单步利润挑最优, 不顾未来
    def myopic_policy(obs):
        n = len(obs)
        f = {k: obs[:, i] for i, k in enumerate(OBS_KEYS)}
        best_c = np.zeros(n); best_q = np.full(n, -1e9)
        budget = obs[:, BUDGET_IDX]
        for c in COINS:
            q = box.immediate_profit(f, np.full(n, c))
            q = np.where(c <= budget, q, -1e9)
            upd = q > best_q
            best_q = np.where(upd, q, best_q); best_c = np.where(upd, c, best_c)
        return best_c

    # ==================================================================
    # [步骤4-5] rollout 评估 + 诊断 (各策略放进真环境跑 episode)
    # ==================================================================
    print("[步骤4] rollout 评估 ...")
    policies = {
        "random":   lambda o: np.random.default_rng(1).choice(COINS, len(o)),
        "behavior": lambda o: behavior_policy(o, np.random.default_rng(2), biased=CONFIG["behavior_biased"]),
        "myopic":   myopic_policy,
        "model":    model_policy,
    }
    print("=" * 60)
    print(f"{'策略':<10}{'每用户日累计利润':>18}")
    print("-" * 60)
    results = {}
    for name, fn in policies.items():
        ev = Sandbox(seed=12345)            # 同一评估种子, 公平对比
        prof, _ = run_episodes(ev, N_EVAL_USERS, fn, np.random.default_rng(7))
        results[name] = prof
        print(f"{name:<10}{prof:>18.4f}")
    print("-" * 60)
    print(f"model vs myopic 提升 = {(results['model']-results['myopic']):+.4f} "
          f"({'model更优(学到长期价值)' if results['model']>results['myopic'] else 'model未超短视'})")
    print(f"model vs behavior(线上) 提升 = {(results['model']-results['behavior']):+.4f}")


if __name__ == "__main__":
    main()
