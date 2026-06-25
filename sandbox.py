"""
sandbox.py —— 有状态环境编排 (Gym 范式)

职责: 串起【投放链路(ads_infra)】与【真实世界(world)】, 持有 World State,
对外提供 reset()/step(coins)。

  step 内部一步:
    召回(ads_infra.recall) → 竞价/预估/rank/竞胜(ads_infra.auction)
    → 真实曝光点击留存(world.realize) → 演化 World State → 返回客观 info

关键: step 只吐【客观事实 info】, 不吐 reward。reward 由 reward.py 外部设计。

三层抽象:
  1. 上帝参数   —— 在 world.py / ads_infra.py 里, 对模型不可见。
  2. World State —— 本文件持有: budget_left/steps_left/alive + 当前用户。随 step 演化。
  3. Observation —— 模型看到的投影向量 (用户 + 本次"市场"摘要 + 预算 + 步数)。

因果链 (金币如何影响利润):
                      ┌─► 用户观看意愿 ┐
  金币 ──────────────┤                ├─► 曝光(看且竞胜) ─► 收入
                      └─► 预估pCTR ─► eCPM ─► rank竞胜 ┘
  金币 ─────────────────────────────────────► 曝光后发金币 = 成本
  金币 ─────────────► 用户留存(下一步还来) ──► 未来更多曝光(长期价值)
"""
import numpy as np

from world import World
from ads_infra import build_inventory, recall, auction, gather, Predictor

# 动作空间: 10 档金币 (0, 2, 4, ..., 18)
COINS = np.arange(0, 20, 2.0)

CONFIG = {
    # ---- 联盟广告 ----
    "union_enabled": True,
    "union_ratio":   0.30,
    "union_ecpm_lo": 5.0,
    "union_ecpm_hi": 11.0,
    # ---- 序列决策 ----
    "horizon":    6,
    "gamma":      0.9,
    # ---- 预算 ----
    "day_budget": 30.0,
    # ---- off-policy 行为策略偏差 ----
    "behavior_biased": False,
    # ---- 投放链路 ----
    "n_ads":            200,    # 广告库存规模
    "recall_K":         8,      # 每次请求召回候选数
    "predict_noise":    0.0,    # 预估随机噪声 std (0=无噪声)
    "predict_bias_std": 0.0,    # 每广告系统性预估偏差 std (0=无偏)
    # ---- 成本 ----
    "coin_cost": 0.5,
}


# 模型可见的观测特征顺序
OBS_KEYS = [
    # 用户画像 (episode 级常量)
    "u_ltv", "u_sens", "u_active7", "u_city_lv", "u_device",
    # 本次"市场"摘要 (胜出广告的可见属性 + 召回竞争强度)
    "win_bid", "win_type", "win_ext_ecpm", "win_ecpm", "second_ecpm",
    # World State 动态量
    "budget_left", "steps_left",
]


class Sandbox:
    """有状态环境。一次并行 N 个用户日 (向量化)。"""

    def __init__(self, config=None, seed=0):
        self.cfg = dict(CONFIG if config is None else config)
        self.rng = np.random.default_rng(seed)
        self.world = World()
        self.predictor = Predictor(self.cfg)
        self.inv = build_inventory(self.cfg["n_ads"], self.rng, self.cfg)
        self.N = 0
        self.user = None
        self.budget_left = None
        self.steps_left = None
        self.alive = None
        self.cur_winner = None     # 本步竞价胜出广告 (用户面对的"市场")

    # ================= 外生采样: 用户画像 =================
    def _sample_users(self, n):
        u = {}
        u["u_ltv"]     = self.rng.uniform(0.0, 2.0, n)
        u["u_sens"]    = self.rng.uniform(0.0, 1.0, n)
        u["u_active7"] = self.rng.integers(0, 8, n).astype(float)
        u["u_city_lv"] = self.rng.integers(1, 6, n).astype(float)
        u["u_device"]  = self.rng.integers(0, 3, n).astype(float)
        return u

    # ================= 一次竞价: 召回→预估→rank→竞胜 =================
    def _run_auction(self, coins):
        """对当前 N 个请求跑投放链路, 返回胜出广告 dict。
        coins 用于 A 类广告的预估 eCPM (金币抬高 pCTR→eCPM)。"""
        cand = recall(self.inv, self.N, self.cfg["recall_K"], self.rng)
        return auction(self.inv, cand, self.user, coins, self.predictor, self.world, self.rng)

    # ================= 组装 observation =================
    def _obs(self):
        w = self.cur_winner
        cols = {
            "u_ltv": self.user["u_ltv"], "u_sens": self.user["u_sens"],
            "u_active7": self.user["u_active7"], "u_city_lv": self.user["u_city_lv"],
            "u_device": self.user["u_device"],
            "win_bid": w["g_bid"], "win_type": w["ad_type"],
            "win_ext_ecpm": w["g_ext_ecpm"], "win_ecpm": w["win_ecpm"],
            "second_ecpm": w["clearing_price"],
            "budget_left": self.budget_left, "steps_left": self.steps_left,
        }
        return np.stack([cols[k] for k in OBS_KEYS], axis=1)

    # ================= Gym 接口 =================
    def reset(self, n):
        """开 n 个新用户日。返回首个 observation。
        注意: 竞价依赖金币, reset 时尚无动作 → 用金币=0 预跑一次竞价定"当前市场"。"""
        self.N = n
        self.user = self._sample_users(n)
        self.budget_left = np.full(n, float(self.cfg["day_budget"]))
        self.steps_left = np.full(n, float(self.cfg["horizon"]))
        self.alive = np.ones(n, dtype=bool)
        self.cur_winner = self._run_auction(np.zeros(n))
        return self._obs()

    def step(self, coins):
        """世界推进一步。返回 (obs, info, done)。info 是客观事实, 不含 reward。"""
        coins = np.asarray(coins, dtype=float)
        coins = np.minimum(coins, self.budget_left)
        coins = np.where(self.alive, coins, 0.0)

        # 用实际金币重跑竞价 (金币影响 A 类 eCPM→竞胜)
        winner = self._run_auction(coins)
        out = self.world.realize(winner, self.user, coins,
                                 winner["clearing_price"], self.cfg["coin_cost"], self.rng)
        out["coins"] = coins
        out["ad_id"] = winner["ad_id"]
        # 已死 episode 的客观量清零
        for k in ("profit", "revenue", "cost", "clicked", "watched"):
            out[k] = np.where(self.alive, out[k], 0.0)

        # ---- 演化 World State ----
        self.budget_left = np.maximum(self.budget_left - out["cost"], 0.0)
        self.steps_left = self.steps_left - 1.0
        survived = out["survived"].astype(bool) & self.alive
        budget_done = self.budget_left < COINS[COINS > 0].min()
        time_done = self.steps_left <= 0
        new_alive = survived & (~time_done) & (~budget_done)
        done = self.alive & (~new_alive)
        active = self.alive.copy()
        self.alive = new_alive

        # ---- 切换到下一步"市场" (金币=0 预竞价, 真实金币在下次 step 重算) ----
        self.cur_winner = self._run_auction(np.zeros(self.N))

        out["active"] = active           # 本步是否真实发生
        return self._obs(), out, done
