"""
Sandbox —— 上帝视角的真实环境模拟器 (有状态 / 序列决策版)

三层抽象 (务必分清):
  1. 上帝参数   —— 世界的"物理定律"(金币如何影响观看/点击/竞胜/留存)。
                  全程不变, 对模型【不可见】。是 Sandbox 的常量。
  2. World State —— 世界的"当前状况", 随 step() 真实演化的动态量。
                  只有三个: budget_left(剩余预算) / steps_left(剩余请求数) / alive(是否留存)。
                  外加 episode 级常量: 当前用户画像(reset 时设定, 一天内不变)。
  3. Observation —— 模型能看到的那部分 state 向量, 是 World State 的投影
                  (用户画像 + 本次广告/上下文 + 剩余预算 + 剩余步数)。上帝参数看不到。

为什么有这三个动态状态, 单步版没有:
  - budget_left: 全局预算约束(pacing) → 逼策略把金币留给高价值请求
  - steps_left:  序列长度信息       → 让策略知道还剩几次机会
  - alive:       金币换留存的结果   → 留存是"长期价值"的来源(发慷慨→用户留→未来更多曝光)

接口 (标准 Gym 范式, 向量化: 一次并行 N 个用户日):
  reset()        → 开 N 个新用户日, 返回首个 observation 矩阵
  step(actions)  → 世界并行推进一步, 返回 (obs, reward, done, info)
  本次广告/上下文是【外生】的(每步随机注入, 不受 action 影响), 不属于 World State。
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# 动作空间: 10 档金币 (0, 2, 4, ..., 18)
COINS = np.arange(0, 20, 2.0)


# ======================================================================
# 配置 (一处改, 全局生效)
# ======================================================================
CONFIG = {
    # ---- 联盟广告 ----
    "union_enabled": True,
    "union_ratio":   0.30,
    "union_ecpm_lo": 5.0,
    "union_ecpm_hi": 11.0,
    # ---- 序列决策 ----
    "horizon":    6,        # 一个用户日最多几次广告请求
    "gamma":      0.9,      # 折扣因子 (未来利润打折)
    # ---- 预算 ----
    "day_budget": 30.0,     # 一个用户日的金币预算 (会绑定, 不够无脑发满)
    # ---- off-policy 行为策略偏差 (#5) ----
    "behavior_biased": False,  # False=无偏随机探索; True=有偏线上策略(制造覆盖空洞)
}


# 模型可见的观测特征顺序 (Observation 向量)
OBS_KEYS = [
    # ---- 用户画像 (episode 级常量) ----
    "u_ltv", "u_sens", "u_active7", "u_city_lv", "u_device",
    # ---- 本次广告 / 上下文 (每步外生重抽) ----
    "g_bid", "g_ctr", "g_cat", "g_material", "c_hour", "c_weekend", "c_slot",
    "ad_type", "g_ext_ecpm",
    # ---- World State 动态量 (随 step 演化) ----
    "budget_left", "steps_left",
]


class Sandbox:
    """有状态的真实世界。一次并行 N 个用户日 (向量化)。"""

    # ================= 第1层: 上帝参数 (定律, 不可见) =================
    COIN_TO_CTR = 0.06
    LTV_TO_CTR  = 0.40
    GQ_TO_CTR   = 6.0
    CTR_BIAS    = -2.2
    WIN_SHARP   = 1.0
    THRESH_BASE = 6.0
    THRESH_PEAK = 2.0
    COIN_COST   = 0.5
    # 即时: 金币换观看
    WATCH_BIAS    = -0.5
    COIN_TO_WATCH = 0.22
    SENS_TO_WATCH = 2.0
    # 长期: 金币换留存 (序列价值的来源) —— 标定为"留存价值够大", 让长期>短视
    RETAIN_BIAS    = -0.6     # 基础留存低 (不发金币就容易流失)
    COIN_TO_RETAIN = 0.28     # 金币显著提升留存 (多发→留住→未来更多曝光)
    SENS_TO_RETAIN = 0.8

    def __init__(self, config=None, seed=0):
        self.cfg = dict(CONFIG if config is None else config)
        self.rng = np.random.default_rng(seed)
        self.N = 0
        # World State (动态量) —— reset 时初始化
        self.user = None          # 当前用户画像 (episode 级常量)
        self.budget_left = None
        self.steps_left = None
        self.alive = None
        self.cur_req = None        # 本次外生广告/上下文

    # ================= 物理定律: (用户,广告,金币) → 各概率 =================
    def watch_prob(self, f, coins):
        logit = (self.WATCH_BIAS + self.COIN_TO_WATCH * coins
                 - self.SENS_TO_WATCH * f["u_sens"])
        return sigmoid(logit)

    def pctr(self, f, coins):
        logit = (self.CTR_BIAS + self.LTV_TO_CTR * f["u_ltv"]
                 + self.GQ_TO_CTR * f["g_ctr"] + self.COIN_TO_CTR * coins)
        return sigmoid(logit)

    def ecpm(self, f, coins):
        ecpm_self = f["g_bid"] * self.pctr(f, coins)
        return np.where(f["ad_type"] == 0, ecpm_self, f["g_ext_ecpm"])

    def win_threshold(self, f):
        peak = self.THRESH_PEAK * np.exp(-((f["c_hour"] - 20.0) ** 2) / (2 * 2.5 ** 2))
        return self.THRESH_BASE + peak

    def win_prob(self, f, coins):
        return sigmoid(self.WIN_SHARP * (self.ecpm(f, coins) - self.win_threshold(f)))

    def retain_prob(self, f, coins):
        """看完这条后继续留下看下一条的概率。金币越多越愿留 → 长期价值来源。"""
        logit = (self.RETAIN_BIAS + self.COIN_TO_RETAIN * coins
                 - self.SENS_TO_RETAIN * f["u_sens"])
        return sigmoid(logit)

    def immediate_profit(self, f, coins):
        """单步期望即时利润 (不含留存的未来价值)。给短视基线/分析用。"""
        watch = self.watch_prob(f, coins)
        wp = self.win_prob(f, coins)
        revenue = np.where(f["ad_type"] == 0, self.win_threshold(f), f["g_ext_ecpm"])
        return watch * wp * (revenue - self.COIN_COST * coins)

    # ================= 外生采样: 用户画像 / 本次广告 =================
    def _sample_users(self, n):
        u = {}
        u["u_ltv"]     = self.rng.uniform(0.0, 2.0, n)            # [因果]
        u["u_sens"]    = self.rng.uniform(0.0, 1.0, n)            # [因果]
        u["u_active7"] = self.rng.integers(0, 8, n).astype(float) # [噪声]
        u["u_city_lv"] = self.rng.integers(1, 6, n).astype(float) # [噪声]
        u["u_device"]  = self.rng.integers(0, 3, n).astype(float) # [噪声]
        return u

    def _sample_requests(self, n):
        r = {}
        r["g_bid"]      = self.rng.uniform(8.0, 20.0, n)           # [因果]
        r["g_ctr"]      = self.rng.uniform(0.05, 0.30, n)          # [因果]
        r["g_cat"]      = self.rng.integers(0, 10, n).astype(float)# [噪声]
        r["g_material"] = self.rng.integers(0, 4, n).astype(float) # [噪声]
        r["c_hour"]     = self.rng.integers(0, 24, n).astype(float)# [因果]
        r["c_weekend"]  = self.rng.integers(0, 2, n).astype(float) # [噪声]
        r["c_slot"]     = self.rng.integers(0, 3, n).astype(float) # [噪声]
        if self.cfg["union_enabled"]:
            r["ad_type"]    = (self.rng.uniform(0, 1, n) < self.cfg["union_ratio"]).astype(float)
            r["g_ext_ecpm"] = self.rng.uniform(self.cfg["union_ecpm_lo"],
                                               self.cfg["union_ecpm_hi"], n)
        else:
            r["ad_type"]    = np.zeros(n)
            r["g_ext_ecpm"] = np.zeros(n)
        return r

    # ================= 组装 observation =================
    def _features(self):
        """当前 World State + 本次广告, 合成一个特征 dict (含上帝可算的全部字段)。"""
        f = dict(self.user)
        f.update(self.cur_req)
        f["budget_left"] = self.budget_left
        f["steps_left"]  = self.steps_left
        return f

    def _obs(self):
        f = self._features()
        return np.stack([f[k] for k in OBS_KEYS], axis=1)

    # ================= Gym 接口 =================
    def reset(self, n):
        """开 n 个新用户日。返回首个 observation 矩阵 (n, len(OBS_KEYS))。"""
        self.N = n
        self.user = self._sample_users(n)
        self.budget_left = np.full(n, float(self.cfg["day_budget"]))
        self.steps_left = np.full(n, float(self.cfg["horizon"]))
        self.alive = np.ones(n, dtype=bool)
        self.cur_req = self._sample_requests(n)
        return self._obs()

    def step(self, coins):
        """世界并行推进一步。

        参数: coins (n,) 本步各 episode 发放的金币 (策略给出)。
        返回: (obs, reward, done, info)
          obs:    下一步观测 (已切换到新广告 + 更新后的预算/步数)
          reward: (n,) 本步即时利润 (已死的 episode 记 0)
          done:   (n,) 本步后该 episode 是否结束
          info:   dict, 含 watched/won/clicked/cost/survived 等明细
        """
        f = self._features()
        coins = np.asarray(coins, dtype=float)
        # 预算约束: 实发金币不超过剩余预算 (clip 到可发范围, 再对齐到档位下界)
        coins = np.minimum(coins, self.budget_left)
        coins = np.where(self.alive, coins, 0.0)

        out = self._sample_outcome(f, coins)
        reward = np.where(self.alive, out["profit"], 0.0)

        # ---- 演化 World State ----
        self.budget_left = np.maximum(self.budget_left - out["cost"], 0.0)
        self.steps_left = self.steps_left - 1.0
        survived = out["survived"].astype(bool) & self.alive
        # episode 结束: 不再留存 / 步数用完 / 预算耗尽
        budget_done = self.budget_left < COINS[COINS > 0].min()  # 连最小档都发不起
        time_done = self.steps_left <= 0
        new_alive = survived & (~time_done) & (~budget_done)
        done = self.alive & (~new_alive)   # 本步从"活"变"死"的
        self.alive = new_alive

        # ---- 切换到下一条外生广告 ----
        self.cur_req = self._sample_requests(self.N)

        info = {**out, "coins": coins}
        return self._obs(), reward, done, info

    def _sample_outcome(self, f, coins):
        """真实采样一次拍卖结果 (含留存判定)。"""
        n = len(coins)
        rng = self.rng
        watch = self.watch_prob(f, coins)
        watched = (rng.uniform(0, 1, n) < watch).astype(float)
        wp = self.win_prob(f, coins)
        won = (rng.uniform(0, 1, n) < wp).astype(float)
        exposed = watched * won
        pc = self.pctr(f, coins)
        clicked = exposed * (rng.uniform(0, 1, n) < pc).astype(float)
        clearing = self.win_threshold(f)
        cpc = clearing / pc
        rev_A = clicked * cpc
        rev_B = exposed * f["g_ext_ecpm"]
        revenue = np.where(f["ad_type"] == 0, rev_A, rev_B)
        cost = exposed * self.COIN_COST * coins
        profit = revenue - cost
        retain = self.retain_prob(f, coins)
        survived = (rng.uniform(0, 1, n) < retain).astype(float)
        return {"watched": watched, "won": won, "clicked": clicked,
                "revenue": revenue, "cost": cost, "profit": profit,
                "survived": survived}
