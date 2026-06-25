"""
world.py —— 真实世界 (上帝视角的"物理定律")

只定义【客观真相】: 金币如何影响用户观看/点击/留存, 以及真实采样。
这些参数对模型、对 ads_infra 的预估层都【不可见】。

与 ads_infra 的分工:
  - world: 真实 pCTR / 观看 / 留存 (真值规则) + 用真值采样最终结果。
  - ads_infra: 平台对 pCTR 的【预估】(带误差) + 竞价排序 + 计费。
两者的差距(预估 vs 真实)正是"预估不准→排错→收入损失"的来源。
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class World:
    """真实世界的物理定律。所有参数是上帝私有, 不可见。"""

    # ---- 点击真值 (注意: 这是真实 pCTR, ads_infra 只能预估它) ----
    COIN_TO_CTR = 0.06
    LTV_TO_CTR  = 0.40
    GQ_TO_CTR   = 6.0
    CTR_BIAS    = -2.2
    # ---- 用户观看意愿 (即时: 金币换观看) ----
    WATCH_BIAS    = -0.5
    COIN_TO_WATCH = 0.22
    SENS_TO_WATCH = 2.0
    # ---- 用户留存 (长期: 金币换"下一步还来") ----
    RETAIN_BIAS    = -0.6
    COIN_TO_RETAIN = 0.28
    SENS_TO_RETAIN = 0.8

    # ============ 真值规则 ============
    def pctr_true(self, f, coins):
        """真实点击率。ads_infra 看不到这个, 只能预估。"""
        logit = (self.CTR_BIAS + self.LTV_TO_CTR * f["u_ltv"]
                 + self.GQ_TO_CTR * f["g_ctr"] + self.COIN_TO_CTR * coins)
        return sigmoid(logit)

    def watch_prob(self, f, coins):
        """用户观看意愿 (金币是看广告的对价)。"""
        logit = (self.WATCH_BIAS + self.COIN_TO_WATCH * coins
                 - self.SENS_TO_WATCH * f["u_sens"])
        return sigmoid(logit)

    def retain_prob(self, f, coins):
        """看完这条后继续留下看下一条的概率。金币越多越愿留 → 长期价值来源。"""
        logit = (self.RETAIN_BIAS + self.COIN_TO_RETAIN * coins
                 - self.SENS_TO_RETAIN * f["u_sens"])
        return sigmoid(logit)

    # ============ 真实采样 ============
    def realize(self, winner, user, coins, clearing_price, coin_cost, rng):
        """胜出广告进入真实曝光/点击/留存流程, 返回【客观事实】。

        参数:
          winner: dict, 胜出广告的真实特征 (含 ad_type, g_bid, g_ctr, g_ext_ecpm)
          user:   dict, 用户特征 (u_ltv, u_sens, ...)
          coins:  (n,) 本步发放金币
          clearing_price: (n,) 二价成交价 (次高位 eCPM, 由 ads_infra 给出)
          coin_cost: 金币面值→成本折算系数
        返回: dict 客观事实 (watched/won/clicked/revenue/cost/profit/survived)
        """
        n = len(coins)
        f = {**user, **winner}
        watch = self.watch_prob(f, coins)
        watched = (rng.uniform(0, 1, n) < watch).astype(float)
        # won 由 ads_infra 决定(是否有广告竞胜); 这里 winner 已是胜出者, won=是否有有效曝光机会
        won = winner["won"]
        exposed = watched * won
        pc = self.pctr_true(f, coins)                 # 用真实 pCTR 采样点击
        clicked = exposed * (rng.uniform(0, 1, n) < pc).astype(float)

        # A 类: 点击计费(二价); B 类: 曝光结算外生 eCPM
        cpc = np.where(pc > 1e-6, clearing_price / np.maximum(pc, 1e-6), 0.0)
        rev_A = clicked * cpc
        rev_B = exposed * winner["g_ext_ecpm"]
        revenue = np.where(winner["ad_type"] == 0, rev_A, rev_B)
        cost = exposed * coin_cost * coins
        profit = revenue - cost

        retain = self.retain_prob(f, coins)
        survived = (rng.uniform(0, 1, n) < retain).astype(float)
        return {"watched": watched, "won": won, "clicked": clicked,
                "revenue": revenue, "cost": cost, "profit": profit,
                "survived": survived, "pctr_true": pc}
