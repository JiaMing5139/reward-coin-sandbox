"""
reward.py —— 奖励设计 (独立于环境)

环境(sandbox/world)只吐【客观事实】info (谁竞胜/是否点击/收入/成本/留存...)。
"把这些客观量组合成 reward" 是建模选择, 不是环境的客观属性 —— 所以独立成此文件。
换优化目标(利润 → 利润−体验惩罚 → ...)时只动这里, 环境一行不改。
"""
import numpy as np


def reward_fn(info, cfg=None):
    """默认奖励 = 平台利润 (收入 − 金币成本)。

    info: sandbox.step 返回的客观事实 dict。
    返回: (n,) reward。

    可扩展示例(预留, 默认关闭):
      - 体验惩罚: reward = profit - λ·(发的金币过多带来的体验损耗)
      - 留存激励: reward = profit + β·survived
    """
    reward = info["profit"].copy()
    if cfg:
        lam = cfg.get("exp_penalty", 0.0)
        if lam:
            # 体验惩罚示例: 金币越多对体验损耗越大(此处用 cost 近似)
            reward = reward - lam * info["cost"]
        beta = cfg.get("retain_bonus", 0.0)
        if beta:
            reward = reward + beta * info["survived"]
    return reward
