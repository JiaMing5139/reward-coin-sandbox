"""
Model —— 金币定价策略模型

职责:
  1. 定义 MLP 网络结构 (forward / backward / update)
  2. 提供训练接口 train()
  3. 提供推理接口 predict()

只负责模型本身, 不含数据生成/评估。
模型是通用回归器: 学的是 Q(obs, coins) ≈ 从该状态发该金币起的【折扣累计回报】。
"长期 vs 单步" 的区别在于 main 喂的训练目标(单步利润 or MC 折扣回报), 与本文件无关。
"""
import numpy as np


class MLP:
    """
    两层全连接网络, ReLU 激活, Adam 优化器。

    输入: [标准化obs, coins归一化, coins^2]
    输出: 标量 (预测折扣累计回报 Q)
    """

    def __init__(self, d_in, d_hid=64, seed=42):
        """
        He 初始化 (ReLU 网络标准做法)。

        参数:
          d_in: 输入维度
          d_hid: 隐层维度
          seed: 随机种子
        """
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, np.sqrt(2.0 / d_in), (d_in, d_hid))
        self.b1 = np.zeros(d_hid)
        self.W2 = r.normal(0, np.sqrt(2.0 / d_hid), (d_hid, 1))
        self.b2 = np.zeros(1)
        # Adam 优化器状态
        self._m = {k: 0.0 for k in ['W1', 'b1', 'W2', 'b2']}
        self._v = {k: 0.0 for k in ['W1', 'b1', 'W2', 'b2']}
        self._t = 0

    def forward(self, X, save_cache=False):
        """
        前向传播。

        参数:
          X: (n, d_in) 特征矩阵
          save_cache: 是否保存中间结果 (训练时需要, 推理时不需要)

        返回:
          out: (n, 1) 预测值
          cache: (可选) 中间结果, 用于反向传播
        """
        z1 = X @ self.W1 + self.b1
        h = np.maximum(0, z1)          # ReLU
        out = h @ self.W2 + self.b2
        if save_cache:
            return out, (X, z1, h)
        return out

    def backward(self, cache, grad_out):
        """
        反向传播, 计算梯度。

        参数:
          cache: forward() 保存的中间结果
          grad_out: 损失对输出的梯度 (n, 1)

        返回:
          grads: dict, 各参数的梯度
        """
        X, z1, h = cache
        g2 = grad_out                   # (n, 1)
        dW2 = h.T @ g2
        db2 = g2.sum(0)
        g1 = (g2 @ self.W2.T) * (z1 > 0)  # ReLU 反向
        dW1 = X.T @ g1
        db1 = g1.sum(0)
        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

    def update(self, grads, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Adam 优化器更新参数。

        参数:
          grads: backward() 返回的梯度
          lr: 学习率
          beta1, beta2: Adam 超参数
          eps: 数值稳定项
        """
        self._t += 1
        for k in grads:
            self._m[k] = beta1 * self._m[k] + (1 - beta1) * grads[k]
            self._v[k] = beta2 * self._v[k] + (1 - beta2) * grads[k] ** 2
            mhat = self._m[k] / (1 - beta1 ** self._t)
            vhat = self._v[k] / (1 - beta2 ** self._t)
            step = lr * mhat / (np.sqrt(vhat) + eps)
            setattr(self, k, getattr(self, k) - step)

    def train(self, X, y, n_epoch=80, batch_sz=512, lr=3e-3, rng=None, verbose=True):
        """
        训练模型 (batch SGD + Adam)。

        参数:
          X: (n, d_in) 特征矩阵
          y: (n,) 目标值 (已标准化)
          n_epoch: 训练轮数
          batch_sz: batch 大小
          lr: 学习率
          rng: numpy 随机数生成器
          verbose: 是否打印训练进度
        """
        if rng is None:
            rng = np.random.default_rng()
        n = len(X)
        for ep in range(n_epoch):
            perm = rng.permutation(n)
            loss_ep = 0.0
            for i in range(0, n, batch_sz):
                idx = perm[i:i + batch_sz]
                X_b, y_b = X[idx], y[idx][:, None]
                pred, cache = self.forward(X_b, save_cache=True)
                loss = ((pred - y_b) ** 2).mean()
                grad = 2 * (pred - y_b) / len(y_b)
                grads = self.backward(cache, grad)
                self.update(grads, lr)
                loss_ep += loss * len(y_b)
            if verbose and (ep + 1) % 10 == 0:
                print(f"  epoch {ep+1:2d}  loss(标准化)={loss_ep/n:.4f}")

    def predict(self, X):
        """
        推理 (不保存 cache)。

        参数:
          X: (n, d_in) 特征矩阵

        返回:
          (n,) 预测值
        """
        return self.forward(X, save_cache=False).ravel()
