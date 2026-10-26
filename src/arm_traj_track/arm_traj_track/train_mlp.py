# -*- coding: utf-8 -*-
"""监督学习: 用演示数据(感知状态 -> 专家控制动作) 训练 MLP 神经网络,
使网络学出"从感知状态(关节编码+前瞻点差+朝向误差)到 3 关节扭矩"的映射,
从而在部署时仅凭网络即可闭环跟踪连续轨迹。"""
import numpy as np

from nn import MLP, Adam

DIMS = [9, 128, 128, 3]      # 感知9维 -> 3关节扭矩


def load_demo():
    d = np.load("demo.npz")
    return d["X"], d["Y"]


def main(epochs=300, batch=64, lr=3e-3, seed=0, n_samples=100000):
    np.random.seed(seed)
    X, Y = load_demo()
    if n_samples and n_samples < len(X):
        idx = np.random.RandomState(seed).choice(len(X), n_samples, replace=False)
        X, Y = X[idx], Y[idx]
    n = len(X)
    Xmean, Xstd = X.mean(0), X.std(0) + 1e-8
    Ymean, Ystd = Y.mean(0), Y.std(0) + 1e-8
    Xs = (X - Xmean) / Xstd
    Ys = (Y - Ymean) / Ystd
    np.save("Xstat.npy", np.stack([Xmean, Xstd]))
    np.save("Ystat.npy", np.stack([Ymean, Ystd]))

    net = MLP(DIMS, seed=seed)
    opt = Adam(lr=lr)
    losses = []
    for ep in range(epochs):
        idx = np.random.permutation(n)
        ep_loss, cnt = 0.0, 0
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            xb, yb = Xs[b], Ys[b]
            pred = net.forward(xb)
            g = net.mse_grad(yb)
            gw, gb = net.backward(g)
            opt.step(*net.params(), gw, gb)
            ep_loss += np.sum((pred - yb) ** 2)
            cnt += len(b)
        losses.append(ep_loss / cnt)
        if (ep + 1) % 30 == 0:
            print(f"epoch {ep+1}/{epochs}  mse={losses[-1]:.6f}")

    net.save("mlp.npz")
    np.save("losses.npy", np.array(losses))
    print("=" * 40)
    print("final mse=%.6f" % losses[-1])
    print("saved: mlp.npz, Xstat.npy, Ystat.npy, losses.npy")


if __name__ == "__main__":
    main()
