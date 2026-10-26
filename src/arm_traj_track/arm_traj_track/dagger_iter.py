# -*- coding: utf-8 -*-
"""DAgger(Dataset Aggregation) 迭代: 缓解监督学习闭环的"分布偏移"。

流程:
  1. 用当前 MLP 部署, 收集其自身轨迹状态(覆盖真实部署分布)
  2. 用 IK+PD 专家对这些状态(含前瞻点)标注正确动作
  3. 与原始演示数据合并, 重新训练 MLP
  4. 重复 2 轮, 使 MLP 在部署分布上也准确 -> 闭环稳定跟踪
"""
import numpy as np

from arm3_env import Arm3Env, inverse_kinematics, KP, LOOKAHEAD_STEPS
from nn import MLP, Adam

DIMS = [9, 128, 128, 3]


def make_policy(net, Xstat, Ystat):
    Xm, Xs = Xstat[0], Xstat[1]
    Ym, Ys = Ystat[0], Ystat[1]

    def p(s):
        xs = (np.asarray(s, dtype=np.float64) - Xm) / Xs
        return net.forward(xs.reshape(1, -1))[0] * Ys + Ym

    return p


def collect_states(env, pol, n_eps=60, max_steps=250):
    """用当前策略部署, 收集自身轨迹状态及对应前瞻点目标。"""
    out = []
    for ep in range(n_eps):
        kind = 'circle' if ep % 2 == 0 else 'eight'
        env.reset(kind=kind, perturb=0.15)
        for _ in range(max_steps):
            s = env.state()
            fx, fy, fth = env._lookahead()
            out.append((s, (fx, fy, fth)))
            env.step(pol(s))
    return out


def label_pd(states):
    """用 IK+PD 专家对状态标注正确动作。"""
    X, Y = [], []
    for s, tgt in states:
        q1 = np.arctan2(s[0], s[1])
        q2 = np.arctan2(s[2], s[3])
        q3 = np.arctan2(s[4], s[5])
        qd = inverse_kinematics(tgt[0] + 0.15, tgt[1])   # 前瞻点(全局)->相对基座
        X.append(s)
        Y.append([np.clip(KP * (qd[0] - q1), -1, 1),
                  np.clip(KP * (qd[1] - q2), -1, 1),
                  np.clip(KP * (qd[2] - q3), -1, 1)])
    return np.asarray(X), np.asarray(Y)


def train_mlp(X, Y, dims, epochs=400, lr=3e-3, seed=0):
    Xmean, Xstd = X.mean(0), X.std(0) + 1e-8
    Ymean, Ystd = Y.mean(0), Y.std(0) + 1e-8
    Xs = (X - Xmean) / Xstd
    Ys = (Y - Ymean) / Ystd
    net = MLP(dims, seed=seed)
    opt = Adam(lr=lr)
    n = len(X)
    for ep in range(epochs):
        idx = np.random.permutation(n)
        for s in range(0, n, 64):
            b = idx[s:s + 64]
            pred = net.forward(Xs[b])
            g = net.mse_grad(Ys[b])
            gw, gb = net.backward(g)
            opt.step(*net.params(), gw, gb)
        if (ep + 1) % 100 == 0:
            print(f"  epoch {ep+1}/{epochs} done")
    return net, Xmean, Xstd, Ymean, Ystd


def main(iter_n=2, n_eps=60, max_steps=250, epochs=400, lr=3e-3):
    np.random.seed(0)
    env = Arm3Env()
    net = MLP(DIMS)
    net.load("mlp.npz")
    Xstat = np.load("Xstat.npy")
    Ystat = np.load("Ystat.npy")
    pol = make_policy(net, Xstat, Ystat)
    d = np.load("demo.npz")
    X0, Y0 = d["X"], d["Y"]

    for it in range(iter_n):
        states = collect_states(env, pol, n_eps, max_steps)
        Xn, Yn = label_pd(states)
        X = np.vstack([X0, Xn])
        Y = np.vstack([Y0, Yn])
        print(f"== iter {it}: total {len(X)} (new {len(Xn)}) ==")
        net, Xm, Xs, Ym, Ys = train_mlp(X, Y, DIMS, epochs, lr)
        net.save("mlp.npz")
        np.save("Xstat.npy", np.stack([Xm, Xs]))
        np.save("Ystat.npy", np.stack([Ym, Ys]))
        Xstat = np.load("Xstat.npy")
        Ystat = np.load("Ystat.npy")
        pol = make_policy(net, Xstat, Ystat)
    print("DAgger done, saved mlp.npz")


if __name__ == "__main__":
    main()
