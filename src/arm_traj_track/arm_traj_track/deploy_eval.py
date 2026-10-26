# -*- coding: utf-8 -*-
"""部署评估: 加载训练好的 MLP, 在 MuJoCo 中形成
"感知(关节编码+前瞻点+朝向误差) -> 神经网络 -> 扭矩控制" 的闭环,
评估末端对连续轨迹的平均跟踪误差, 并对比 IK+PD 专家。"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from arm3_env import Arm3Env, traj_point
from nn import MLP

DIMS = [9, 128, 128, 3]


def make_policy(net, Xstat, Ystat):
    Xmean, Xstd = Xstat[0], Xstat[1]
    Ymean, Ystd = Ystat[0], Ystat[1]

    def policy(s):
        xs = (np.asarray(s, dtype=np.float64) - Xmean) / Xstd
        ys = net.forward(xs.reshape(1, -1))[0]
        return ys * Ystd + Ymean

    return policy


def run_episode(env, policy, max_steps=300, use_pd=False):
    kind = 'circle' if np.random.rand() < 0.5 else 'eight'
    env.reset(kind=kind, perturb=0.15)
    curve = []
    for _ in range(max_steps):
        u = env.ik_pd_ctrl() if use_pd else policy(env.state())
        env.step(u)
        x, y, _ = env._tip()
        cx, cy, _ = traj_point(env.traj_t, env.kind)     # 当前轨迹点
        curve.append(np.hypot(x - cx, y - cy))
    return curve


def main(n_eps=30, max_steps=300, seed=0):
    np.random.seed(seed)
    env = Arm3Env()
    net = MLP(DIMS)
    net.load("mlp.npz")
    Xstat = np.load("Xstat.npy")
    Ystat = np.load("Ystat.npy")
    policy = make_policy(net, Xstat, Ystat)

    mlp_curves = [run_episode(env, policy, max_steps) for _ in range(n_eps)]
    pd_curves = [run_episode(env, None, max_steps, use_pd=True) for _ in range(n_eps)]

    def avg(curves):
        return np.array([np.mean(c) for c in curves])

    mlp, pd = avg(mlp_curves), avg(pd_curves)
    print("=" * 50)
    print("MLP 闭环: mean_err=%.4f median=%.4f success(<0.05)=%.2f%%"
          % (mlp.mean(), np.median(mlp), (mlp < 0.05).mean() * 100))
    print("IK+PD:   mean_err=%.4f median=%.4f success(<0.05)=%.2f%%"
          % (pd.mean(), np.median(pd), (pd < 0.05).mean() * 100))

    plt.figure(figsize=(7, 5))
    for c in mlp_curves[:3]:
        plt.plot(np.arange(len(c)), c, alpha=0.8)
    plt.xlabel("control step")
    plt.ylabel("tracking error (m)")
    plt.title("MLP closed-loop trajectory tracking error")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("tracking_error_curve.png", dpi=120)
    print("saved: tracking_error_curve.png")

    with open("eval_summary.csv", "w") as f:
        f.write("method,mean_err,median_err,success_rate\n")
        f.write("MLP,%.4f,%.4f,%.2f\n" % (mlp.mean(), np.median(mlp), (mlp < 0.05).mean() * 100))
        f.write("IKPD,%.4f,%.4f,%.2f\n" % (pd.mean(), np.median(pd), (pd < 0.05).mean() * 100))
    print("saved: eval_summary.csv")


if __name__ == "__main__":
    main()
