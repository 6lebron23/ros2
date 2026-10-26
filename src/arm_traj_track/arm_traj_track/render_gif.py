# -*- coding: utf-8 -*-
"""渲染动图: 用训练好的 MLP 闭环驱动机械臂跟踪连续轨迹, 用 matplotlib(Agg, 无显示器)
绘制机械臂 + 轨迹 + 目标点, 输出 gif。"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

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


def get_pts(env):
    pts = []
    for nm in ('base', 'link1', 'link2', 'link3', 'tip'):
        p = env.data.body(nm).xpos[:2]
        pts.append((float(p[0]), float(p[1])))
    return pts


def main(kind='circle', max_steps=400, out='arm_traj.gif', seed=7):
    np.random.seed(seed)
    env = Arm3Env()
    net = MLP(DIMS)
    net.load("mlp.npz")
    Xstat = np.load("Xstat.npy")
    Ystat = np.load("Ystat.npy")
    policy = make_policy(net, Xstat, Ystat)

    env.reset(kind=kind, perturb=0.15)

    # 预生成轨迹曲线
    tt = np.linspace(0, 6.28, 120)
    tr_x, tr_y = [], []
    for t in tt:
        x, y, _ = traj_point(t, kind)
        tr_x.append(x)
        tr_y.append(y)

    frames = []
    for _ in range(max_steps):
        u = policy(env.state())
        env.step(u)
        pts = get_pts(env)
        cx, cy, _ = traj_point(env.traj_t, env.kind)
        err = np.hypot(pts[-1][0] - cx, pts[-1][1] - cy)
        frames.append((pts, cx, cy, err))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.set_xlim(0, 0.7)
    ax.set_ylim(-0.4, 0.4)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.plot(tr_x, tr_y, 'k--', lw=1, label='target trajectory')
    (line,) = ax.plot([], [], 'o-', lw=3, color='#0a7ac9', ms=6)
    (tgt,) = ax.plot([], [], 'o', color='r', ms=8)
    err_text = ax.text(0.03, 0.35, '', fontsize=10)
    ax.legend(loc='upper left', fontsize=8)

    def update(i):
        pts, cx, cy, err = frames[i]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        line.set_data(xs, ys)
        tgt.set_data([cx], [cy])
        err_text.set_text('error=%.3f m' % err)
        return line, tgt, err_text

    ani = FuncAnimation(fig, update, frames=len(frames), interval=50, blit=False)
    ani.save(out, writer=PillowWriter(fps=20))
    print('saved:', out)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', default='circle')
    ap.add_argument('--out', default='arm_traj.gif')
    ap.add_argument('--max_steps', type=int, default=400)
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()
    main(kind=a.kind, max_steps=a.max_steps, out=a.out, seed=a.seed)
