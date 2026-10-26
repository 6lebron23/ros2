#!/usr/bin/env python3
"""用 IK+PD 专家生成监督演示数据(沿圆/8字轨迹)。"""
import argparse

import numpy as np

from arm3_env import Arm3Env, STEPS_PER_EP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=400)
    ap.add_argument('--out', default='demo.npz')
    args = ap.parse_args()

    env = Arm3Env()
    states, actions = [], []
    for ep in range(args.episodes):
        kind = 'circle' if ep % 2 == 0 else 'eight'
        s = env.reset(kind=kind)
        for _ in range(STEPS_PER_EP):
            u = env.ik_pd_ctrl()
            states.append(s)
            actions.append(u)
            s = env.step(u)
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    np.savez(args.out, X=states, Y=actions)
    print('[collect] 样本数=%d 状态维度=%d 动作维度=%d'
          % (states.shape[0], states.shape[1], actions.shape[1]))


if __name__ == '__main__':
    main()
