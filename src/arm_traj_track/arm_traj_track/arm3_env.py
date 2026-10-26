"""MuJoCo 3 关节平面机械臂连续轨迹跟踪环境。

机器人：3 关节平面机械臂（L1=L2=L3=0.15m），末端点可自由控制。
任务：末端点沿给定轨迹（圆 / 8 字）连续跟踪（仅位置，不约束朝向）。
感知：3 关节编码器(三角编码) + 轨迹前瞻点相对末端坐标差 + 末端朝向误差。
控制：3 个关节扭矩。
专家：最小范数数值逆运动学(IK) + 前瞻点轨迹规划 + PD 控制器。
"""
import math
import os

import numpy as np
import mujoco

L1, L2, L3 = 0.15, 0.15, 0.15
TRAJ_CENTER = (0.22, 0.0)    # 轨迹中心(距基座0.37, 机械臂半伸)
TRAJ_RADIUS = 0.05            # 轨迹半径(距基座0.32~0.42, 均 < 最大伸展0.45)
TRAJ_SPEED = 0.35          # rad/s，轨迹参数推进速度
LOOKAHEAD_STEPS = 8         # 前瞻步数
KP = 8.0
KD = 0.0
Q_LIMIT = 2.5               # 数值IK的关节角限幅(略小于xml的3.0, 避免撞限位)
STEPS_PER_EP = 500


def _xml_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'models', 'arm3.xml')


def forward_kinematics(q1, q2, q3):
    """正运动学：由关节角求末端全局坐标与朝向。返回 (x, y, theta)。"""
    c1, s1 = math.cos(q1), math.sin(q1)
    c12, s12 = math.cos(q1 + q2), math.sin(q1 + q2)
    c123, s123 = math.cos(q1 + q2 + q3), math.sin(q1 + q2 + q3)
    x = L1 * c1 + L2 * c12 + L3 * c123
    y = L1 * s1 + L2 * s12 + L3 * s123
    theta = q1 + q2 + q3
    return x, y, theta


def inverse_kinematics(tx, ty, q_curr=None):
    """2 关节解析位置 IK：末端点到 (tx,ty)，第 3 关节伸直(q3=0)。

    q3=0 时末端 = L1*c1 + (L2+L3)*c12，等效两段：A=L1, B=L2+L3。
    用 2 关节解析法解 q1,q2（无冗余、解唯一、稳定），q3 保持伸直。
    返回 (q1, q2, 0.0)。
    """
    A = L1
    B = L2 + L3
    d = math.hypot(tx, ty)
    d = max(1e-3, min(d, A + B - 1e-4))
    cos2 = (d * d - A * A - B * B) / (2 * A * B)
    cos2 = max(-1.0, min(1.0, cos2))
    q2 = math.acos(cos2)
    q1 = math.atan2(ty, tx) - math.atan2(B * math.sin(q2),
                                         A + B * math.cos(q2))
    return q1, q2, 0.0


def traj_point(t, kind='circle'):
    """给定轨迹参数 t，返回轨迹上的点与目标朝向 (x, y, theta_d)。"""
    cx, cy = TRAJ_CENTER
    r = TRAJ_RADIUS
    if kind == 'circle':
        x = cx + r * math.cos(t)
        y = cy + r * math.sin(t)
        theta_d = t - math.pi / 2.0        # 末端沿切向(仅作感知展示)
    else:                                   # eight 8 字
        x = cx + r * math.sin(2.0 * t)
        y = cy + r * math.sin(t)
        theta_d = math.atan2(math.cos(t), 2.0 * math.cos(2.0 * t))
    return x, y, theta_d


class Arm3Env:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(_xml_path())
        self.data = mujoco.MjData(self.model)
        self.tip_id = self.model.body('tip').id
        self.traj_t = 0.0
        self.kind = 'circle'

    def _base_xy(self):
        return (float(self.model.body('base').pos[0]),
                float(self.model.body('base').pos[1]))

    def reset(self, kind=None, init_q=None, perturb=0.1):
        self.kind = kind if kind is not None else self.kind
        self.traj_t = 0.0
        mujoco.mj_resetData(self.model, self.data)
        if init_q is None:
            fx, fy, _ = self._lookahead()
            bx, by = self._base_xy()
            qd = inverse_kinematics(fx - bx, fy - by)
            init_q = np.array(qd) + np.random.uniform(-perturb, perturb, 3)
        self.data.qpos[0:3] = init_q
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def _tip(self):
        return float(self.data.xpos[self.tip_id, 0]), \
               float(self.data.xpos[self.tip_id, 1]), \
               float(self.data.qpos[0] + self.data.qpos[1] + self.data.qpos[2])

    def _lookahead(self):
        """轨迹前瞻点(位置+朝向)。"""
        return traj_point(self.traj_t + LOOKAHEAD_STEPS * self.model.opt.timestep,
                          self.kind)

    def state(self):
        """6维三角编码 + 前瞻点相对末端差 + 末端朝向误差 = 9维感知。"""
        q = self.data.qpos[0:3]
        x, y, th = self._tip()
        fx, fy, fth = self._lookahead()
        dx, dy = fx - x, fy - y
        e_th = fth - th
        return np.array([math.sin(q[0]), math.cos(q[0]),
                         math.sin(q[1]), math.cos(q[1]),
                         math.sin(q[2]), math.cos(q[2]),
                         dx, dy, e_th], dtype=np.float32)

    def ik_pd_ctrl(self, kp=KP, kd=KD):
        """IK + PD 专家控制：对前瞻点求目标关节角(最小范数数值IK)，PD出力矩。"""
        fx, fy, fth = self._lookahead()
        bx, by = self._base_xy()
        qd = inverse_kinematics(fx - bx, fy - by)
        q = self.data.qpos[0:3]
        dq = self.data.qvel[0:3]
        u = np.clip(kp * (np.array(qd) - q) - kd * dq, -1.0, 1.0)
        return u

    def step(self, ctrl):
        self.data.ctrl[0:3] = np.clip(ctrl, -2.0, 2.0)
        mujoco.mj_step(self.model, self.data)
        self.traj_t += TRAJ_SPEED * self.model.opt.timestep
        return self.state()
