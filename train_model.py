import numpy as np


class Train:
    def __init__(self):
        self.weight = 337.8  # 车重（吨）
        # 戴维斯参数
        self.a = 8.4
        self.b = 0.1071
        self.c = 0.00472

        # 效率参数
        self.n1 = 0  # 牵引电机效率
        self.n1_b = 0  # 制动电机效率
        self.n2 = 0.9702  # 变压器效率
        self.n3 = 0.96  # 变流器效率
        self.n4 = 0.97  # 齿轮箱效率

        # 最大牵引力最大制动力
        self.max_traction_force = 0
        self.max_brake_force = 0

        # 最大加速度最大减速度
        self.max_trac_acc = 1.5
        self.max_bra_acc = -1.2

    def get_max_traction_force(self, cur_v):  # 速度单位为km/h，用当前速度
        self.get_n1(cur_v)
        if cur_v <= 57:
            self.max_traction_force = 409
        else:
            self.max_traction_force = 0.02893 * cur_v * cur_v - 8.417 * cur_v + 785.8

    def get_max_brake_force(self, cur_v):  # 速度单位为km/h
        self.get_n1_b(cur_v)
        if cur_v <= 6:
            self.max_brake_force = 63.29 * cur_v
        elif 6 < cur_v <= 100:
            self.max_brake_force = 379.75
        else:
            self.max_brake_force = -2.685 * cur_v + 642

    # 计算牵引电机效率
    def get_n1(self, cur_v):  # 速度单位为km/h
        if cur_v <= 55:
            self.n1 = -0.802 * np.exp(-0.1467 * cur_v) + 0.8904
        else:
            self.n1 = 0.927

    # 计算制动电机效率
    def get_n1_b(self, cur_v):  # 速度单位为km/h
        if 0 < cur_v <= 60:
            self.n1_b = -0.09175 * np.exp(17 / cur_v) + 1.048
            if self.n1_b < 0:
                self.n1_b = 0
        else:
            self.n1_b = 0.94

    # 计算牵引能耗和再生制动能耗
    def get_traction_power(self, ave_v, delta_t, action):  # 速度单位为km/h，用平均速度，时间单位为s，action为算法输出的百分比
        return self.max_traction_force * (action / 1) * (ave_v / 3.6) * (delta_t / 3600) / (self.n1 * self.n2 * self.n3 * self.n4)

    def get_re_power(self, ave_v, delta_t, action):  # 速度单位为km/h，时间单位为s，action为算法输出的百分比
        return self.max_brake_force * (action / 1) * (ave_v / 3.6) * (delta_t / 3600) * (self.n1_b * self.n2 * self.n3 * self.n4)
