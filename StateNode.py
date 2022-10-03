import numpy as np


class StateNode:
    def __init__(self, state, step, line, agent, episode, train_flag, train_model):
        self.line = line  # 区间
        self.train_model = train_model  # 列车模型
        self.agent = agent  # 神经网络
        # self.ou_noise = ou_noise  # OU噪声

        self.state = state  # 当前状态 ,0是时间，1是速度
        self.acc = 0  # 当前合加速度
        self.tm_acc = 0  # 电机牵引加速度
        self.bm_acc = 0  # 电机制动加速度
        self.g_acc = 0  # 坡度加速度
        self.c_acc = 0  # 曲率加速度
        self.action = np.array(0).reshape(1)  # 当前牵引制动力请求百分比

        self.step = step  # 当前阶段
        self.episode = episode  # 当前幕数
        self.max_step = self.line.length / self.line.delta_distance - 1
        self.current_reward = 0  # 当前奖励
        self.current_q = 0  # 当前Q值

        self.last_node_state = 0  # 前一个节点的状态
        self.last_node_action = 0  # 前一个节点的动作
        self.last_node_acc = 0  # 前一个节点的加速度

        self.next_state = np.zeros(2)  # 状态转移后的状态
        self.t_power = 0.0  # 状态转移过程的牵引能耗
        self.re_power = 0  # 状态转移过程产生的再生制动能量

        self.train_flag = train_flag  # 训练测试标志位
        self.comfort_punish = 0  # 舒适度惩罚标志位
        self.speed_punish = 0  # 超速惩罚标志位

        self.current_limit_speed = 0  # 当前限速
        self.c_limit_speed = float("inf")  # 用来画图
        self.a_limit_speed = float("inf")  # ATP限速，用来画图
        self.ave_v = 0  # 本阶段平均速度
        self.p_indicator = -10  # 超速惩罚

        self.ATP_limit = float("inf")

    # # 获取最大step
    # def get_max_step(self):
    #     self.max_step = self.line.length / self.line.delta_distance

    def get_last_node(self, node_list):  # 获取上一个节点的状态、动作加速度
        if len(node_list) != 1:
            self.last_node_state = node_list[self.step - 1].state
            self.last_node_action = node_list[self.step - 1].action
            self.last_node_acc = node_list[self.step - 1].acc
        else:
            self.last_node_state = np.zeros(2)
            self.last_node_action = np.array(0).reshape(1)
            self.last_node_acc = 0

    # 下面是动作的产生过程
    def get_action(self):  # 选择动作
        if self.train_flag:
            if self.episode > 100:
                self.action = self.agent.choose_action(self.state)
                self.action = np.array(self.action).reshape(1)
                self.action = self.action + np.random.normal(0, 1 * 0.1, size=1)
            else:
                self.action = np.array(np.random.uniform(-1, 1)).reshape(1)
        else:
            self.action = self.agent.choose_action(self.state)
            self.action = np.array(self.action).reshape(1)

    def get_ATP_limit(self):
        key_list = []
        key = 0
        key_next = 0
        for i in self.line.speed_limit.keys():
            key_list.append(i)
        for j in range(len(key_list) - 1):
            if key_list[j] <= self.step * self.line.delta_distance < key_list[j + 1]:
                key = key_list[j]  # 当前限速点位置
                key_next = key_list[j + 1]  # 下一个限速点位置
        limit_speed = self.line.speed_limit[key]  # 当前位置限速
        limit_speed_next = self.line.speed_limit[key_next]  # 下一个限速点限速
        final_locate = key_next // self.line.delta_distance
        length = final_locate - self.step
        if limit_speed_next < limit_speed:
            self.ATP_limit = np.sqrt(limit_speed_next * limit_speed_next / 3.6 / 3.6 - 2 * self.train_model.max_bra_acc * length * self.line.delta_distance) * 3.6

    def reshape_action(self):  # 重整动作
        if self.step <= 0.1 * self.max_step:
            self.action = (self.action + 1) / 2
        elif self.max_step - self.step <= 0.1 * self.max_step:
            self.action = (self.action - 1) / 2
        else:
            self.action = self.action
        low_bound = -1
        upper_bound = 1
        # 重整当前动作
        self.action = low_bound + (self.action + 1.0) * 0.5 * (upper_bound - low_bound)
        self.action = np.clip(self.action, low_bound, upper_bound)
        # 重整上一个节点的动作，这句话好像可以不要
        self.last_node_action = low_bound + (self.last_node_action + 1.0) * 0.5 * (upper_bound - low_bound)
        self.last_node_action = np.clip(self.last_node_action, low_bound, upper_bound)

    # 下面是合加速度的计算过程
    def get_gradient_acc(self):  # 获取当前位置坡度加速度
        key_list = []
        key = 0
        for i in self.line.gradient.keys():
            key_list.append(i)
        for j in range(len(key_list) - 1):
            if key_list[j] <= self.step * self.line.delta_distance < key_list[j + 1]:
                key = key_list[j]
        gradient = self.line.gradient[key]
        self.g_acc = -9.8 * gradient / 1000  # g_ acc = 9.8 * g /1000
        del key_list

    def get_curve_acc(self):  # 获取当前位置曲率加速度
        key_list = []
        key = 0
        for i in self.line.curve.keys():
            key_list.append(i)
        for j in range(len(key_list) - 1):
            if key_list[j] <= self.step * self.line.delta_distance < key_list[j + 1]:
                key = key_list[j]
        curve = self.line.curve[key]
        if curve != 0:
            self.c_acc = - 3 * 9.8 / (5 * curve)  # c_acc = 3g/5R
        else:
            self.c_acc = 0
        del key_list

    def get_current_tra_acc(self):  # 计算当前牵引加速度
        # self.train_model.get_max_traction_force(self.state[1] * 3.6)  # 当前车辆的最大牵引力
        tra_force = self.train_model.max_traction_force * self.action  # 当前输出的牵引力
        self.tm_acc = tra_force / self.train_model.weight

    def get_current_b_acc(self):  # 计算当前制动加速度
        # self.train_model.get_max_brake_force(self.state[1] * 3.6)
        bra_force = self.train_model.max_brake_force * abs(self.action)  # 单位是kN
        self.bm_acc = - bra_force / self.train_model.weight

    def get_m_acc(self):  # 判断当前是制动还是牵引
        self.train_model.get_max_traction_force(self.state[1] * 3.6)  # 当前车辆的最大牵引力
        self.train_model.get_max_brake_force(self.state[1] * 3.6)
        if self.action < 0:
            self.tm_acc = 0
            self.get_current_b_acc()
        else:
            self.get_current_tra_acc()
            self.bm_acc = 0

    def get_acc(self):
        self.get_m_acc()
        self.get_gradient_acc()
        self.get_curve_acc()
        self.acc = self.tm_acc + self.bm_acc + self.g_acc + self.c_acc
        if self.acc > 1.5:
            self.acc = np.array(1.5).reshape(1)
        if self.acc < -1.5:
            self.acc = np.array(-1.5).reshape(1)

    # 下面是能耗的计算过程
    def get_ave_v(self):  # 获取平均速度
        self.ave_v = 0.5 * (self.state[1] + self.next_state[1])

    def get_t_power(self):
        delta_t = self.next_state[0] - self.state[0]
        self.t_power = self.train_model.get_traction_power(self.ave_v * 3.6, delta_t, self.action)

    def get_re_power(self):
        delta_t = self.next_state[0] - self.state[0]
        self.re_power = self.train_model.get_re_power(self.ave_v * 3.6, delta_t, self.action)

    def get_power(self):
        self.get_ave_v()
        if self.action < 0:
            self.t_power = np.array(0.0).reshape(1)
            self.get_re_power()
        else:
            self.re_power = np.array(0.0).reshape(1)
            self.get_t_power()

    # 下面是超速检查过程
    def speed_check(self):
        key_list = []
        key = 0
        for i in self.line.speed_limit.keys():
            key_list.append(i)
        for j in range(len(key_list) - 1):
            if key_list[j] <= self.step * self.line.delta_distance < key_list[j + 1]:
                key = key_list[j]
        limit_speed = self.line.speed_limit[key]
        if self.state[1] * 3.6 >= min(limit_speed, self.ATP_limit):  # 超速
            self.speed_punish = 1
            self.current_limit_speed = min(limit_speed, self.ATP_limit)
        else:
            self.speed_punish = 0
            self.current_limit_speed = min(limit_speed, self.ATP_limit)

    def speed_check2(self):
        key_list = []
        key = 0
        for i in self.line.speed_limit.keys():
            key_list.append(i)
        for j in range(len(key_list) - 1):
            if key_list[j] <= self.step * self.line.delta_distance < key_list[j + 1]:
                key = key_list[j]
        limit_speed = self.line.speed_limit[key]
        self.c_limit_speed = self.line.speed_limit[key]
        self.a_limit_speed = min(self.c_limit_speed, self.ATP_limit)
        if self.state[1] * 3.6 >= limit_speed:  # 超速
            self.speed_punish = 1
        else:
            self.speed_punish = 0

    # 下面是舒适度检查过程
    def comfort_check(self):
        if abs(self.acc - self.last_node_acc) >= 0.3:
            self.comfort_punish = 1.5  # 不舒适
        else:
            self.comfort_punish = 0

    # 下面是状态转移函数
    def get_next_state(self):
        time = np.array(self.state[0]).reshape(1)
        temp_velocity = np.array(self.state[1]).reshape(1)
        temp_square_velocity = temp_velocity * temp_velocity + 2 * self.acc * self.line.delta_distance
        if temp_square_velocity <= 1:
            temp_square_velocity = np.array(1).reshape(1)
        velocity = np.sqrt(temp_square_velocity)
        if velocity <= 0:
            velocity = np.array(1).reshape(1)
        temp_time = self.line.delta_distance / (velocity / 2 + temp_velocity / 2)
        time = time + temp_time
        self.next_state[0] = time  # 状态转移后的时间
        self.next_state[1] = velocity  # 状态转移后的速度

    # 下面是完整的一般状态转移过程
    def state_transition(self):
        self.get_action()
        self.reshape_action()
        self.get_acc()
        self.get_next_state()
        self.get_power()

    # 加入shield后的安全状态转移过程
    def safe_state_transition(self):
        self.get_ATP_limit()
        self.get_action()
        self.reshape_action()
        self.get_acc()
        if self.Shield_Check():
            self.get_safe_action()
            self.reshape_action()
            self.get_acc()
        self.get_next_state()
        self.get_power()

    # 判断是否需要使用Shield
    def Shield_Check(self):
        key_list = []
        key = 0
        for i in self.line.speed_limit.keys():
            key_list.append(i)
        for j in range(len(key_list) - 1):
            if key_list[j] <= self.step * self.line.delta_distance < key_list[j + 1]:
                key = key_list[j]
        next_limit_speed = self.line.speed_limit[key]
        temp_velocity = np.array(self.state[1]).reshape(1)
        temp_square_velocity = temp_velocity * temp_velocity + 2 * self.acc * self.line.delta_distance
        if temp_square_velocity <= 1:
            temp_square_velocity = np.array(1).reshape(1)
        velocity = np.sqrt(temp_square_velocity)
        if velocity <= 0:
            velocity = np.array(1).reshape(1)
        if velocity * 3.6 > min(next_limit_speed, self.ATP_limit):
            return 1
        else:
            return 0

    # 获取安全动作的函数
    def get_safe_action(self):
        xunhuan_count = 0
        chaosu_flag = 0
        initial_velocity = self.state[1].copy()
        while chaosu_flag != 1:
            xunhuan_count += 1
            if xunhuan_count > 30:
                temp_acc = self.acc - self.g_acc - self.c_acc
                if temp_acc >= 0:
                    self.action = temp_acc * self.train_model.weight / self.train_model.max_traction_force
                else:
                    self.action = temp_acc * self.train_model.weight / self.train_model.max_brake_force
                if self.action > 1:
                    self.action = np.array(1).reshape(1)
                if self.action < -1:
                    self.action = np.array(-1).reshape(1)
                break
            # temp_velocity = self.state[1]
            temp_velocity = initial_velocity
            temp_square_velocity = temp_velocity * temp_velocity + 2 * self.acc * self.line.delta_distance
            if temp_square_velocity <= 1:
                temp_square_velocity = np.array(1).reshape(1)
            velocity = np.sqrt(temp_square_velocity)
            if velocity <= 0:
                velocity = np.array(1).reshape(1)
            self.state[1] = velocity
            self.speed_check()
            if self.speed_punish:
                # self.acc = self.acc - 0.2 * (velocity / self.current_limit_speed)
                if xunhuan_count == 1:
                    if self.acc > 0:
                        # self.acc = self.acc - 0.2
                        self.acc = self.acc - 0.2 * (velocity / self.current_limit_speed)
                    else:
                        self.acc = self.acc - 0.2 * (velocity / self.current_limit_speed)
                else:
                    self.acc = self.acc - 0.2 * (velocity / self.current_limit_speed)
            else:
                chaosu_flag = 1
                temp_acc = self.acc - self.g_acc - self.c_acc
                if temp_acc >= 0:
                    self.action = temp_acc * self.train_model.weight / self.train_model.max_traction_force
                else:
                    self.action = temp_acc * self.train_model.weight / self.train_model.max_brake_force
                if self.action > 1:
                    self.action = np.array(1).reshape(1)
                if self.action < -1:
                    self.action = np.array(-1).reshape(1)
        self.state[1] = initial_velocity

    # 下面是奖励的计算
    def get_reward(self, unsafe_counts, total_power):
        self.speed_check2()
        self.comfort_check()
        if self.step == self.max_step:
            done = 1
            if abs(self.next_state[0] - self.line.scheduled_time) <= 10 and abs(self.next_state[1]) <= 3:
                e_reward = 100  # 准时且停下施加额外奖励
            else:
                e_reward = 0
            if self.line.scheduled_time - self.next_state[0] < -10:
                t_punish = -self.next_state[0] + self.line.scheduled_time  # 晚点超过10s的惩罚
            elif self.line.scheduled_time - self.next_state[0] > 0:
                t_punish = -self.line.scheduled_time + self.next_state[0]  # 早到的惩罚
            else:
                t_punish = -self.next_state[0] + self.line.scheduled_time  # 晚点10s之内的惩罚
            if self.speed_punish:
                unsafe_counts += 1
                # self.current_reward = -0.001 * total_power - 25 * self.next_state[1] + 1 * t_punish + e_reward - 10 * self.comfort_punish
                self.current_reward = -3 * (total_power - self.line.ac_power) - 2 * self.next_state[1] + 2 * t_punish + e_reward - 10 * self.comfort_punish
                # self.current_reward = -1 * (total_power - self.line.ac_power) - 25 * self.next_state[1] + 1 * t_punish + e_reward - 10 * self.comfort_punish + self.p_indicator
            else:
                unsafe_counts += 0
                # self.current_reward = -0.001 * total_power - 25 * self.next_state[1] + 1 * t_punish + e_reward - 10 * self.comfort_punish
                self.current_reward = -3 * (total_power - self.line.ac_power) - 2 * self.next_state[1] + 2 * t_punish + e_reward - 10 * self.comfort_punish
                # self.current_reward = -1 * (total_power - self.line.ac_power) - 25 * self.next_state[1] + 1 * t_punish + e_reward - 10 * self.comfort_punish
        else:
            done = 0  # 能耗前的系数影响平化程度，时间项的系数影响整体的曲线形状
            temp_time = self.line.delta_distance / (self.state[1] / 2 + self.next_state[1] / 2)
            if self.speed_punish:
                unsafe_counts += 1
                # self.current_reward = -1.5 * self.t_power - 1.5 * self.re_power - 3.4 * abs(1 * temp_time - (self.line.scheduled_time / (self.max_step + 1))) + self.p_indicator - 10 * self.comfort_punish
                self.current_reward = -4.3 * self.t_power - 4.3 * self.re_power - 15.5 * abs(
                    1 * temp_time - 1 * (abs(self.line.scheduled_time - self.state[0]) / (self.max_step + 1 - self.step))) + self.p_indicator - 10 * self.comfort_punish  # 当前step的运行时间和剩余距离平均时间的差值
                # self.current_reward = -1.5 * self.t_power - 1.5 * self.re_power - abs(1 * (
                #         2 * self.line.delta_distance * (self.max_step + 1 - self.step) / (abs(self.line.scheduled_time - self.state[0])) - self.state[
                #     1])) + self.p_indicator - 10 * self.comfort_punish  # 当前step速度和剩余平均速度的差值
            else:
                unsafe_counts += 0
                # self.current_reward = -1.5 * self.t_power - 1.5 * self.re_power - 3.4 * abs(1 * temp_time - (self.line.scheduled_time / (self.max_step + 1))) - 10 * self.comfort_punish
                self.current_reward = -4.3 * self.t_power - 4.3 * self.re_power - 15.5 * abs(
                    1 * temp_time - 1 * (abs(self.line.scheduled_time - self.state[0]) / (self.max_step + 1 - self.step))) - 10 * self.comfort_punish
                # self.current_reward = -1.5 * self.t_power - 1.5 * self.re_power - abs(1 * (
                #         2 * self.line.delta_distance * (self.max_step + 1 - self.step) / (abs(self.line.scheduled_time - self.state[0])) - self.state[
                #     1])) - 10 * self.comfort_punish
        return done, unsafe_counts

    def get_current_q(self):  # 获取当前Q值
        self.current_q = self.agent.target_critic(self.state, self.action.detach())
