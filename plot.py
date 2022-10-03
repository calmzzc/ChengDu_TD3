#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-10-07 20:57:11
LastEditor: John
LastEditTime: 2021-09-23 12:23:01
Discription:
Environment:
'''
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import numpy as np


def chinese_font():
    return FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)  # 系统字体路径，此处是mac的


def plot_rewards(rewards, ma_rewards, tag="train", env='CartPole-v0', algo="DQN", save=True, path='./'):
    # sns.set()
    plt.title("average learning curve of {} for {}".format(algo, env))
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(ma_rewards, label='ma rewards')
    plt.legend()
    if save:
        plt.savefig(path + "{}_rewards_curve".format(tag))
    plt.show()


def plot_rewards_cn(rewards, ma_rewards, tag="train", env='Optimal Control', algo="DDPG", save=True, path='./'):
    ''' 中文画图
    '''
    # sns.set()
    plt.figure()
    plt.title(u"{}环境下{}算法的奖励曲线".format(env, algo), fontproperties=chinese_font())
    plt.xlabel(u'回合数', fontproperties=chinese_font())
    ma_std = np.std(ma_rewards, ddof=1)
    x = np.linspace(0, len(ma_rewards), len(ma_rewards))
    # for i in range(len(ma_rewards)):
    #     ma_rewards[i] = ma_rewards[i].reshape(-1)
    ma_rewards_pstd = [i + ma_std for i in ma_rewards]  # 加方差
    ma_rewards_pstd = np.array(ma_rewards_pstd)
    ma_rewards_pstd = ma_rewards_pstd.reshape(1, ma_rewards_pstd.shape[0])[0]
    ma_rewards_mstd = [i - ma_std for i in ma_rewards]  # 减方差
    ma_rewards_mstd = np.array(ma_rewards_mstd)
    ma_rewards_mstd = ma_rewards_mstd.reshape(1, ma_rewards_mstd.shape[0])[0]
    # plt.plot(rewards)
    plt.plot(ma_rewards)
    plt.fill_between(x, ma_rewards_pstd, ma_rewards_mstd, facecolor='blue', alpha=0.3)
    plt.legend((u'滑动平均奖励', u'奖励',), loc="best", prop=chinese_font())
    if save:
        plt.savefig(path + f"{tag}_rewards_curve_cn")
    # plt.show()


def plot_power_cn(power, ma_power, tag="train", env='Optimal Control', algo="DDPG", save=True, path='./'):
    ''' 中文画图
    '''
    # sns.set()
    plt.figure()
    plt.title(u"{}环境下{}算法的能耗曲线".format(env, algo), fontproperties=chinese_font())
    plt.xlabel(u'回合数', fontproperties=chinese_font())
    ma_std = np.std(ma_power, ddof=1)
    x = np.linspace(0, len(ma_power), len(ma_power))
    ma_power_pstd = [i + ma_std for i in ma_power]  # 加方差
    ma_power_pstd = np.array(ma_power_pstd)
    ma_power_pstd = ma_power_pstd.reshape(1, ma_power_pstd.shape[0])[0]
    ma_power_mstd = [i - ma_std for i in ma_power]  # 减方差
    ma_power_mstd = np.array(ma_power_mstd)
    ma_power_mstd = ma_power_mstd.reshape(1, ma_power_mstd.shape[0])[0]
    plt.plot(power)
    plt.plot(ma_power)
    plt.legend((u'滑动平均能耗', u'能耗',), loc="best", prop=chinese_font())
    plt.fill_between(x, ma_power_pstd, ma_power_mstd, facecolor='blue', alpha=0.3)
    if save:
        plt.savefig(path + f"{tag}_power_curve_cn")
    # plt.show()


def plot_unsafecounts_cn(unsafe_counts, ma_unsafe_counts, tag="train", env='Optimal Control', algo="DDPG", save=True,
                         path='./'):
    ''' 中文画图
    '''
    # sns.set()
    plt.figure()
    plt.title(u"{}环境下{}算法的不安全动作次数曲线".format(env, algo), fontproperties=chinese_font())
    plt.xlabel(u'回合数', fontproperties=chinese_font())
    # ma_std = np.std(ma_unsafe_counts, ddof=1)
    # x = np.linspace(0, len(ma_unsafe_counts), len(ma_unsafe_counts))
    # ma_unsafe_counts_pstd = [i + ma_std for i in ma_unsafe_counts]  # 加方差
    # ma_unsafe_counts_pstd = np.array(ma_unsafe_counts_pstd)
    # ma_unsafe_counts_pstd = ma_unsafe_counts_pstd.reshape(1, ma_unsafe_counts_pstd.shape[0])[0]
    # ma_unsafe_counts_mstd = [i - ma_std for i in ma_unsafe_counts]  # 减方差
    # ma_unsafe_counts_mstd = np.array(ma_unsafe_counts_mstd)
    # ma_unsafe_counts_mstd = ma_unsafe_counts_mstd.reshape(1, ma_unsafe_counts_mstd.shape[0])[0]
    plt.plot(unsafe_counts)
    plt.plot(ma_unsafe_counts)
    plt.legend((u'滑动平均不安全次数', u'不安全次数',), loc="best", prop=chinese_font())
    # plt.fill_between(x, ma_unsafe_counts_pstd, 0, facecolor='blue', alpha=0.3)
    if save:
        plt.savefig(path + f"{tag}_unsafe counts_curve_cn")
    # plt.show()


def plot_losses(losses, algo="DQN", save=True, path='./'):
    # sns.set()
    plt.title("loss curve of {}".format(algo))
    plt.xlabel('epsiodes')
    plt.plot(losses, label='rewards')
    plt.legend()
    if save:
        plt.savefig(path + "losses_curve")
    plt.show()


def plot_speed(total_v_list, total_t_list, total_a_list, total_acc_list, tag="train", env='Train Optimal', algo="DDPG", save=True,
               path='./'):
    # sns.set()
    plt.figure()
    plt.title(u"{}环境下{}算法的训练速度曲线".format(env, algo), fontproperties=chinese_font())
    ax1 = plt.axes(projection='3d')
    for i in range(len(total_v_list)):
        if i % 50 == 0:
            a = np.array(total_v_list[i]).reshape(-1)
            b = np.array(total_t_list[i]).reshape(-1)
            c = np.linspace(1, len(total_t_list[i]), len(total_t_list[i]))
            ax1.plot3D(b, c, a)
    plt.legend((u'速度曲线',), loc="best", prop=chinese_font())
    if save:
        plt.savefig(path + f"{tag}_speed_profile_cn")
    plt.figure()
    for i in range(len(total_a_list)):
        if i % 50 == 0:
            plt.plot(total_a_list[i])
    plt.legend((u'动作曲线',), loc='best', prop=chinese_font())
    if save:
        plt.savefig(path + f"{tag}_action_cn")
    for i in range(len(total_acc_list)):
        if i % 50 == 0:
            plt.plot(total_acc_list[i])
    plt.legend((u'加速度曲线',), loc='best', prop=chinese_font())
    if save:
        plt.savefig(path + f"{tag}_acc_cn")
    plt.show()


def evalplot_speed(total_v_list, total_t_list, total_a_list, total_acc_list, limit_list, A_limit_list, tag="eval", env='Train Optimal', algo="DDPG", save=True,
                   path='./'):
    # sns.set()
    plt.figure()
    plt.title(u"{}环境下{}算法的评价速度曲线".format(env, algo), fontproperties=chinese_font())
    ax1 = plt.axes(projection='3d')
    for i in range(len(total_v_list)):
        if i % 6 == 0:
            a = np.array(total_v_list[i]).reshape(-1)
            b = np.array(total_t_list[i]).reshape(-1)
            c = np.linspace(1, len(total_t_list[i]) * 40, len(total_t_list[i]))
            ax1.plot3D(b, c, a)
    plt.legend((u'速度曲线',), loc="best", prop=chinese_font())
    if save:
        plt.savefig(path + f"{tag}_speed_profile_cn")
    plt.figure()
    plt.plot(total_a_list[1])
    plt.legend((u'动作曲线',), loc='best', prop=chinese_font())
    if save:
        plt.savefig(path + f"{tag}_action_cn")
    plt.figure()
    plt.plot(total_acc_list[1])
    plt.legend((u'加速度曲线',), loc='best', prop=chinese_font())
    if save:
        plt.savefig(path + f"{tag}_acc_cn")
    plt.figure()
    plt.plot(np.linspace(1, len(total_v_list[1]) * 40, len(total_v_list[1])), total_v_list[1])
    plt.plot(np.linspace(1, len(total_v_list[1]) * 40, len(total_v_list[1])), limit_list)
    plt.plot(np.linspace(1, len(total_v_list[1]) * 40, len(total_v_list[1])), A_limit_list)
    plt.legend((u'速度曲线', u'静态限速曲线', u'防护曲线'), loc='best', prop=chinese_font())
    if save:
        plt.savefig(path + f"{tag}_v_cn")
    plt.show()


def plot_trainep_speed(total_v_list, total_t_list, total_a_list, total_ep_list, total_acc_list, tag="ep_train", env='Train Optimal',
                       algo="DDPG", save=True,
                       path='./'):
    # sns.set()
    plt.figure()
    plt.title(u"{}环境下{}算法的训练速度曲线（分幕）".format(env, algo), fontproperties=chinese_font())
    ax1 = plt.axes(projection='3d')
    for i in range(len(total_ep_list)):
        if i % 50 == 0:
            a = np.array(total_v_list[i]).reshape(-1)
            b = np.array(total_t_list[i]).reshape(-1)
            c = np.linspace(total_ep_list[i], total_ep_list[i], len(total_t_list[i]))
            ax1.plot3D(b, c, a)
    plt.legend((u'速度曲线',), loc="best", prop=chinese_font())
    if save:
        plt.savefig(path + f"{tag}_speed_profile_cn")
    plt.figure()
    ax2 = plt.axes(projection='3d')
    for j in range(len(total_ep_list)):
        if j % 50 == 0:
            a1 = np.array(total_a_list[j]).reshape(-1)
            b1 = np.array(total_t_list[j]).reshape(-1)
            c1 = np.linspace(total_ep_list[j], total_ep_list[j], len(total_t_list[j]))
            ax2.plot3D(b1, c1, a1)
    plt.legend((u'动作曲线',), loc='best', prop=chinese_font())
    if save:
        plt.savefig(path + f"{tag}_action_cn")
    plt.figure()
    ax3 = plt.axes(projection='3d')
    for j in range(len(total_ep_list)):
        if j % 50 == 0:
            a1 = np.array(total_acc_list[j]).reshape(-1)
            b1 = np.array(total_t_list[j]).reshape(-1)
            c1 = np.linspace(total_ep_list[j], total_ep_list[j], len(total_t_list[j]))
            ax3.plot3D(b1, c1, a1)
    plt.legend((u'加速度曲线',), loc='best', prop=chinese_font())
    if save:
        plt.savefig(path + f"{tag}_acc_cn")
    plt.show()


def plot_evalep_speed(total_v_list, total_t_list, total_a_list, total_ep_list, total_acc_list, tag="ep_eval",
                      env='Train Optimal',
                      algo="DDPG", save=True,
                      path='./'):
    # sns.set()
    plt.figure()
    # plt.title(u"{}环境下{}算法的评价速度曲线（分幕）".format(env, algo), fontproperties=chinese_font())
    ax1 = plt.axes(projection='3d')
    for i in range(len(total_ep_list)):
        if i % 6 == 0:
            a = np.array(total_v_list[i]).reshape(-1)
            b = np.array(total_t_list[i]).reshape(-1)
            c = np.linspace(total_ep_list[i], total_ep_list[i], num=len(total_t_list[i]))
            ax1.plot3D(b, c, a)
    plt.legend((u'速度曲线',), loc="best", prop=chinese_font())
    if save:
        plt.savefig(path + f"{tag}_speed_profile_cn")
    plt.figure()
    ax2 = plt.axes(projection='3d')
    for j in range(len(total_ep_list)):
        if j % 6 == 0:
            a1 = np.array(total_a_list[j]).reshape(-1)
            b1 = np.array(total_t_list[j]).reshape(-1)
            c1 = np.linspace(total_ep_list[j], total_ep_list[j], num=len(total_t_list[j]))
            ax2.plot3D(b1, c1, a1)
    plt.legend((u'动作曲线',), loc='best', prop=chinese_font())
    if save:
        plt.savefig(path + f"{tag}_action_cn")
    plt.figure()
    ax3 = plt.axes(projection='3d')
    for j in range(len(total_ep_list)):
        if j % 6 == 0:
            a1 = np.array(total_acc_list[j]).reshape(-1)
            b1 = np.array(total_t_list[j]).reshape(-1)
            c1 = np.linspace(total_ep_list[j], total_ep_list[j], num=len(total_t_list[j]))
            ax3.plot3D(b1, c1, a1)
    plt.legend((u'加速度曲线',), loc='best', prop=chinese_font())
    if save:
        plt.savefig(path + f"{tag}_acc_cn")
    plt.show()
