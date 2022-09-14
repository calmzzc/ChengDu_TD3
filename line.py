import numpy as np


class Section1:
    def __init__(self):
        self.start_station = '''JiTouQiao'''  # 出发站
        self.end_station = '''BaiFuoQiao'''  # 到达站
        self.length = 1470  # 站间长度
        self.delta_distance = 49  # 位置离散
        # self.max_iteration = self.length / self.delta_distance
        self.scheduled_time = 86  # 计划运行时间
        self.speed_limit = {  # 线路限速
            0: 80, 240: 120, 1025: 80, 1470: 0
        }
        self.gradient = {  # 坡度
            0: 0, 209: -22, 459: -5, 709: 5.47, 959: 26, 1234: 0
        }
        self.curve = {0: 0, 850: 2500, 1065: 0, 1107: 1500, 1303: 0}  # 曲率
        self.direction = "ShangXing"  # 运行方向
        self.tra_power = 51.56  # 牵引能耗
        self.re_power = 25.76  # 再生制动产生能量
        self.ac_power = 25.80  # 实际能耗


class Section2:
    def __init__(self):
        self.start_station = '''BaiFuoQiao'''
        self.end_station = '''JiuJiangBei'''
        self.length = 4180
        self.delta_distance = 76
        # self.max_iteration = self.length / self.delta_distance
        self.scheduled_time = 183.15
        self.speed_limit = {
            0: 80, 245: 120, 1468: 100, 2391: 120, 3709: 80, 4180: 0
        }
        self.gradient = {
            0: 0, 216: -12.9, 1341: 16, 1741: 5, 2326: 15, 3091: 9, 3541: -12.5, 3841: 0
        }
        self.curve = {0: 0, 635: 3000, 848: 0, 937: 1500, 1464: 0, 1604: 650, 1887: 0, 1926: 600, 2229: 0, 2775: 3000,
                      3038: 0}
        self.direction = "ShangXing"
        self.tra_power = 115.86
        self.re_power = 32.66
        self.ac_power = 83.20


class Section3:
    def __init__(self):
        self.start_station = '''JiuJiangBei'''
        self.end_station = '''MingGuang'''
        self.length = 6640
        self.delta_distance = 40
        # self.max_iteration = self.length / self.delta_distance
        self.scheduled_time = 275.78
        self.speed_limit = {
            0: 45, 334: 110, 1088: 100, 3306: 140, 6197: 80, 6640: 0
        }
        self.gradient = {
            0: 0, 412: -24.58, 862: -10.8, 1842: 28.5, 2422: 11, 3262: -7.84, 4012: 9, 4612: 3.5, 6012: 9.74, 6397: 0
        }
        self.curve = {0: 0, 274: 3500, 440: 0, 491: 3500, 641: 0, 1449: 600, 1874: 0, 2150: 600, 3170: 0, 3361: 3500,
                      3567: 0, 3638: 3500, 3844: 0, 4004: 2500, 4298: 0}
        self.direction = "ShangXing"
        self.tra_power = 159.38
        self.re_power = 58.78
        self.ac_power = 100.6


class Section4:
    def __init__(self):
        self.start_station = '''MingGuang'''
        self.end_station = '''WenQuanDaDao'''
        self.length = 1980
        self.delta_distance = 36
        # self.max_iteration = self.length / self.delta_distance
        self.scheduled_time = 104.144
        self.speed_limit = {
            0: 80, 197: 110, 1509: 80, 1980: 0
        }
        self.gradient = {
            0: 0, 171: -20, 421: -3.5, 921: 10, 1471: 22, 1801: 0
        }
        self.curve = {0: 0, 1449: 5000, 1610: 0, 1721: 1200, 1876: 0}
        self.direction = "ShangXing"
        self.tra_power = 56.22
        self.re_power = 22.62
        self.ac_power = 33.6


class Section5:
    def __init__(self):
        self.start_station = '''WenQuanDaDao'''
        self.end_station = '''FengXiHe'''
        self.length = 2040
        self.delta_distance = 40
        # self.max_iteration = self.length / self.delta_distance
        self.scheduled_time = 104.177
        self.speed_limit = {
            0: 80, 206: 120, 1583: 80, 2040: 0
        }
        self.gradient = {
            0: 0, 247: -2, 527: -3, 1912: 0
        }
        self.curve = {0: 0, 1250: 6000, 1369: 0, 1758: 6000, 1879: 0}
        self.direction = "ShangXing"
        self.tra_power = 68.14
        self.re_power = 40.68
        self.ac_power = 27.46


class Section6:
    def __init__(self):
        self.start_station = '''FengXiHe'''
        self.end_station = '''ShiWuYiYuan'''
        self.length = 1850
        self.delta_distance = 37
        # self.max_iteration = self.length / self.delta_distance
        self.scheduled_time = 105.129
        self.speed_limit = {
            0: 80, 197: 100, 1120: 70, 1850: 0
        }
        self.gradient = {
            0: 0, 144: 10, 1689: 0
        }
        self.curve = {0: 0, 235: 800, 631: 0, 873: 3000, 1155: 0, 1462: 450, 1779: 0}
        self.direction = "ShangXing"
        self.tra_power = 63
        self.re_power = 21.96
        self.ac_power = 41.04


class Section7:
    def __init__(self):
        self.start_station = '''ShiWuYiYuan'''
        self.end_station = '''HuangShi'''
        self.length = 2960
        self.delta_distance = 37
        # self.max_iteration = self.length / self.delta_distance
        self.scheduled_time = 105.129
        self.speed_limit = {
            0: 95, 925: 120, 2520: 80, 2960: 0
        }
        self.gradient = {
            0: 0, 248: -25, 648: 17, 1508: 3.25, 2808: 29, 2788: 0
        }
        self.curve = {0: 0, 100: 600, 382: 0, 474: 600, 852: 0, 1012: 1200, 1651: 0, 1847: 3500, 2042: 0, 2650: 5000,
                      2723: 0, 2763: 5000, 2838: 0}
        self.direction = "ShangXing"
        self.tra_power = 101.8
        self.re_power = 28.7
        self.ac_power = 73.1


class Section8:
    def __init__(self):
        self.start_station = '''HuangShi'''
        self.end_station = '''JinXing'''
        self.length = 4280
        self.delta_distance = 40
        # self.max_iteration = self.length / self.delta_distance
        self.scheduled_time = 172.1
        self.speed_limit = {
            0: 100, 760: 140, 3832: 80, 4280: 0
        }
        self.gradient = {
            0: 0, 255: 5.9, 2095: 0, 2995: 8.97, 3945: 0
        }
        self.curve = {0: 0, 287: 600, 606: 0, 818: 1004, 1146: 0, 2565: 5004, 2705: 0, 4027: 1000, 4149: 0}
        self.direction = "ShangXing"
        self.tra_power = 119.62
        self.re_power = 46.62
        self.ac_power = 73


class Section9:
    def __init__(self):
        self.start_station = '''BaiFuoQiao'''  # 出发站
        self.end_station = '''JiTouQiao'''  # 到达站
        self.length = 1470  # 站间长度
        self.delta_distance = 49  # 位置离散
        # self.max_iteration = self.length / self.delta_distance
        self.scheduled_time = 91.747  # 计划运行时间
        self.speed_limit = {  # 线路限速
            0: 80, 213: 120, 1009: 80, 1470: 0
        }
        self.gradient = {  # 坡度
            0: 0, 234: -26, 509: -5.47, 759: 5, 1009: 22, 1259: 0
        }
        self.curve = {0: 0, 165: 1500, 361: 0, 403: 2500, 618: 0}  # 曲率
        self.direction = "XiaXing"  # 运行方向
        self.tra_power = 45.38  # 牵引能耗
        self.re_power = 22.86  # 再生制动产生能量
        self.ac_power = 22.52  # 实际能耗


class Section10:
    def __init__(self):
        self.start_station = '''JiuJiangBei'''
        self.end_station = '''BaiFuoQiao'''
        self.length = 4180
        self.delta_distance = 38
        # self.max_iteration = self.length / self.delta_distance
        self.scheduled_time = 204.47
        self.speed_limit = {
            0: 45, 269: 85, 981: 105, 1476: 95, 2372: 120, 3727: 80, 4180: 0
        }
        self.gradient = {
            0: 0, 338: 12.5, 638: -9, 1088: -15, 1853: -5, 2438: -16, 2838: 12.9, 3963: 0
        }
        self.curve = {0: 0, 1141: 3000, 1404: 0, 1950: 600, 2253: 0, 2292: 650, 2575: 0, 2715: 1500, 3242: 0,
                      3331: 3000,
                      3544: 0}
        self.direction = "XiaXing"
        self.tra_power = 87.92
        self.re_power = 40.08
        self.ac_power = 47.84


class Section11:
    def __init__(self):
        self.start_station = '''MingGuang'''
        self.end_station = '''JiuJiangBei'''
        self.length = 6640
        self.delta_distance = 40
        # self.max_iteration = self.length / self.delta_distance
        self.scheduled_time = 258.642
        self.speed_limit = {
            0: 80, 220: 140, 3014: 100, 4232: 95, 5098: 120, 6187: 80, 6640: 0
        }
        self.gradient = {
            0: 0, 244: -9.74, 629: -3.5, 2029: -9.0, 2629: 7.84, 3379: -11.0, 4219: -28.5, 4799: 10.8, 5779: 24.58,
            6229: 0
        }
        self.curve = {0: 0, 2343: 2500, 2637: 0, 2797: 3500, 3003: 0, 3074: 3500, 3280: 0, 3471: 600, 4491: 0,
                      4767: 600,
                      5192: 0, 6000: 3500, 6150: 0, 6201: 3500, 6367: 0}
        self.direction = "XiaXing"
        self.tra_power = 142.4
        self.re_power = 61.76
        self.ac_power = 80.64


class Section12:
    def __init__(self):
        self.start_station = '''WenQuanDaDao'''
        self.end_station = '''MingGuang'''
        self.length = 1980
        self.delta_distance = 36
        # self.max_iteration = self.length / self.delta_distance
        self.scheduled_time = 101.081
        self.speed_limit = {
            0: 80, 244: 120, 1530: 80, 1980: 0
        }
        self.gradient = {
            0: 0, 173: -22, 503: -10, 1053: 3.5, 1553: 20, 1803: 0
        }
        self.curve = {0: 0, 98: 1200, 253: 0, 364: 5000, 525: 0}
        self.direction = "XiaXing"
        self.tra_power = 61.04
        self.re_power = 16.88
        self.ac_power = 44.16


class Section13:
    def __init__(self):
        self.start_station = '''FengXiHe'''
        self.end_station = '''WenQuanDaDao'''
        self.length = 2040
        self.delta_distance = 40
        # self.max_iteration = self.length / self.delta_distance
        self.scheduled_time = 103.309
        self.speed_limit = {
            0: 80, 235: 120, 1597: 80, 2040: 0
        }
        self.gradient = {
            0: 0, 126: 3, 1511: 2, 1791: 0
        }
        self.curve = {0: 0, 159: 6000, 280: 0, 669: 6000, 788: 0}
        self.direction = "XiaXing"
        self.tra_power = 67.26
        self.re_power = 32.1
        self.ac_power = 35.16


class Section14:
    def __init__(self):
        self.start_station = '''ShiWuYiYuan'''
        self.end_station = '''FengXiHe'''
        self.length = 1850
        self.delta_distance = 37
        # self.max_iteration = self.length / self.delta_distance
        self.scheduled_time = 106.156
        self.speed_limit = {
            0: 70, 431: 100, 1445: 80, 1850: 0
        }
        self.gradient = {
            0: 0, 162: -10, 1707: 0
        }
        self.curve = {0: 0, 72: 450, 389: 0, 696: 3000, 978: 0, 1220: 800, 1616: 0}
        self.direction = "XiaXing"
        self.tra_power = 48.38
        self.re_power = 34.16
        self.ac_power = 14.22


class Section15:
    def __init__(self):
        self.start_station = '''HuangShi'''
        self.end_station = '''ShiWuYiYuan'''
        self.length = 2960
        self.delta_distance = 37
        # self.max_iteration = self.length / self.delta_distance
        self.scheduled_time = 137.973
        self.speed_limit = {
            0: 100, 337: 120, 1721: 95, 2960: 0
        }
        self.gradient = {
            0: 0, 175: -29, 1155: -3.25, 1455: -17, 2315: 25, 2715: 0
        }
        self.curve = {0: 0, 125: 5000, 200: 0, 240: 5000, 313: 0, 921: 3500, 1116: 0, 1312: 1200, 1621: 0, 1673: 1200,
                      1951: 0, 2111: 600, 2489: 0, 2581: 600, 2863: 0}
        self.direction = "XiaXing"
        self.tra_power = 55.34
        self.re_power = 45.04
        self.ac_power = 10.3


class Section16:
    def __init__(self):
        self.start_station = '''JinXing'''
        self.end_station = '''HuangShi'''
        self.length = 4280
        self.delta_distance = 40
        # self.max_iteration = self.length / self.delta_distance
        self.scheduled_time = 171.15
        self.speed_limit = {
            0: 100, 547: 140, 2958: 115, 3298: 100, 4280: 0
        }
        self.gradient = {
            0: 0, 335: -8.97, 1285: 0, 2185: -5.9, 4025: 0
        }
        self.curve = {0: 0, 131: 1000, 253: 0, 1575: 5004, 1715: 0, 3134: 1004, 3462: 0, 3674: 600, 3993: 0}
        self.direction = "XiaXing"
        self.tra_power = 92.94
        self.re_power = 58.64
        self.ac_power = 34.3


Section = {"Section1": Section1(), "Section2": Section2(), "Section3": Section3(), "Section4": Section4(),
           "Section5": Section5(), "Section6": Section6(), "Section7": Section7(), "Section8": Section8()}
Section2 = {"Section9": Section9(), "Section10": Section10(), "Section11": Section11(), "Section12": Section12(),
            "Section13": Section13(), "Section14": Section14(), "Section15": Section15(),
            "Section16": Section16()}
