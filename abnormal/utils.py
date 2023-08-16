import numpy as np
import math


def get_angle(center_coor, coor2, coor3):
    L1 = cal_dis(coor2,coor3)
    L2 = cal_dis(center_coor,coor3)
    L3 = cal_dis(center_coor,coor2)
    Angle = cal_angle(L1,L2,L3)
    return Angle


def cal_dis(coor1, coor2):
    out = np.square(coor1[0] - coor2[0]) + np.square(coor1[1] - coor2[1])
    return np.sqrt(out)


def cal_angle(L1, L2, L3):
    out = (np.square(L2) + np.square(L3) - np.square(L1)) / (2 * L2 * L3)
    try:
        return math.acos(out) * (180 / math.pi)
    except ValueError:
        return 180


if __name__ == '__main__':
    coord_0, coord_1, coord_2 = ((0,0), (1,0), (0,1))
    print(get_angle(coord_0, coord_1, coord_2))
    coord_3, coord_4 = ((1,1), (3,3))
    print(cal_dis(coord_3, coord_4))
