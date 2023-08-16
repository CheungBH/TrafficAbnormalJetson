import numpy as np
from collections import defaultdict


class ReverseHandler:
    def __init__(self):
        self.boxes_central = defaultdict(list)
        self.boxes_move = defaultdict(list)
        self.direction = "right"
        self.max_time = 10
        self.max_time2 = 20
        self.valid_thresh = 0.6  # For the valid ratio of small changing with max time
        self.moving_thresh = 0.02  # For ratio of distance and box width
        self.fluctuate_thresh = 0.15  # For positive number ratio

    def process(self, ids, boxes, kps, kps_scores):
        boxes = boxes.tolist()
        ids = ids.tolist()
        for i, box in zip(ids, boxes):
            self.boxes_central[i].append([(box[0]+box[2])/2, (box[1]+box[3])/2])
            if len(self.boxes_central[i]) < 2:
                pass
            else:
                self.boxes_move[i].append(self.point_move_calculate(self.boxes_central[i][-1], self.boxes_central[i][-2]))
        status = np.zeros(len(boxes))
        for i in range(len(ids)):
            status[i] = self.judge_reverse(self.boxes_move[ids[i]], boxes[i])
        return status

    def point_move_calculate(self, p1, p2):
        _x = p1[0] - p2[0]
        _y = p1[1] - p2[1]
        return [_x, _y]

    def judge_reverse(self, moves, box,):    # 0是正常行进，1是出现逆行
        global pos_num
        if len(moves) < self.max_time:
            return 0
        count = []
        # moves_minus = []
        # moves_plus = []
        for i in range(self.max_time):
            count.append(moves[-i-1][0])
        # if sum([abs(item)/(box[2]-box[0]) < self.moving_thresh for item in count])/self.max_time > self.valid_thresh:
        #     if moves[-i-1][0] < 0:
        #         moves_minus.append(moves[-i-1][0])
        #     if moves[-i-1][0] > 0:
        #         moves_plus.append(moves[-i-1][0])
        #     if len(moves_minus)/self.max_time < 0.75:
        #         if len(moves_minus)/self.max_time > 0.25:
        #            if len(moves_plus)/self.max_time < 0.75:
        #                if len(moves_plus)/self.max_time > 0.25:
        #                     return 0

        #valid_tmp = []



        #if sum([abs(item)/(box[2]-box[0]) < self.moving_thresh for item in count])/self.max_time > self.valid_thresh:   # 最大容忍度里单次行进小于移动阈值的次数大于阈值

        valid_tmp = []
        for item in count:
            if abs(item)/(box[2]-box[0]) < self.moving_thresh:
                valid_tmp.append(1)
        if sum(valid_tmp)/self.max_time > self.valid_thresh:
            pos_num = sum([c > 0 for c in count])    # 计算正向移动的次数
            """
            pos_tmp = []
            for c in count:
                if c > 0:
                    pos_tmp.append(1)
            pos_num = sum(pos_tmp)
            """
            """
            if self.fluctuate_thresh < pos_num / self.max_time < 1 - self.fluctuate_thresh:
            """

            if self.fluctuate_thresh < pos_num / self.max_time < 1 - self.fluctuate_thresh:
                return 0  # 0是normal，有波动才是正常行进，非逆行行为
            else:
                pass

        count2 = []
        count_sum = sum(count)
        if self.direction == "right":
            if count_sum > 0:
                return 0
            else:
                if sum(valid_tmp) == self.max_time:
                    return 0
                else:
                    for i in range(self.max_time2):
                        count2.append(moves[-i - 1][0])
                        count_sum2 = sum(count2)
                        if count_sum2 > -6:
                            return 0
                        else:
                            return 1

        elif self.direction == "left":
            return 1 if count_sum > 0 else 0
        else:
            raise ValueError


