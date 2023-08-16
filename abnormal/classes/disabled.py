import numpy as np
import abnormal.utils as utils


class DisabledHandler:
    def __init__(self):
        self.hw_threshold = 0.78

    def process(self, ids, boxes, kps, kps_scores):
        #return np.zeros(len(boxes))
        ResultVec = np.zeros(len(boxes))
        for index, bbox in enumerate(boxes):
            angle_lbody = utils.get_angle(kps[index][11], kps[index][0], kps[index][15]) # hip nose ankle
            angle_rbody = utils.get_angle(kps[index][12], kps[index][0], kps[index][16])

            # print('angle_lbody = {}'.format(angle_lbody))
            # print('angle_rbody = {}'.format(angle_rbody))

            # if (angle_lbody < 140 or angle_rbody < 140) and ((bbox[2] - bbox[0]) / (bbox[3] - bbox[1]) < self.hw_threshold):
            #     ResultVec[index] = 1
            # else:
            #     ResultVec[index] = 0

        return ResultVec
    
def TestFunction():
    print("hello, world")

