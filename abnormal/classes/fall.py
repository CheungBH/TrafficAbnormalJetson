import numpy as np
import abnormal.utils as utils


class FallHandler:
    def __init__(self):
        self.hw_threshold = 0.78

    def process(self, ids, boxes, kps, kps_scores):
        #return np.zeros(len(boxes))
        ResultVec = np.zeros(len(boxes))
        for index, bbox in enumerate(boxes):

            h_point_l = kps[index][15]
            h_point_r = kps[index][16]

            angle_lfall = utils.get_angle(kps[index][15], kps[index][0], (0, h_point_l[1]))
            angle_rfall = utils.get_angle(kps[index][16], kps[index][0], (0, h_point_r[1]))

            if angle_lfall > 90:
                angle_lfall = 180 - angle_lfall
            if angle_rfall > 90:
                angle_rfall = 180 - angle_rfall

            bb_ratio = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])

            # if (angle_lfall < 50 or angle_rfall < 50) and (bb_ratio > 1):
            #     ResultVec[index] = 1
            # else:
            #     ResultVec[index] = 0

        return ResultVec
