from abnormal.classes import *
# import cv2


class AbnormalHandler:
    def __init__(self):
        self.abnormal = [WheelchairHandler(), FallHandler(), DisabledHandler(), ReverseHandler()]
        self.abnormal_name = ["Wheelchair", "Fall", "Disabled", "Reverse"]

    def process(self, ids, boxes, kps, kps_scores):
        inp = {"id": ids, "boxes": boxes, "kps": kps, "kps_scores": kps_scores}
        self.reset(inp)
        for idx, abnormal in enumerate(self.abnormal):
            if idx > 1:
                continue
            self.status[idx] = abnormal.process(ids, boxes, kps, kps_scores)

    def reset(self, inputs):
        self.id = inputs["id"]
        self.boxes = inputs["boxes"]
        self.kps = inputs["kps"]
        self.status = np.zeros((len(self.abnormal), len(self.boxes)))

    # def visualize(self, img):
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     check_alarm = self.status.sum(1) > 0
    #     alarm_word = self.alarm_wording(check_alarm)
    #     alarm_color = (3, 255, 255) if check_alarm.sum() else (0, 255, 0)
    #     cv2.putText(img, alarm_word, (0, 70), font, 3, alarm_color, 3, cv2.LINE_AA)

    def alarm_wording(self, check_alarm):
        if check_alarm.sum() == 0:
            return "Normal"
        s = ""
        for alarm, name in zip(check_alarm, self.abnormal_name):
            if alarm:
                s += "{} + ".format(name)
        return s[:-2]



