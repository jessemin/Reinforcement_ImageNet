import cv2
import os
from action import Action
from bbdata import BoundingBoxData
from util import Util


class Env:
    def __init__(self, wnid, image_id, alpha=0.2):
        self.wnid = wnid
        self.current_image_id = str(image_id[image_id.find("_")+1:])
        self.image = cv2.imread(os.path.join("Images", wnid, wnid+"_"+self.current_image_id+".JPEG"))
        self.h, self.w = [_*1.0 for _ in self.image.shape[:2]]
        # bb format: x1, y1, x2, y2
        self.current_bb = (0.0, 0.0, self.w, self.h)
        self.alpha = alpha
        self.correct_bb = tuple([_*1.0 for _ in BoundingBoxData(str(wnid), self.current_image_id).bounding_boxes[0]])
        self.util = Util()

    def apply_action(self, action_type):
        x_1, y_1, x_2, y_2 = self.current_bb
        alpha_w, alpha_h = self.alpha * (x_2 - x_1), self.alpha * (y_2 - y_1)
        if action_type == 0:
            self.current_bb = (x_1+alpha_w, y_1, x_2+alpha_w, y_2)
        if action_type == 1:
            self.current_bb = (x_1-alpha_w, y_1, x_2-alpha_w, y_2)
        if action_type == 2:
            self.current_bb = (x_1, y_1+alpha_h, x_2, y_2+alpha_h)
        if action_type == 3:
            self.current_bb = (x_1, y_1-alpha_h, x_2, y_2-alpha_h)
        if action_type == 4:
            self.current_bb = (x_1-alpha_w, y_1-alpha_h, x_2+alpha_w, y_2+alpha_h)
        if action_type == 5:
            self.current_bb = (x_1+alpha_w, y_1+alpha_h, x_2-alpha_w, y_2-alpha_h)
        if action_type == 6:
            self.current_bb = (x_1, y_1+alpha_h, x_2, y_2-alpha_h)
        if action_type == 7:
            self.current_bb = (x_1+alpha_w, y_1, x_2-alpha_w, y_2)
        if action_type == 8:
            return True
        x_1, y_1, x_2, y_2 = self.current_bb
        if x_1 < 0:
            x_1 = 0.0
        if x_2 > self.w:
            x_2 = self.w
        if y_1 < 0:
            y_1 = 0.0
        if y_2 > self.h:
            y_2 = self.h
        if x_1 > x_2:
            x_1 = x_2 - 10.0
        if y_1 > y_2:
            y_1 = y_2 - 10.0
        self.current_bb = (x_1, y_1, x_2, y_2)
        return False

    # return new bb, r, done
    def step(self, action):
        action_type = action.get_action_type()
        iou_1 = self.util.computeIOU(self.current_bb, self.correct_bb)
        done = self.apply_action(action_type)
        iou_2 = self.util.computeIOU(self.current_bb, self.correct_bb)
        reward = 1.0 if iou_2-iou_1 > 0 else -1.0
        return self.current_bb, reward, done


if __name__=='__main__':
    env = Env('n00007846', 'n00007846_104414')
    print env.correct_bb
    print env.current_bb
    print env.step(Action(4))
    print env.step(Action(5))
    print env.step(Action(4))
    print env.step(Action(6))
    print env.step(Action(8))
