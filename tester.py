# import custom libs
from util import Util
from bbdata import BBCollector
from env import Env
from action import Action
from model import DQNAgent
# import keras
from keras.models import Sequential, Model, load_model
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dense, Activation, concatenate, Dropout, Input, Flatten, Reshape
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras import initializers
# import Python libs
import numpy as np
import os
from collections import deque
import sys
import cv2
import random
import pickle
from sklearn.model_selection import train_test_split


if __name__=='__main__':
    # sample code for trainng
    # 1) create a DQNAgent
    agent = DQNAgent()
    agent.epsilon = 0
    # 2) load model and test set
    agent.model = load_model(os.path.join("saved", "model_2_5900.h5"))
    print "Finished loading the pre-trained model..."
    test_images = pickle.load(open(os.path.join("saved", "test_list_n00007846.p"), "rb"))
    print "Finished loading the testing set..."
    wnid = 'n00007846'
    for img in test_images:
        raw_image = cv2.imread(os.path.join("Images", wnid, img))
        h, w = [_*1.0 for _ in raw_image.shape[:2]]
        image_id = img[:img.find(".")]
        # not all images of wnid have bounding box annotations
        bbCollector = BBCollector(wnid)
        bbs = bbCollector.allBBs
        if image_id not in bbs.keys():
            continue
        env = Env(wnid, image_id)
        cur_bb = env.current_bb
        correct_bb = tuple(bbs[image_id].bounding_boxes[0])
        im_state = cv2.resize(raw_image, (224, 224)).astype(np.float64)
        im_state = np.expand_dims(im_state, axis=0)
        im_state = preprocess_input(im_state)
        agent.reset_action_history()
        state = [im_state, agent.get_history_matrix()]
        while True:
            global_step += 1
            agent.num_step += 1
            action = agent.get_action(state, cur_bb, correct_bb, w, h)
            agent.update_action_history(action)
            new_bb, reward, done = env.step(action)

            cropped_image = raw_image[int(new_bb[1]):int(new_bb[3]), int(new_bb[0]):int(new_bb[2])]
            new_im_state = cv2.resize(cropped_image, (224, 224)).astype(np.float64)
            new_im_state = np.expand_dims(new_im_state, axis=0)
            new_im_state = preprocess_input(new_im_state)
            new_state = [new_im_state, agent.get_history_matrix()]
            state, cur_bb = new_state, new_bb

            if done:
                print cur_bb, " ", correct_bb, " ", image_id
                break
