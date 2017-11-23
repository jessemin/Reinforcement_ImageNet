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
    # 2) load model and test set
    agent.model = load_model(os.path.join("saved", "model_2_5900.h5"))
    test_images = pickle.load(open("test_list_n00007846.p", "rb"))
    wnid = 'n00007846'
    print test_images[0]
    '''
    for i in range(num_episodes):
        with open('annotation_list.dat', 'r') as f:
            for _wnid in f:
                wnid = _wnid.rstrip()
                image_dir = os.path.join("Images", str(wnid))
                bbCollector = BBCollector(wnid)
                bbs = bbCollector.allBBs
                if os.path.isdir(image_dir):
                    all_images = os.listdir(image_dir)
                    global_step = 0
                    # split into 90% for train and 10% for test
                    images, images_test = train_test_split(all_images,
                                                           test_size=0.2)
                    pickle.dump(images,
                                open(os.path.join("Models/",
                                                  "train_list_"+str(wnid)+".p"),
                                                  "wb"))
                    pickle.dump(images_test,
                                open(os.path.join("Models/",
                                                  "test_list_"+str(wnid)+".p"),
                                                  "wb"))
                    print "Total Images for wnid: {}, Train Images: {}, Test Images: {}".format(len(all_images), len(images), len(images_test))
                    for episode_index in range(num_episodes):
                        print "Episode: ", episode_index
                        for img in images:
                            raw_image = cv2.imread(os.path.join("Images", wnid, img))
                            h, w = [_*1.0 for _ in raw_image.shape[:2]]
                            image_id = img[:img.find(".")]
                            # not all images of wnid have bounding box annotations
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
                            print image_id, correct_bb
                            while True:
                                global_step += 1
                                agent.num_step += 1
                                action = agent.get_action(state, cur_bb, correct_bb, w, h)
                                agent.update_action_history(action)
                                new_bb, reward, done = env.step(action)
                                if reward == 1.0:
                                    t_reward_num += 1
                                    # if agent.epsilon > agent.epsilon_min:
                                    #     agent.epsilon *= agent.epsilon_decay
                                print image_id, new_bb, t_reward_num, global_step
                                cropped_image = raw_image[int(new_bb[1]):int(new_bb[3]), int(new_bb[0]):int(new_bb[2])]
                                new_im_state = cv2.resize(cropped_image, (224, 224)).astype(np.float64)
                                new_im_state = np.expand_dims(new_im_state, axis=0)
                                new_im_state = preprocess_input(new_im_state)
                                new_state = [new_im_state, agent.get_history_matrix()]
                                agent.remember(state, action.get_action_type(), reward, new_state, done)
                                state, cur_bb = new_state, new_bb
                                if len(agent.memory) >= agent.batch_size and global_step % 100 == 0:
                                    agent.model.save(os.path.join("Models", "model_"+str(episode_index)+"_"+str(global_step)+".h5"))
                                if done:
                                    break
                                if len(agent.memory) >= agent.batch_size:
                                    agent.replay()
                        agent.model.save(os.path.join("Models", "model_"+str(episode_index)+".h5"))
    '''
