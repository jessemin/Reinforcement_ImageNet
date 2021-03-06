# import custom libs
from util import Util
from bbdata import BBCollector
from env import Env
from action import Action
# import keras
from keras.models import Sequential, Model
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


# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, state_size=4096, action_size=9, action_history_size=4, batch_size=100):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.action_history_size = action_history_size
        self.action_history = deque(maxlen=self.action_history_size)
        for _ in range(self.action_history_size):
            self.action_history.append(-1)
        self.batch_size = batch_size
        self.gamma = 0.95          # discount rate
        self.epsilon = 1.0         # initial exploration rate
        self.epsilon_min = 0.1    # minimum possible epsilon value
        self.epsilon_decay = 0.99   # decaying rate for epsilon
        self.learning_rate = 5e-4
        self.model = self._build_model()
        self.util = Util()
        self.num_step = 0

    def _build_model(self):
        # 1) VGG16 for processing bb
        # load VGG16 image_model (input size: (224, 224, 3), output size: 4096)
        image_model = VGG16(weights='imagenet', include_top=True)
        # 4 history action vectors (36,)
        history_input = Input(shape=(self.action_history_size*self.action_size,),
                              name='history_input')
        # concatenate history and output from vgg16 before applying softmax
        x = concatenate([history_input, image_model.layers[-2].output])
        # 2) DQN with dropouts
        x = Dense(1024,
                  activation='relu',
                  kernel_initializer=initializers.VarianceScaling(scale=0.01),
                  bias_initializer='zeros')(x)
        x = Dropout(0.2)(x)
        x = Dense(1024,
                  activation='relu',
                  kernel_initializer=initializers.VarianceScaling(scale=0.01),
                  bias_initializer='zeros')(x)
        x = Dropout(0.2)(x)
        output = Dense(self.action_size,
                       activation='linear',
                       kernel_initializer=initializers.VarianceScaling(scale=0.01))(x)
        model = Model(inputs=[image_model.input, history_input], output=output)
        # model.compile(loss='mse',
        #               optimizer=Adam(lr=1e-6, clipnorm=1.0))
        model.compile(loss='mse',
                     optimizer=SGD(lr=self.learning_rate, clipnorm=0.5))
        return model

    def reset_action_history(self):
        for _ in range(self.action_history_size):
            self.action_history.append(-1)

    def update_action_history(self, action):
        self.action_history.append(action.get_action_type())

    def get_history_matrix(self):
        history_matrix = np.zeros((self.action_history_size,self.action_size))
        for index, a in enumerate(self.action_history):
            if a == -1:
                continue
            else:
                history_matrix[index][a] = 1
        return history_matrix.reshape(1,self.action_history_size*self.action_size)

    def add_to_replay(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def simulate_action(self, current_bb, action_type, w, h, alpha=0.2):
        x_1, y_1, x_2, y_2 = current_bb
        answer = None
        alpha_w, alpha_h = alpha * (x_2 - x_1), alpha * (y_2 - y_1)
        if action_type == 0:
            answer = (x_1+alpha_w, y_1, x_2+alpha_w, y_2)
        if action_type == 1:
            answer = (x_1-alpha_w, y_1, x_2-alpha_w, y_2)
        if action_type == 2:
            answer = (x_1, y_1+alpha_h, x_2, y_2+alpha_h)
        if action_type == 3:
            answer = (x_1, y_1-alpha_h, x_2, y_2-alpha_h)
        if action_type == 4:
            answer = (x_1-alpha_w, y_1-alpha_h, x_2+alpha_w, y_2+alpha_h)
        if action_type == 5:
            answer = (x_1+alpha_w, y_1+alpha_h, x_2-alpha_w, y_2-alpha_h)
        if action_type == 6:
            answer = (x_1, y_1+alpha_h, x_2, y_2-alpha_h)
        if action_type == 7:
            answer = (x_1+alpha_w, y_1, x_2-alpha_w, y_2)
        if action_type == 8:
            answer = current_bb
        x_1, y_1, x_2, y_2 = answer
        if x_1 < 0:
            x_1 = 0.0
        if x_2 > w:
            x_2 = w
        if y_1 < 0:
            y_1 = 0.0
        if y_2 > h:
            y_2 = h
        answer = (x_1, y_1, x_2, y_2)
        return answer

    # state is (o, h)
    def get_action(self, state, current_bb, correct_bb, w, h):
        if np.random.rand() <= self.epsilon:
            positive = []
            for i in range(9):
                if self.util.computeIOU(self.simulate_action(current_bb, i, w, h), correct_bb) > 0.5:
                    positive.append(i)
            if len(positive) == 0:
                return Action(random.randrange(self.action_size))
            while True:
                random_action = random.choice(positive)
                if self.util.computeIOU(self.simulate_action(current_bb, random_action, w, h), correct_bb) < 0.5:
                    continue
                return Action(random_action)
        act_values = self.model.predict(state)
        return Action(np.argmax(act_values[0]))

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1)
        # linear decay
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon - (self.epsilon-self.epsilon_min) * (self.num_step * 1.0 / 1000000)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


class DDQNAgent(DQNAgent):
    def __init__(self, state_size=4096, action_size=9, action_history_size=4, batch_size=100):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.action_history_size = action_history_size
        self.action_history = deque(maxlen=self.action_history_size)
        for _ in range(self.action_history_size):
            self.action_history.append(-1)
        self.batch_size = batch_size
        self.gamma = 0.95          # discount rate
        self.epsilon = 1.0         # initial exploration rate
        self.epsilon_min = 0.1    # minimum possible epsilon value
        self.epsilon_decay = 0.99   # decaying rate for epsilon
        self.learning_rate = 5e-4
        self.model = self._build_model()
        self.model2 = self._build_model2()
        self.util = Util()
        self.num_step = 0

    def _build_model2(self):
        # 1) VGG16 for processing bb
        # load VGG16 image_model (input size: (224, 224, 3), output size: 4096)
        image_model = VGG16(weights='imagenet', include_top=True)
        # 4 history action vectors (36,)
        history_input = Input(shape=(self.action_history_size*self.action_size,),
                              name='history_input2')
        # concatenate history and output from vgg16 before applying softmax
        x = concatenate([history_input, image_model.layers[-2].output])
        # 2) DQN with dropouts
        x = Dense(1024,
                  activation='relu',
                  kernel_initializer=initializers.VarianceScaling(scale=0.01),
                  bias_initializer='zeros')(x)
        x = Dropout(0.2)(x)
        x = Dense(1024,
                  activation='relu',
                  kernel_initializer=initializers.VarianceScaling(scale=0.01),
                  bias_initializer='zeros')(x)
        x = Dropout(0.2)(x)
        output = Dense(self.action_size,
                       activation='linear',
                       kernel_initializer=initializers.VarianceScaling(scale=0.01))(x)
        model = Model(inputs=[image_model.input, history_input], output=output)
        # model.compile(loss='mse',
        #               optimizer=Adam(lr=1e-6, clipnorm=1.0))
        model.compile(loss='mse',
                     optimizer=SGD(lr=self.learning_rate, clipnorm=0.5))
        return model

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            random_num = random.choice([0, 1])
            if random_num == 0:
                target = reward
                if not done:
                  target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
                target_f = self.model.predict(state)
                target_f[0][action] = target
                self.model2.fit(state, target_f, epochs=1)
            if random_num == 1:
                target = reward
                if not done:
                  target = reward + self.gamma * np.amax(self.model2.predict(next_state)[0])
                target_f = self.model2.predict(state)
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1)
        # linear decay
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon - (self.epsilon-self.epsilon_min) * (self.num_step * 1.0 / 1000000)

    # state is (o, h)
    def get_action(self, state, current_bb, correct_bb, w, h):
        if np.random.rand() <= self.epsilon:
            positive = []
            for i in range(9):
                if self.util.computeIOU(self.simulate_action(current_bb, i, w, h), correct_bb) > 0.5:
                    positive.append(i)
            if len(positive) == 0:
                return Action(random.randrange(self.action_size))
            while True:
                random_action = random.choice(positive)
                if self.util.computeIOU(self.simulate_action(current_bb, random_action, w, h), correct_bb) < 0.5:
                    continue
                return Action(random_action)
        act_values = self.model.predict(state)
        act_values2 = self.model2.predict(state)
        return Action(np.argmax(act_values[0]+act_values2[0]))


if __name__=='__main__':
    # sample code for trainng
    # 1) create a DQNAgent
    agent = DDQNAgent()
    # 2) print model summary
    print agent.model.summary()
    # 3) run episodes
    num_episodes = 1000000
    t_reward_num = 0
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
                                if len(agent.memory) >= agent.batch_size and global_step % 500 == 0:
                                    agent.model.save(os.path.join("Models", "model_"+str(episode_index)+"_"+str(global_step)+".h5"))
                                if done:
                                    break
                                if len(agent.memory) >= agent.batch_size:
                                    agent.replay()
                        agent.model.save(os.path.join("Models", "model_"+str(episode_index)+".h5"))
