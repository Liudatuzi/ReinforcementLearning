from gridworld import CliffWalk
from gym import spaces
from random import random
from gym import Env
import gym
import numpy as np
import matplotlib.pyplot as plt

class QAgent(object):
    def __init__(self,env):
        self.env=env
        self.Q={}
        self.state=None#the current observation
        self.episode=[]
        self.step=[]
        self.SarsaReward=[]
        self.QReward=[]
    def act(self,a):#perform an activity
        return self.env.step(a)
    #we use dictionary here so we need a string
    def getStateName(self,state):
        return str(state)
    def isStateInQ(self,s):
        return self.Q.get(s) is not None
    def initState(self,s,randomized=True):
        if not self.isStateInQ(s):
            self.Q[s]={}
            for action in range(self.env.action_space.n):
                if randomized is True:
                    defaultValue=0.0
                else:
                    defaultValue=0.0
                self.Q[s][action]=defaultValue
    def assertStateInQ(self,s,randomized=True):
        #if there is any state that do not exist
        if not self.isStateInQ(s):
            self.initState(s,randomized)
    def getQ(self,s,a):
        #get Q(s,a)
        self.assertStateInQ(s,randomized=True)
        return self.Q[s][a]
    def setQ(self,s,a,value):
        self.assertStateInQ(s,randomized=True)
        self.Q[s][a]=value
    def performPolicy(self,s,episode_num,use_epsilon):
        episilon=0.1
        #episilon=0

        Q_s=self.Q[s]
        str_act="unknown"
        rand_value=random()
        action=None
        if use_epsilon and rand_value<episilon:
            action=self.env.action_space.sample()
            #choose a random action
            #as the episode increases the unsure probability decreases
        else:
            str_act=max(Q_s,key=Q_s.get)
            action=int(str_act)
        return action


    def QLearning(self,gamma,alpha,max_episode_num):
        total_time,time_in_episode,num_episode=0,0,0

        while num_episode<max_episode_num:#the terminal condition
            self.state=self.env.reset()
            s0=self.getStateName(self.state)
            self.assertStateInQ(s0, randomized=True)
            self.env.render()
            a0=self.performPolicy(s0,num_episode,use_epsilon=True)
            time_in_episode=0
            reward_in_episode=0
            is_Done=False
            while not is_Done:
                s1,r1,is_Done,info=self.act(a0)
                self.env.render()
                s1=self.getStateName(s1)
                self.assertStateInQ(s1,randomized=True)
                #get A'
                a1=self.performPolicy(s1,num_episode,use_epsilon=False)
                old_q=self.getQ(s0,a0)
                new_q=self.getQ(s1,a1)
                td_target=r1+gamma*new_q
                new_q=old_q+alpha*(td_target-old_q)
                self.setQ(s0,a0,new_q)
                if(num_episode==max_episode_num):#the last episode
                    print("t:{0:>2}: s:{1}, a:{2:2}, s1:{3}". \
                          format(time_in_episode, s0, a0, s1))
                s0,a0=s1,a1
                time_in_episode+=1
                reward_in_episode+=r1
            print("QEpisode {0} takes {1} steps.".format(
                num_episode, time_in_episode))  # 显示每一个Episode花费了多少步
            self.episode.append(num_episode)
            self.QReward.append(reward_in_episode)
            total_time+=time_in_episode
            num_episode+=1
        return
    def save(self):
        #Episode=np.array(self.episode)
        Sarsa=np.array(self.QReward)
        np.save('q_learn_1.npy', Sarsa)
        #np.save('episode.npy',Episode)


'''''
    def print(self):
        x=self.episode
        y=self.SarsaReward
        y1=self.QReward
        plt.plot(x,y,color="orange",label="Sarsa")
        plt.plot(x,y1, color="brown",label="Q-learning")
        plt.title("epsilon=0")
        plt.xlabel("Episode")
        plt.ylabel("reward")
        plt.legend()
        plt.savefig("0.png")
        plt.show()
'''
def main():
    env=CliffWalk()
    agent=QAgent(env)
    print("learning...")
    #agent.QLearning(gamma=0.9,alpha=0.1,max_episode_num=800)
    agent.QLearning(gamma=0.9, alpha=0.1, max_episode_num=800)
    #agent.save()



if __name__ == '__main__':
    main()










