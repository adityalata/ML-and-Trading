""""""
"""  		  	   		  		 		  		  		    	 		 		   		 		  
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		  		 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		  		 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		  		 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		  		 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 		  		  		    	 		 		   		 		  
or edited.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		  		 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		  		 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Student Name: Aditya Lata (replace with your name)  		  	   		  		 		  		  		    	 		 		   		 		  
GT User ID: alata6 (replace with your User ID)  		  	   		  		 		  		  		    	 		 		   		 		  
GT ID: 903952381 (replace with your GT ID)  		  	   		  		 		  		  		    	 		 		   		 		  
"""

import random as rand

import numpy as np


def gtid():
    """
    :return: The GT ID of the student
    :rtype: int
    """
    return 903952381  # replace with your GT ID number


class QLearner(object):
    """
       This is a Q learner object.

       :param num_states: The number of states to consider.
       :type num_states: int
       :param num_actions: The number of actions available.
       :type num_actions: int
       :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.
       :type alpha: float
       :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.
       :type gamma: float
       :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.
       :type rar: float
       :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.
       :type radr: float
       :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.
       :type dyna: int
       :param verbose: If “verbose” is True, your code can print out information for debugging.
       :type verbose: bool
       """

    #
    def __init__(self, num_states=100, num_actions=4, alpha=0.2, gamma=0.9, rar=0.5, radr=0.99, dyna=0, verbose=False):
        """
                Constructor method  - initialization
        """
        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.random_action_rate = rar
        self.random_action_decay_rate = radr
        self.alpha = alpha
        self.gamma = gamma
        self.dyna = dyna
        self.q_table = np.zeros((num_states, num_actions))  # 2D array holding Q values
        self.experience_list = []  # list to store experience tuples
        np.random.seed(gtid())

    #
    def querysetstate(self, s):
        """
        Update the state without updating the Q-table

        :param s: The new state
        :type s: int
        :return: The selected action
        :rtype: int
        """
        self.s = s  # setting new state
        if rand.random() < self.random_action_rate:  # choosing random action
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.q_table[s])  # choosing action with best Q value

        if self.verbose:
            print('querysetstate : ', ' s = ', s, ' a = ', action)

        return action

    #
    def query(self, s_prime, r):
        """
        Update the Q table and return an action

        :param s_prime: The new state
        :type s_prime: int
        :param r: The immediate reward
        :type r: float
        :return: The selected action
        :rtype: int
        """

        self.q_table[self.s, self.a] = self.get_new_q_value(self.s, self.a, s_prime, r)
        self.experience_list.append((self.s, self.a, s_prime, r))

        if self.dyna != 0:  # hallucinate
            for _ in range(self.dyna):  # number of dyna updates for each regular update
                random_experience = self.experience_list[rand.randint(0, len(self.experience_list) - 1)]  # randomly select experience tuple
                self.q_table[random_experience[0], random_experience[1]] = self.get_new_q_value(random_experience[0], random_experience[1], random_experience[2], random_experience[3])

        if rand.random() < self.random_action_rate:  # choosing random action
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.q_table[s_prime])  # choosing action with best Q value

        self.random_action_rate *= self.random_action_decay_rate
        self.s = s_prime
        self.a = action

        if self.verbose:
            print('query : ', ' s = ', s_prime, ' a = ', action, ' r = ', r)

        return action

    #
    def get_new_q_value(self, s, a, s_prime, immediate_reward):
        """
            :return: the new Q value for given current state and action
        """
        max_q = self.q_table[s_prime, np.argmax(self.q_table[s_prime])]
        return (1 - self.alpha) * self.q_table[s, a] + self.alpha * (immediate_reward + self.gamma * max_q)

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "alata6"  # replace tb34 with your Georgia Tech username


if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
