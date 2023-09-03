""""""
"""Assess a betting strategy.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
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

import numpy as np
import sys

MAX_SPINS_PER_EPISODE = 1000
EPISODE_WIN_UPPER_LIMIT = 80
INFINITE_BANKROLL = sys.maxsize/2-1


def author():
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		  		 		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		  		 		  		  		    	 		 		   		 		  
    """
    return "alata6"  # replace tb34 with your Georgia Tech username.


def gtid():
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    :return: The GT ID of the student  		  	   		  		 		  		  		    	 		 		   		 		  
    :rtype: int  		  	   		  		 		  		  		    	 		 		   		 		  
    """
    return 903952381  # replace with your GT ID number


def get_spin_result(win_prob):
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    :param win_prob: The probability of winning  		  	   		  		 		  		  		    	 		 		   		 		  
    :type win_prob: float  		  	   		  		 		  		  		    	 		 		   		 		  
    :return: The result of the spin.  		  	   		  		 		  		  		    	 		 		   		 		  
    :rtype: bool  		  	   		  		 		  		  		    	 		 		   		 		  
    """
    result = False
    if np.random.random() <= win_prob:
        result = True
    return result

def get_balance_and_next_bet_as_per_strategy(current_balance,current_bet,win_prob):
    """
        current strategy -> even money bet ->  if you bet N chips and win, you keep your N chips, and you win another N chips. If you bet N chips and you lose, then those N chips are lost
        :param win_prob: The probability of winning
        :type win_prob: float
        :return: the resultant balance after the current spin, and the subsequent bet amount based on strategy
        :rtype: tuple
        """
    if(get_spin_result(win_prob)):
        return (current_balance+current_bet,1)
    else:
        return (current_balance-current_bet,current_bet*2)


def bet_episode_simulator(win_prob, episode_win_upper_limit=EPISODE_WIN_UPPER_LIMIT,
                          max_spins_per_episode=MAX_SPINS_PER_EPISODE, bankroll=INFINITE_BANKROLL):
    # initializing episode params
    result_array = np.zeros(max_spins_per_episode, dtype=np.int_)
    episode_winnings = 0
    bet_amount = 1
    spin_number = 1

    # we continue betting until the episode spin count is not exceeded, the target episode win is not achieved, and wallet balance doesnt exceed bankroll
    while spin_number <= max_spins_per_episode and episode_winnings < episode_win_upper_limit and episode_winnings > -bankroll:
        # important corner case to handle is the situation where the next bet should be $N, but you only have $M (where M<N)
        bet_amount = min(bet_amount, episode_winnings + bankroll)
        episode_winnings, bet_amount = get_balance_and_next_bet_as_per_strategy(episode_winnings, bet_amount, win_prob)
        spin_number += 1

    # if the target of $80 winnings is reached, stop betting, and allow the $80 value to persist from spin to spin
    if episode_winnings >= episode_win_upper_limit:
        result_array[spin_number:] = episode_win_upper_limit

    # once the player has lost all their money (i.e., episode_winnings reach -256), stop betting and fill that number (-256) forward
    if episode_winnings <= -bankroll:
        result_array[spin_number:] = -bankroll

    return result_array

def test_code():
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    Method to test your code  		  	   		  		 		  		  		    	 		 		   		 		  
    """
    win_prob = win_prob = 9.0/19 # 18/38 pockets are red, 18/38 pockets are black on the American Roulette wheel
    np.random.seed(gtid())  # do this only once  		  	   		  		 		  		  		    	 		 		   		 		  
    print(get_spin_result(win_prob))  # test the roulette spin  		  	   		  		 		  		  		    	 		 		   		 		  
    # add your code here to implement the experiments  		  	   		  		 		  		  		    	 		 		   		 		  


if __name__ == "__main__":
    test_code()
