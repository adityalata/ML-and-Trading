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
import matplotlib.pyplot as plt

MAX_SPINS_PER_EPISODE = 1000
EPISODE_WIN_UPPER_LIMIT = 80
INFINITE_BANKROLL = int(sys.maxsize/2-1)  # alias for INT_MAX, safety value for bankroll as bets twice of this would not be safe to handle


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


def get_balance_and_next_bet_as_per_strategy(current_winnings, current_bet, win_prob):
    """
        current strategy -> even money bet ->  if you bet N chips and win, you keep your N chips, and you win another N chips. If you bet N chips and you lose, then those N chips are lost
        :param win_prob: The probability of winning
        :type win_prob: float
        :return: the resultant winnings after the current spin, and the subsequent bet amount based on strategy
        :rtype: tuple
        """
    if (get_spin_result(win_prob)):
        return (current_winnings + current_bet, 1)
    else:
        return (current_winnings - current_bet, current_bet * 2)


def bet_episode_simulator(win_prob, episode_win_upper_limit=EPISODE_WIN_UPPER_LIMIT, max_spins_per_episode=MAX_SPINS_PER_EPISODE, bankroll=INFINITE_BANKROLL):
    # initializing episode params
    result_array = np.zeros(max_spins_per_episode + 1, dtype=np.int_)  # All winnings must be tracked by storing them in a NumPy array. You might call that array winnings where winnings[0] should be set to 0 (just before the first spin)
    episode_winnings = 0
    bet_amount = 1
    spin_number = 1

    # we continue betting until the episode spin count is not exceeded, the target episode win is not achieved, and wallet balance doesn't exceed bankroll
    while spin_number <= max_spins_per_episode and episode_winnings < episode_win_upper_limit and episode_winnings > -bankroll:
        # important corner case to handle is the situation where the next bet should be $N, but you only have $M (where M<N)
        bet_amount = min(bet_amount, episode_winnings + bankroll)
        episode_winnings, bet_amount = get_balance_and_next_bet_as_per_strategy(episode_winnings, bet_amount, win_prob)
        result_array[spin_number] = episode_winnings
        spin_number += 1

    # if the target of $80 winnings is reached, stop betting, and use the value you obtained for the rest of the spins - https://edstem.org/us/courses/43166/discussion/3332273
    if episode_winnings >= episode_win_upper_limit:
        result_array[spin_number:] = episode_winnings

    # once the player has lost all their money (i.e., episode_winnings reach -256), stop betting and fill that number (-256) forward
    if episode_winnings <= -bankroll:
        result_array[spin_number:] = -bankroll

    return result_array


def run_multiple_simulation_episodes(win_prob, number_of_simulations, episode_win_upper_limit=EPISODE_WIN_UPPER_LIMIT, max_spins_per_episode=MAX_SPINS_PER_EPISODE, bankroll=INFINITE_BANKROLL):
    episodes_result_array = np.zeros((number_of_simulations, max_spins_per_episode + 1), dtype=np.int_)  # All winnings must be tracked by storing them in a NumPy array. You might call that array winnings where winnings[0] should be set to 0 (just before the first spin)
    for simulation_number in range(number_of_simulations):
        episodes_result_array[simulation_number] = bet_episode_simulator(win_prob, episode_win_upper_limit=episode_win_upper_limit, max_spins_per_episode=max_spins_per_episode, bankroll=bankroll)
    return episodes_result_array


def produce_chart_number(win_prob, figure_number, figure_title, number_of_simulations=1, bankroll=INFINITE_BANKROLL, legend_position='lower right'):
    """
        General functionality for all plots
    """

    plt.figure(figure_number)
    plt.axis([0, 300, -256, 100])  # horizontal (X) axis must range from 0 to 300, the vertical (Y) axis must range from –256 to +100
    plt.xlabel('Spins/Bet Number per Episode')
    plt.ylabel('Total Episode Earnings in USD')
    plt.title(figure_title)

    if figure_number == 1:
        for i in range(number_of_simulations):
            plt.plot(bet_episode_simulator(win_prob), label='Run {}'.format(i))
        plt.legend(loc=legend_position, shadow=True, fontsize='medium')
    elif figure_number == 2 or figure_number == 4:
        episodes_result_array = run_multiple_simulation_episodes(win_prob, number_of_simulations, bankroll=bankroll)
        mean_winnings_for_spins = np.mean(episodes_result_array, axis=0)  # calculate for each spin
        std_winnings_for_spins = np.std(episodes_result_array, axis=0)
        std_plus = mean_winnings_for_spins + std_winnings_for_spins
        std_minus = mean_winnings_for_spins - std_winnings_for_spins
        add_std_dev_legend(mean_winnings_for_spins, std_plus, std_minus, 'Mean', legend_position)
    elif figure_number == 3 or figure_number == 5:
        episodes_result_array = run_multiple_simulation_episodes(win_prob, number_of_simulations, bankroll=bankroll)
        median_winnings_for_spins = np.median(episodes_result_array, axis=0)  # calculate for each spin
        std_winnings_for_spins = np.std(episodes_result_array, axis=0)
        std_plus = median_winnings_for_spins + std_winnings_for_spins
        std_minus = median_winnings_for_spins - std_winnings_for_spins
        add_std_dev_legend(median_winnings_for_spins, std_plus, std_minus, 'Median', legend_position)

    plt.savefig('Figure_{}.png'.format(figure_number))
    plt.close(figure_number)


def add_std_dev_legend(central_line, std_plus_line, std_minus_line, chart_label, legend_position='lower right'):
    """
        Shared functionality for plots that require a central tendency legend
    """
    plt.plot(central_line, label=chart_label)
    plt.plot(std_plus_line, label='{} + Std'.format(chart_label))
    plt.plot(std_minus_line, label='{} - Std'.format(chart_label))
    plt.legend(loc=legend_position, shadow=True, fontsize='medium')


def test_code():
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    Method to test your code  		  	   		  		 		  		  		    	 		 		   		 		  
    """
    win_prob = win_prob = 9.0 / 19  # 18/38 pockets are red, 18/38 pockets are black on the American Roulette wheel
    np.random.seed(gtid())  # do this only once  		  	   		  		 		  		  		    	 		 		   		 		  
    # print(get_spin_result(win_prob))  # test the roulette spin
    # add your code here to implement the experiments
    # run_multiple_simulation_episodes(win_prob, 2, bankroll=256)
    produce_chart_number(win_prob, 1, "10 episodes w/o bankroll", number_of_simulations=10)
    produce_chart_number(win_prob, 2, "1000 episodes w/o bankroll and mean", number_of_simulations=1000)
    produce_chart_number(win_prob, 3, "1000 episodes w/o bankroll and median", number_of_simulations=1000)
    produce_chart_number(win_prob, 4, "1000 episodes with bankroll and mean", bankroll=256, number_of_simulations=1000)
    produce_chart_number(win_prob, 5, "1000 episodes with bankroll and median", bankroll=256, number_of_simulations=1000)


if __name__ == "__main__":
    test_code()
