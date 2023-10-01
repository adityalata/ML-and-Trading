""""""  		  	   		  		 		  		  		    	 		 		   		 		  
"""  		  	   		  		 		  		  		    	 		 		   		 		  
template for generating data to fool learners (c) 2016 Tucker Balch  		  	   		  		 		  		  		    	 		 		   		 		  
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
  		  	   		  		 		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		  	   		  		 		  		  		    	 		 		   		 		  
GT User ID: tb34 (replace with your User ID)  		  	   		  		 		  		  		    	 		 		   		 		  
GT ID: 900897987 (replace with your GT ID)  		  	   		  		 		  		  		    	 		 		   		 		  
"""
import numpy as np
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
# this function should return a dataset (X and Y) that will work  		  	   		  		 		  		  		    	 		 		   		 		  
# better for linear regression than decision trees  		  	   		  		 		  		  		    	 		 		   		 		  
def best_4_lin_reg(seed=1489683273):  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		  		 		  		  		    	 		 		   		 		  
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		  		 		  		  		    	 		 		   		 		  
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    :param seed: The random seed for your data generation.  		  	   		  		 		  		  		    	 		 		   		 		  
    :type seed: int  		  	   		  		 		  		  		    	 		 		   		 		  
    :return: Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		  		 		  		  		    	 		 		   		 		  
    :rtype: numpy.ndarray  		  	   		  		 		  		  		    	 		 		   		 		  
    """
    np.random.seed(seed)
    x_rows = 10
    x_cols = 2
    rand_low = 1
    rand_high = 101
    x = np.random.randint(low=rand_low, high=rand_high, size=(x_rows, x_cols))
    """
    Linear Regression works well with data where output is a linear function of input variables
    Using the minimum number of rows(10) in training data X, linear reg is able to perform better than DT on linear Y
    """
    y = x[:, 0] * 7 + x[:, 1] * -13
    debug = False  # do not commit as True
    if debug:
        print("===========================================================================================================")
        print("best_4_lin_reg with seed ", seed)
        print("x.shape", x.shape, "x_rows", x_rows, "x_cols", x_cols, "rand_low", rand_low, "rand_high", rand_high)
        print("y.shape", y.shape)
        print("===========================================================================================================")
    return x, y
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
def best_4_dt(seed=1489683273):  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		  		 		  		  		    	 		 		   		 		  
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		  		 		  		  		    	 		 		   		 		  
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    :param seed: The random seed for your data generation.  		  	   		  		 		  		  		    	 		 		   		 		  
    :type seed: int  		  	   		  		 		  		  		    	 		 		   		 		  
    :return: Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		  		 		  		  		    	 		 		   		 		  
    :rtype: numpy.ndarray  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    np.random.seed(seed)
    x_rows = 1000
    x_cols = 10
    rand_low = 1
    rand_high = 100
    x = np.random.randint(low=rand_low, high=rand_high, size=(x_rows, x_cols))
    """
    DT works well where the target Y is a non linear function of input params X
    DT works better with higher training data rows
    """
    y = np.power(x[:, 0], 2) + np.sin(x[:, 1]) - np.tanh(x[:, 2]) + np.log(x[:, 3]) - np.sqrt(x[:, 4])
    debug = False  # do not commit as True
    if debug:
        print("===========================================================================================================")
        print("best_4_dt with seed ", seed)
        print("x.shape", x.shape, "x_rows", x_rows, "x_cols", x_cols, "rand_low", rand_low, "rand_high", rand_high)
        print("y.shape", y.shape)
        print("===========================================================================================================")
    return x, y  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
def author():  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		  		 		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    return "alata6"  # Change this to your user ID
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
# if __name__ == "__main__":
#     print("they call me Tim.")
