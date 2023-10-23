""""""  		  	   		  		 		  		  		    	 		 		   		 		  
"""  		  	   		  		 		  		  		    	 		 		   		 		  
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		  		 		  		  		    	 		 		   		 		  
Note, this is NOT a correct DTLearner; Replace with your own implementation.  		  	   		  		 		  		  		    	 		 		   		 		  
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


class DTLearner(object):
    """
    This is a regression Decision Tree Learner.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """

    def __init__(self, leaf_size=1, verbose=False):
        """
        Constructor method
        """
        self.decision_tree = None
        self.verbose = verbose
        self.leaf_size = leaf_size
        if verbose:
            print("Initialized DT with leaf_size : ", leaf_size)

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "alata6"  # replace tb34 with your Georgia Tech username

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        data = np.column_stack((data_x, data_y))  # Stack 1-D arrays as columns into a 2-D array.
        self.decision_tree = self.recursively_build_tree(data)

    def recursively_build_tree(self, data):
        """
        Recursively builds and returns root of the tree

        Decision tree's Internal nodes are represented as follows:
        [ split_factor , split_value, left node (offset from current node), right node (offset from current node) ]
        Decision tree's Leaf nodes are represented as follows :
        [ None(since no split at leaf node), Y value(leaf value), NaN(no edge), NaN(no edge) ]
        The None keyword is used to define a null value, or no value at all.
        np.nan (Not a Number) allows for vectorized operations; it's a float value, while None, by definition, forces object type

        :param data: A set of feature values used to train the learner appended with The value we are attempting to predict
        :type data: numpy.ndarray
        :return: root of the tree
        """
        number_elements = data.shape[0]
        if number_elements < 1:  # safety if number_elements=0
            pass
        elif (number_elements == 1) or (np.unique(data[:, -1]).size == 1):  # base case : only 1 training data or all items have same Y value
            return np.array([[None, data[0, -1], np.nan, np.nan]])

        elif number_elements <= self.leaf_size:  # When the tree is constructed recursively, if there are leaf_size or fewer elements at the time of the recursive call, the data should be aggregated into a leaf
            return np.array([[None, np.mean(data[:, -1]), np.nan, np.nan]])

        else:
            feature_index = self.determine_best_feature(data)
            split_val = np.median(data[:, feature_index])

            left_subtree = data[data[:, feature_index] <= split_val]  # feature values LTE to the split_value
            right_subtree = data[data[:, feature_index] > split_val]  # feature values GT the split_value

            # edge case : algorithm picks the largest or smallest element as the split_val
            if left_subtree.shape[0] == number_elements or right_subtree.shape[0] == number_elements:
                if self.verbose:
                    print("All items on one side of median with median split_val", split_val, " thus merging to leaf")
                return np.array([[None, np.mean(data[:, -1]), np.nan, np.nan]])

            left_tree = self.recursively_build_tree(left_subtree)
            right_tree = self.recursively_build_tree(right_subtree)

            root_node = np.array([[feature_index, split_val, 1, left_tree.shape[0] + 1]])  # using 1 since we only need offset
            return np.row_stack((root_node, left_tree, right_tree))  # Stack arrays in sequence vertically (row wise).

    #
    def determine_best_feature(self, data):
        """
            Finds the best feature to split on and returns its column index
            We define “best feature to split on” as the feature (Xi) that has the highest absolute value correlation with Y
        """
        correlation = np.corrcoef(data, rowvar=False)  # Return Pearson product-moment correlation coefficients
        abs_corr_with_y = abs(correlation[:-1, -1])  # neg corrcoef would indicate inverse correlation
        sort_order = np.argsort(abs_corr_with_y)
        return sort_order[-1]

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        predicted_results = np.zeros(points.shape[0])
        for i in range(points.shape[0]):
            predicted_results[i] = (self.traverse_decision_tree(0, points[i, :]))  # start traversing trained Decision Tree from root - index 0
        return predicted_results

    #
    def traverse_decision_tree(self, node_index, point):
        """
        Decision tree's Internal nodes are represented as follows:
        [ split_factor , split_value, left node (offset from current node), right node (offset from current node) ]
        Decision tree's Leaf nodes are represented as follows :
        [ None(since no split at leaf node), Y value(leaf value), NaN(no edge), NaN(no edge) ]

        :param node_index:  index of current decision node
        :param point:  data_x
        :return:
        """
        node = self.decision_tree[node_index]
        if node[0] is None:  # reached leaf node
            return node[1]  # Y value
        feature_index = int(node[0])  # split_factor
        if point[feature_index] <= node[1]:  # traverse left subtree since value of point at feature index is LTE to the split value
            return self.traverse_decision_tree(node_index + int(node[2]), point)
        else:  # traverse right subtree since value of point at feature index is GT to the split value
            return self.traverse_decision_tree(node_index + int(node[3]), point)


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
