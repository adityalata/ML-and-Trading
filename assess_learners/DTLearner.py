import numpy as np
from scipy.stats import pearsonr


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
        self.decision_tree = self.recursively_build_tree(data_x, data_y)

    def recursively_build_tree(self, data_x, data_y):
        """
        Recursively builds and returns root of the tree

        Decision tree's Internal nodes are represented as follows:
        [ split_factor , split_value, left node (offset from current node), right node (offset from current node) ]
        Decision tree's Leaf nodes are represented as follows :
        [ None(since no split at leaf node), Y value(leaf value), NaN(no edge), NaN(no edge) ]
        The None keyword is used to define a null value, or no value at all.
        np.nan (Not a Number) allows for vectorized operations; it's a float value, while None, by definition, forces object type

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        :return: root of the tree
        """
        # todo safety for empty input
        if (data_x.shape[0] == 1) or (np.unique(data_y).size == 1):  #  base case : only 1 training data or all items have same Y value
            return np.array([[None, data_y[0], np.nan, np.nan]])

        elif data.shape[0] <= self.leaf_size:  #  When the tree is constructed recursively, if there are leaf_size or fewer elements at the time of the recursive call, the data should be aggregated into a leaf
            return np.array([[None, np.mean(data_y), np.nan, np.nan]])

        else:
            feature_index = self.determine_best_feature(data_x, data_y)
            split_val = np.median(data_x[:, feature_index])  #  todo check if pop median req

            left_subtree = data_x[data_x[:, feature_index] <= split_val]  #  feature values LTE to the split_value
            right_subtree = data_x[data_x[:, feature_index] > split_val]  #  feature values GT the split_value

            #  todo check if this case can actually occur
            #  ideally all values to one side of median implies all Y are same thus should have caught above
            if left_subtree.shape[0] == data.shape[0] or right_subtree.shape[0] == data.shape[0]:
                if self.verbose:
                    print("All items on one side of median")
                return np.array([[None, np.mean(data_y), np.nan, np.nan]])

            left_tree = self.build_tree(left_subtree)
            right_tree = self.build_tree(right_subtree)

            root_node = np.array([[feature_index, split_val, 1, left_tree.shape[0] + 1]])

            return np.row_stack((root_node, left_tree, right_tree))  #  Stack arrays in sequence vertically (row wise).

    #
    def determine_best_feature(self, data_x, data_y):
        """
            Finds the best feature to split on and returns its column index
            We define “best feature to split on” as the feature (Xi) that has the highest absolute value correlation with Y
        """
        feature_correlations = np.zeros(data_x.shape[1])

        for i in range(data_x.shape[1]):
            correlation, _ = pearsonr(data_x[:, i], data_y)  # returns tuple of Pearson correlation coefficient and p-value for testing non-correlation
            feature_correlations[i] = abs(correlation)

        return np.argmax(feature_correlations)  # Returns the indices of the maximum values along an axis.

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        #         todo impl


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
