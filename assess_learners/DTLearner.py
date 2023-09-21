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
        self.verbose = verbose
        self.leaf_size = leaf_size  # move along, these aren't the drones you're looking for

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "alata6"  # replace tb34 with your Georgia Tech username

    def add_evidence(self, Xtrain, Ytrain):
        """
        Add training data to learner

        :param Xtrain: A set of feature values used to train the learner
        :type Xtrain: numpy.ndarray
        :param Ytrain: The value we are attempting to predict given the X data
        :type Ytrain: numpy.ndarray
        """
    #     todo impl

    def query(self, Xtest):
        """
        Estimate a set of test points given the model we built.

        :param Xtest: A numpy array with each row corresponding to a specific query.
        :type Xtest: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        #         todo impl


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
