import numpy as np
import DTLearner as dt

class BagLearner(object):
    """
    This is a regression Bag Learner (i.e., a BagLearner containing Random Trees).

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """

    def __init__(self, learner=dt.DTLearner, kwargs={'leaf_size': 1}, bags=20, boost=False, verbose=False):
        """
        Constructor method
        """
        self.verbose = verbose
        learners = []
        for i in range(0, bags):
            learners.append(learner(**kwargs))
        self.learners = learners
        self.boost = boost

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
    #     todo impl

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
